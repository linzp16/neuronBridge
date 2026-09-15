"""Model-independent IR compiler for event-driven generated learning rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .equation_ast import (
    Binary,
    EventStatement,
    Identifier,
    Number,
    Ode,
    Unary,
    identifiers_in,
    parse_equations,
    parse_event_statement,
)
from .model_ir import ModelIrError
from .schema import ParameterSpec, SchemaError, compile_parameters, require_identifier


@dataclass(frozen=True)
class LearningState:
    name: str
    slot: int
    ode: Ode
    method: str
    decay_parameter: str | None


@dataclass(frozen=True)
class CompiledLearningRule:
    implementation_name: str
    canonical_name: str
    variant_id: int
    dense_factory_model_id: int
    parameters: tuple[ParameterSpec, ...]
    states: tuple[LearningState, ...]
    events: tuple[tuple[str, tuple[EventStatement, ...]], ...]
    has_post: bool
    has_trigger: bool
    catalog_category: str
    clear_parameter: str | None
    clear_default: bool
    clear_state: str | None
    reward_parameter: str | None
    punishment_parameter: str | None
    punishment_type: int
    dense_capability: str | None
    dense_roles: tuple[tuple[str, str], ...]
    legacy_cpu: bool
    dense_gpu: bool

    def event(self, name: str) -> tuple[EventStatement, ...]:
        return dict(self.events).get(name, ())


def _exact_decay_parameter(ode: Ode) -> str | None:
    expression = ode.expression
    if not isinstance(expression, Binary) or expression.operator != "/":
        return None
    if not isinstance(expression.left, Unary) or expression.left.operator != "-":
        return None
    if not isinstance(expression.left.operand, Identifier) or expression.left.operand.name != ode.state:
        return None
    return expression.right.name if isinstance(expression.right, Identifier) else None


def compile_learning_rule(spec: dict[str, Any]) -> CompiledLearningRule:
    try:
        if spec.get("kind") != "custom_learning_rule":
            raise ModelIrError("learning-rule IR requires custom_learning_rule")
        class_name = require_identifier(spec.get("implementation_name"), "implementation_name")
        canonical_name = spec.get("canonical_name", class_name)
        if not isinstance(canonical_name, str) or not canonical_name:
            raise ModelIrError("canonical_name must be a string")
        parameters = compile_parameters(spec.get("parameters"))
        parameter_names = {parameter.name for parameter in parameters}
        odes = parse_equations(str(spec["equations"]))
        state_names = tuple(ode.state for ode in odes)
        if len(state_names) != len(set(state_names)):
            raise ModelIrError("learning state names must be unique")

        integration = spec.get("integration")
        if not isinstance(integration, dict) or integration.get("method") != "exact_exponential":
            raise ModelIrError("learning rules currently require exact_exponential integration")
        if integration.get("time") != "arrival_ticks_times_base_dt":
            raise ModelIrError("unsupported learning-rule time contract")
        states: list[LearningState] = []
        for slot, ode in enumerate(odes):
            decay_parameter = _exact_decay_parameter(ode)
            if decay_parameter is not None:
                if decay_parameter not in parameter_names:
                    raise ModelIrError(f"state {ode.state!r} decay parameter is not declared")
                method = "exact_exponential"
            elif isinstance(ode.expression, Number) and float(ode.expression.value) == 0.0:
                method = "hold"
            else:
                raise ModelIrError(
                    f"state {ode.state!r} must be -state / parameter or a zero derivative"
                )
            states.append(LearningState(ode.state, slot, ode, method, decay_parameter))

        raw_events = spec.get("events")
        if not isinstance(raw_events, dict) or "on_pre" not in raw_events:
            raise ModelIrError("learning rules require events.on_pre")
        if set(raw_events) - {"on_pre", "on_post", "on_trigger"}:
            raise ModelIrError("unsupported learning-rule event")
        has_post = "on_post" in raw_events
        has_trigger = "on_trigger" in raw_events
        symbols = set(state_names) | parameter_names | {"weight"}
        events: list[tuple[str, tuple[EventStatement, ...]]] = []
        for event_name in ("on_pre", "on_post", "on_trigger"):
            if event_name not in raw_events:
                continue
            block = raw_events[event_name]
            if not isinstance(block, list) or not block:
                raise ModelIrError(f"{event_name} must be a non-empty statement array")
            event_symbols = symbols | ({"trigger_factor"} if event_name == "on_trigger" else set())
            statements: list[EventStatement] = []
            for source in block:
                if not isinstance(source, str):
                    raise ModelIrError("event statements must be strings")
                statement = parse_event_statement(source)
                if statement.target not in set(state_names) | {"weight"}:
                    raise ModelIrError(f"unknown event target {statement.target!r}")
                if statement.target == "weight" and statement.operator != "+=":
                    raise ModelIrError("legacy weight updates currently require +=")
                if not identifiers_in(statement.expression) <= event_symbols:
                    raise ModelIrError(f"{event_name} expression uses unknown identifiers")
                statements.append(statement)
            events.append((event_name, tuple(statements)))

        trigger = spec.get("trigger")
        clear_parameter = clear_state = reward_parameter = punishment_parameter = None
        clear_default = False
        punishment_type = 1
        if has_trigger:
            if not isinstance(trigger, dict):
                raise ModelIrError("on_trigger requires trigger metadata")
            clear_parameter = trigger.get("clear_parameter")
            clear_state = trigger.get("clear_state")
            reward_parameter = trigger.get("reward_parameter")
            punishment_parameter = trigger.get("punishment_parameter")
            clear_default = trigger.get("clear_default")
            punishment_type = trigger.get("punishment_type", 1)
            if (
                not isinstance(clear_parameter, str)
                or clear_state not in state_names
                or reward_parameter not in parameter_names
                or punishment_parameter not in parameter_names
                or not isinstance(clear_default, bool)
                or not isinstance(punishment_type, int)
                or trigger.get("weight_bounds") != "connection"
            ):
                raise ModelIrError("invalid trigger metadata")
        elif trigger is not None:
            raise ModelIrError("trigger metadata requires an on_trigger event")

        runtime = spec.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ModelIrError("runtime must be an object")
        dense = runtime.get("dense", {})
        if not isinstance(dense, dict):
            raise ModelIrError("runtime.dense must be an object")
        roles = dense.get("roles", {})
        if not isinstance(roles, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in roles.items()
        ):
            raise ModelIrError("runtime.dense.roles must map strings to strings")
        backends = spec.get("backends")
        if not isinstance(backends, dict):
            raise ModelIrError("backends declaration is required")
        return CompiledLearningRule(
            class_name,
            canonical_name,
            int(spec["variant_id"]),
            int(spec["dense_factory_model_id"]),
            parameters,
            tuple(states),
            tuple(events),
            has_post,
            has_trigger,
            str(spec.get("catalog_category", "other")),
            clear_parameter,
            clear_default,
            clear_state,
            reward_parameter,
            punishment_parameter,
            punishment_type,
            dense.get("capability"),
            tuple(sorted(roles.items())),
            backends.get("legacy_cpu") is True,
            backends.get("dense_gpu") is True,
        )
    except SchemaError as error:
        raise ModelIrError(str(error)) from error
