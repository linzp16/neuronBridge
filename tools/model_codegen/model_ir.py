"""Validated, model-independent IR for generated time-driven neurons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .equation_ast import (
    Binary,
    Comparison,
    EventStatement,
    Identifier,
    Number,
    Ode,
    Unary,
    identifiers_in,
    parse_comparison,
    parse_equations,
    parse_event_statement,
    parse_expression,
)
from .schema import ParameterSpec, SchemaError, compile_parameters, require_identifier


class ModelIrError(ValueError):
    """A supported model spec cannot be lowered to the maintained IR."""


_INPUT_TYPES = {
    "excitatory_conductance": (0, "AddToState"),
    "inhibitory_conductance": (1, "AddToState"),
    "current": (3, "AddToCurrentAccumulator"),
}


@dataclass(frozen=True)
class InputBinding:
    name: str
    state_slot: int
    connection_type: int
    delivery: str
    kind: str


@dataclass(frozen=True)
class StateUpdate:
    name: str
    slot: int
    method: str
    initial: object
    unless_refractory: bool
    decay_parameter: str | None


@dataclass(frozen=True)
class CompiledNeuronSpec:
    implementation_name: str
    canonical_name: str
    variant_id: int
    dense_factory_model_id: int
    parameters: tuple[ParameterSpec, ...]
    state_names: tuple[str, ...]
    odes: tuple[Ode, ...]
    state_updates: tuple[StateUpdate, ...]
    input_bindings: tuple[InputBinding, ...]
    voltage_slot: int
    spike_condition: Comparison
    reset_statements: tuple[EventStatement, ...]
    differential_state_count: int
    refractory_parameter: str | None
    dense_capability: str | None
    dense_roles: tuple[tuple[str, str], ...]
    legacy_cpu: bool
    legacy_gpu: bool
    legacy_gpu_capability: str | None
    legacy_gpu_implementation_name: str | None
    legacy_gpu_roles: tuple[tuple[str, str], ...]
    dense_gpu: bool


def _exact_decay_parameter(ode: Ode) -> str | None:
    expression = ode.expression
    if not isinstance(expression, Binary) or expression.operator != "/":
        return None
    if not isinstance(expression.left, Unary) or expression.left.operator != "-":
        return None
    if not isinstance(expression.left.operand, Identifier) or expression.left.operand.name != ode.state:
        return None
    if not isinstance(expression.right, Identifier):
        return None
    return expression.right.name


def _is_zero(expression: object) -> bool:
    return isinstance(expression, Number) and float(expression.value) == 0.0


def compile_time_driven_neuron(spec: dict[str, Any]) -> CompiledNeuronSpec:
    try:
        if spec.get("kind") != "custom_time_driven_neuron":
            raise ModelIrError("IR compiler only accepts custom_time_driven_neuron")
        implementation_name = require_identifier(spec.get("implementation_name"), "implementation_name")
        canonical_name = spec.get("canonical_name", implementation_name)
        if not isinstance(canonical_name, str) or not canonical_name:
            raise ModelIrError("canonical_name must be a string")
        parameters = compile_parameters(spec.get("parameters"))
        parameter_names = {parameter.name for parameter in parameters}
        equations = parse_equations(str(spec["equations"]))
        states = tuple(ode.state for ode in equations)
        if len(states) != len(set(states)):
            raise ModelIrError("differential state names must be unique")
        equation_symbols = set(states) | parameter_names
        for ode in equations:
            if not identifiers_in(ode.expression) <= equation_symbols:
                raise ModelIrError(f"equation for {ode.state!r} uses unknown identifiers")

        runtime = spec.get("runtime")
        if not isinstance(runtime, dict):
            raise ModelIrError("runtime declaration is required")
        raw_updates = runtime.get("state_update")
        if not isinstance(raw_updates, dict) or set(raw_updates) != set(states):
            raise ModelIrError("runtime.state_update must describe every equation state")
        updates: list[StateUpdate] = []
        for slot, ode in enumerate(equations):
            declaration = raw_updates[ode.state]
            if not isinstance(declaration, dict):
                raise ModelIrError(f"state update {ode.state!r} must be an object")
            if set(declaration) - {"method", "initial", "unless_refractory"}:
                raise ModelIrError(f"state update {ode.state!r} has unsupported fields")
            method = declaration.get("method")
            if method not in {"forward_euler", "exact_exponential", "hold"}:
                raise ModelIrError(f"unsupported state update method {method!r}")
            initial_source = declaration.get("initial", "0")
            if not isinstance(initial_source, str):
                raise ModelIrError(f"state {ode.state!r} initial value must be an expression")
            initial = parse_expression(initial_source)
            if not identifiers_in(initial) <= parameter_names:
                raise ModelIrError(f"state {ode.state!r} initial value uses unknown identifiers")
            decay_parameter = None
            if method == "exact_exponential":
                decay_parameter = _exact_decay_parameter(ode)
                if decay_parameter not in parameter_names:
                    raise ModelIrError(
                        f"state {ode.state!r} exact exponential requires -state / parameter"
                    )
            elif method == "hold" and not _is_zero(ode.expression):
                raise ModelIrError(f"held state {ode.state!r} requires a zero derivative")
            updates.append(StateUpdate(
                ode.state,
                slot,
                method,
                initial,
                bool(declaration.get("unless_refractory", False)),
                decay_parameter,
            ))
        differential_count = sum(update.method == "forward_euler" for update in updates)
        if any(update.method != "forward_euler" for update in updates[:differential_count]):
            raise ModelIrError("forward Euler states must be contiguous at the start of equations")

        raw_inputs = spec.get("inputs", [])
        if not isinstance(raw_inputs, list):
            raise ModelIrError("inputs must be an array")
        bindings: list[InputBinding] = []
        for item in raw_inputs:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ModelIrError("each input requires a name")
            kind = item.get("kind")
            if kind not in _INPUT_TYPES:
                raise ModelIrError(f"unsupported input kind {kind!r}")
            if item["name"] not in states:
                raise ModelIrError(f"input {item['name']!r} must name a declared state")
            connection_type, delivery = _INPUT_TYPES[kind]
            bindings.append(InputBinding(
                item["name"], states.index(item["name"]), connection_type, delivery, kind
            ))

        events = spec.get("events")
        if not isinstance(events, dict) or not isinstance(events.get("spike"), dict):
            raise ModelIrError("events.spike is required")
        spike = events["spike"]
        condition_source = spike.get("threshold")
        reset = spike.get("reset")
        if not isinstance(condition_source, str) or not isinstance(reset, dict) or not reset:
            raise ModelIrError("events.spike requires threshold and non-empty reset")
        condition = parse_comparison(condition_source)
        symbols = set(states) | parameter_names
        if not (identifiers_in(condition.left) | identifiers_in(condition.right)) <= symbols:
            raise ModelIrError("spike threshold uses unknown identifiers")
        reset_statements: list[EventStatement] = []
        for target, source in reset.items():
            if target not in states or not isinstance(source, str):
                raise ModelIrError("spike reset targets states with expression values")
            statement = parse_event_statement(f"{target} = {source}")
            if not identifiers_in(statement.expression) <= symbols:
                raise ModelIrError("spike reset uses unknown identifiers")
            reset_statements.append(statement)

        voltage_state = runtime.get("voltage_state")
        if voltage_state not in states:
            raise ModelIrError("runtime.voltage_state must name a state")
        refractory_parameter = runtime.get("refractory_parameter")
        if refractory_parameter is not None and refractory_parameter not in parameter_names:
            raise ModelIrError("runtime.refractory_parameter must name a parameter")
        if any(update.unless_refractory for update in updates) and refractory_parameter is None:
            raise ModelIrError("refractory state updates require runtime.refractory_parameter")

        backends = spec.get("backends")
        if not isinstance(backends, dict):
            raise ModelIrError("backends declaration is required")
        dense = runtime.get("dense", {})
        if not isinstance(dense, dict):
            raise ModelIrError("runtime.dense must be an object")
        roles = dense.get("roles", {})
        if not isinstance(roles, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in roles.items()):
            raise ModelIrError("runtime.dense.roles must map strings to strings")
        legacy_gpu = runtime.get("legacy_gpu", {})
        if not isinstance(legacy_gpu, dict):
            raise ModelIrError("runtime.legacy_gpu must be an object")
        legacy_gpu_roles = legacy_gpu.get("roles", {})
        if not isinstance(legacy_gpu_roles, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in legacy_gpu_roles.items()
        ):
            raise ModelIrError("runtime.legacy_gpu.roles must map strings to strings")
        legacy_gpu_name = legacy_gpu.get("implementation_name")
        if legacy_gpu_name is not None:
            legacy_gpu_name = require_identifier(
                legacy_gpu_name, "runtime.legacy_gpu.implementation_name")
        return CompiledNeuronSpec(
            implementation_name,
            canonical_name,
            int(spec["variant_id"]),
            int(spec["dense_factory_model_id"]),
            parameters,
            states,
            equations,
            tuple(updates),
            tuple(bindings),
            states.index(voltage_state),
            condition,
            tuple(reset_statements),
            differential_count,
            refractory_parameter,
            dense.get("capability"),
            tuple(sorted(roles.items())),
            backends.get("legacy_cpu") is True,
            backends.get("legacy_gpu") is True,
            legacy_gpu.get("capability"),
            legacy_gpu_name,
            tuple(sorted(legacy_gpu_roles.items())),
            backends.get("dense_gpu") is True,
        )
    except SchemaError as error:
        raise ModelIrError(str(error)) from error
