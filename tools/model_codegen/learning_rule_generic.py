"""Generic CPU and capability-based CUDA lowering for learning-rule IR."""

from __future__ import annotations

from string import Template

from .equation_ast import emit_cpp, identifiers_in
from .learning_rule_ir import CompiledLearningRule, compile_learning_rule
from .model_ir import ModelIrError
from .schema import cpp_constraint


_DENSE_CAPABILITY_ROLES = {
    "pair_stdp_v1": {
        "pre_state": ("pre_trace", None),
        "post_state": ("post_trace", None),
        "ltp_amplitude_parameter": ("max_ltp", "Max_LTP"),
        "ltd_amplitude_parameter": ("max_ltd", "Max_LTD"),
        "ltp_tau_parameter": ("ltp_tau", "LTP_tau"),
        "ltd_tau_parameter": ("ltd_tau", "LTD_tau"),
    },
    "reward_stdp_v1": {
        "pre_state": ("pre_trace", None),
        "post_state": ("post_trace", None),
        "eligibility_state": ("eligibility", None),
        "ltp_amplitude_parameter": ("max_ltp", "Max_LTP"),
        "ltd_amplitude_parameter": ("max_ltd", "Max_LTD"),
        "ltp_tau_parameter": ("ltp_tau", "LTP_tau"),
        "ltd_tau_parameter": ("ltd_tau", "LTD_tau"),
        "reward_parameter": ("reward", "RewardFactor"),
        "punishment_parameter": ("punishment", "PunishmentFactor"),
    },
}


def _literal(value: float | int, storage: str) -> str:
    return str(value) if storage == "int32" else f"{float(value)!r}f"


def _validate_identity(spec, rule: CompiledLearningRule) -> None:
    if (
        spec.get("canonical_name", rule.implementation_name) != rule.implementation_name
        or spec.get("extends")
        != {"base_class": "CustomLearningRuleModel", "contract_version": 1}
        or spec.get("api") != {"visibility": "experimental"}
        or not rule.legacy_cpu
    ):
        raise ModelIrError("invalid generated learning-rule identity/backend declaration")


def _event_cpp(rule: CompiledLearningRule, event_name: str) -> str:
    symbols = {state.name: f"s[{state.slot}]" for state in rule.states}
    symbols.update({parameter.name: parameter.name + "_" for parameter in rule.parameters})
    if event_name == "on_trigger":
        symbols["trigger_factor"] = "trigger_factor"
    statements = []
    for statement in rule.event(event_name):
        rhs = emit_cpp(statement.expression, symbols)
        if statement.target == "weight":
            statements.append(f"connection->UpdateWeight({rhs});")
        else:
            statements.append(f"{symbols[statement.target]} {statement.operator} {rhs};")
    return "\n        ".join(statements)


def render(spec):
    rule = compile_learning_rule(spec)
    _validate_identity(spec, rule)
    if rule.dense_gpu:
        _dense_symbols(rule)
    class_name = rule.implementation_name
    fields = "\n".join(
        f"    {'int' if parameter.storage == 'int32' else 'float'} "
        f"{parameter.name}_ = {_literal(parameter.default, parameter.storage)};"
        for parameter in rule.parameters
    )
    setters = "\n".join(
        f'    Read{class_name}(p, "{key}", {parameter.name}_);'
        for parameter in rule.parameters
        for key in (parameter.api_name, *parameter.aliases)
    )
    getters = "\n".join(
        f'    p["{parameter.api_name}"] = {parameter.name}_;'
        for parameter in rule.parameters
    )
    constraints = " ||\n        ".join(
        cpp_constraint(parameter, parameter.name + "_") for parameter in rule.parameters
    )
    decaying = [state for state in rule.states if state.decay_parameter is not None]
    state_arguments = "".join(f", float inv_{state.name}" for state in decaying)
    state_initializers = "".join(
        f", inv_{state.name}_(inv_{state.name})" for state in decaying
    )
    inverse_fields = "\n".join(f"    float inv_{state.name}_;" for state in decaying)
    decay_lines = "\n".join(
        f"        StateDecay(index, {state.slot}, exp(-delta * inv_{state.name}_));"
        for state in decaying
    )
    state_ctor_values = "".join(
        f", 1.0f / {state.decay_parameter}_" for state in decaying
    )
    clear_field = (
        f"    bool clear_ = {'true' if rule.clear_default else 'false'};"
        if rule.has_trigger else ""
    )
    clear_setter = (
        f'    Read{class_name}(p, "{rule.clear_parameter}", clear_);'
        if rule.has_trigger else ""
    )
    clear_getter = (
        f'    p["{rule.clear_parameter}"] = clear_;'
        if rule.has_trigger else ""
    )
    common = dict(
        class_name=class_name,
        fields=fields,
        clear_field=clear_field,
        setters=setters,
        clear_setter=clear_setter,
        getters=getters,
        clear_getter=clear_getter,
        constraints=constraints,
        state_class=class_name + "State",
        state_count=len(rule.states),
        state_arguments=state_arguments,
        state_initializers=state_initializers,
        inverse_fields=inverse_fields,
        decay_lines=decay_lines,
        state_ctor_values=state_ctor_values,
        has_post="true" if rule.has_post else "false",
        has_trigger="true" if rule.has_trigger else "false",
    )
    header = HEADER.substitute(**common)
    if rule.has_trigger:
        source = TRIGGER_SOURCE.substitute(
            on_pre=_event_cpp(rule, "on_pre"),
            on_post=_event_cpp(rule, "on_post"),
            on_trigger=_event_cpp(rule, "on_trigger"),
            reward_member=rule.reward_parameter + "_",
            punishment_member=rule.punishment_parameter + "_",
            punishment_type=rule.punishment_type,
            clear_slot=next(state.slot for state in rule.states if state.name == rule.clear_state),
            **common,
        )
    else:
        source = PAIR_SOURCE.substitute(
            on_pre=_event_cpp(rule, "on_pre"),
            on_post=_event_cpp(rule, "on_post") if rule.has_post else "",
            **common,
        )
    return header, source


def _dense_symbols(rule: CompiledLearningRule) -> dict[str, str]:
    roles = dict(rule.dense_roles)
    contract = _DENSE_CAPABILITY_ROLES.get(rule.dense_capability)
    if contract is None:
        raise ModelIrError("no Dense lowering for this learning-rule capability")
    expected = {role: local for role, (local, _) in contract.items()}
    if set(roles) != set(expected):
        raise ModelIrError(f"{rule.dense_capability} requires a complete Dense role map")
    state_names = {state.name for state in rule.states}
    parameter_map = {parameter.name: parameter for parameter in rule.parameters}
    parameter_names = set(parameter_map)
    symbols = {}
    for role, local_name in expected.items():
        declared_name = roles[role]
        valid_names = state_names if role.endswith("_state") else parameter_names
        if declared_name not in valid_names:
            raise ModelIrError(f"Dense role {role!r} references an unknown symbol")
        symbols[declared_name] = local_name
        if not role.endswith("_state"):
            expected_api = contract[role][1]
            parameter = parameter_map[declared_name]
            if expected_api not in (parameter.api_name, *parameter.aliases):
                raise ModelIrError(
                    f"Dense role {role!r} must expose API key {expected_api!r}"
                )
    mapped_states = {
        roles[role] for role in expected if role.endswith("_state")
    }
    if mapped_states != state_names:
        raise ModelIrError(
            f"{rule.dense_capability} state layout must match its declared Dense roles"
        )
    if (
        rule.dense_capability == "reward_stdp_v1"
        and rule.clear_parameter != "ClearEligibilityAfterTrigger"
    ):
        raise ModelIrError(
            "reward_stdp_v1 must expose ClearEligibilityAfterTrigger"
        )
    symbols["weight"] = "weight"
    symbols["trigger_factor"] = "trigger_factor"
    return symbols


def dense_default_injections(rule: CompiledLearningRule) -> str:
    """Normalize public parameter aliases and defaults to the fixed Dense ABI keys."""
    contract = _DENSE_CAPABILITY_ROLES.get(rule.dense_capability)
    if contract is None:
        raise ModelIrError("no Dense lowering for this learning-rule capability")
    roles = dict(rule.dense_roles)
    parameter_map = {parameter.name: parameter for parameter in rule.parameters}
    lines = []
    for role, (_, abi_key) in contract.items():
        if abi_key is None:
            continue
        parameter = parameter_map[roles[role]]
        public_keys = tuple(dict.fromkeys((parameter.api_name, *parameter.aliases)))
        alternatives = [key for key in public_keys if key != abi_key]
        lines.append(
            f'        if (resolved.RuleParameter.find("{abi_key}") == '
            "resolved.RuleParameter.end()) {"
        )
        for index, key in enumerate(alternatives):
            keyword = "if" if index == 0 else "else if"
            lines.extend([
                f'            {keyword} (resolved.RuleParameter.find("{key}") != '
                "resolved.RuleParameter.end()) {",
                f'                resolved.RuleParameter["{abi_key}"] = '
                f'resolved.RuleParameter["{key}"];',
                "            }",
            ])
        default_line = (
            f'resolved.RuleParameter["{abi_key}"] = '
            f'{_literal(parameter.default, parameter.storage)};'
        )
        if alternatives:
            lines.extend(["            else {", f"                {default_line}", "            }"])
        else:
            lines.append(f"            {default_line}")
        lines.append("        }")
    return "\n".join(lines)


def _cuda_event(rule: CompiledLearningRule, event_name: str, symbols) -> str:
    lines = []
    for statement in rule.event(event_name):
        if not identifiers_in(statement.expression) <= set(symbols):
            raise ModelIrError(f"{event_name} uses a symbol unavailable in the Dense capability")
        target = symbols.get(statement.target)
        if target is None:
            raise ModelIrError(f"{event_name} target is unavailable in Dense")
        lines.append(
            f"    {target} {statement.operator} "
            f"{emit_cpp(statement.expression, symbols, 'cuda')};"
        )
    return "\n".join(lines)


def render_cuda(spec):
    rule = compile_learning_rule(spec)
    _validate_identity(spec, rule)
    if not rule.dense_gpu:
        raise ModelIrError("Dense CUDA was not enabled for this learning rule")
    symbols = _dense_symbols(rule)
    from .learning_rule_emitter import CUDA_HEADER, PAIR_CUDA_HEADER

    prefix = "ApplyDense" + rule.implementation_name
    if rule.dense_capability == "pair_stdp_v1":
        return PAIR_CUDA_HEADER.substitute(
            pre_name=prefix + "PreDeviceEntry",
            post_name=prefix + "PostDeviceEntry",
            on_pre=_cuda_event(rule, "on_pre", symbols),
            on_post=_cuda_event(rule, "on_post", symbols),
        )
    return CUDA_HEADER.substitute(
        pre_name=prefix + "PreDeviceEntry",
        post_name=prefix + "PostDeviceEntry",
        trigger_name=prefix + "TriggerDeviceEntry",
        on_pre=_cuda_event(rule, "on_pre", symbols),
        on_post=_cuda_event(rule, "on_post", symbols),
        on_trigger=_cuda_event(rule, "on_trigger", symbols),
    )


HEADER = Template('''#pragma once
#include "source_file_realtime_v1_async/LearningRule/inc/Custom/CustomLearningRuleModel.h"

class $class_name final : public CustomLearningRuleModel {
public:
    explicit $class_name(std::map<std::string, boost::any> parameters);
    void InitState(int connections, int neurons, float base_dt) override;
    void ApplyPreSynaticSpike(Interconnections* connection, int time, Simulation* simulation) override;
    void ApplyPostSynaticSpike(Neuron* neuron, int time, Simulation* simulation) override;
    std::map<std::string, boost::any> GetParameters() override;
private:
$fields
$clear_field
};
''')


STATE_SOURCE = '''#include "neuronbridge_codegen/$class_name.h"
#include "source_file_realtime_v1_async/LearningRule/inc/SynapseState.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include <cmath>
#include <stdexcept>

namespace {
class $state_class final : public SynapseState {
public:
    $state_class(int count$state_arguments)
        : SynapseState(count, $state_count)$state_initializers {}
    void SetUpdate(int index, int time, float dt) override {
        const float delta = (time - LastUpdate[index]) * dt;
$decay_lines
        LastUpdate[index] = time;
    }
    void ApplyPresynapticSpike(int) override {}
    void ApplyPostsynapticSpike(int) override {}
private:
$inverse_fields
};
template<class T> void Read$class_name(std::map<std::string, boost::any>& p, const char* key, T& value) {
    auto found = p.find(key);
    if (found != p.end()) { value = boost::any_cast<T>(found->second); p.erase(found); }
}
}

$class_name::$class_name(std::map<std::string, boost::any> p)
    : CustomLearningRuleModel($has_post, $has_trigger) {
$setters
$clear_setter
    if (!p.empty() || $constraints)
        throw std::invalid_argument("invalid $class_name parameters");
}

void $class_name::InitState(int count, int, float dt) {
    if (State || count < 0 || !std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("invalid $class_name state initialization");
    State = new $state_class(count$state_ctor_values);
}
'''


PAIR_SOURCE = Template(STATE_SOURCE + '''
void $class_name::ApplyPreSynaticSpike(
    Interconnections* connection, int time, Simulation* simulation) {
    auto* state = static_cast<$state_class*>(State);
    const int index = connection->LearningRuleIndex_withPost;
    state->SetUpdate(index, time, simulation->basetimesteps);
    float* s = state->StateValue + index * $state_count;
    $on_pre
}

void $class_name::ApplyPostSynaticSpike(Neuron* neuron, int time, Simulation* simulation) {
    auto* state = static_cast<$state_class*>(State);
    ForEachPostConnection(neuron, [&](Interconnections* connection, int index) {
        state->SetUpdate(index, time, simulation->basetimesteps);
        float* s = state->StateValue + index * $state_count;
        $on_post
    });
}

std::map<std::string, boost::any> $class_name::GetParameters() {
    std::map<std::string, boost::any> p;
$getters
    return p;
}
''')


TRIGGER_SOURCE = Template(STATE_SOURCE + '''
void $class_name::ApplyPreSynaticSpike(Interconnections* connection, int time, Simulation* simulation) {
    auto* state = static_cast<$state_class*>(State);
    if (!connection->TriggerLearning) {
        int index = connection->LearningRuleIndex_withPostAndTrigger;
        state->SetUpdate(index, time, simulation->basetimesteps);
        float* s = state->StateValue + index * $state_count;
        $on_pre
        return;
    }
    float trigger_factor = connection->type == $punishment_type
        ? $punishment_member : $reward_member;
    ForEachPostTriggerConnection(connection->TargetNeuron, [&](Interconnections* connection, int index) {
        state->SetUpdate(index, time, simulation->basetimesteps);
        float* s = state->StateValue + index * $state_count;
        $on_trigger
        if (clear_) s[$clear_slot] = 0.0f;
    });
}

void $class_name::ApplyPostSynaticSpike(Neuron* neuron, int time, Simulation* simulation) {
    auto* state = static_cast<$state_class*>(State);
    ForEachPostTriggerConnection(neuron, [&](Interconnections*, int index) {
        state->SetUpdate(index, time, simulation->basetimesteps);
        float* s = state->StateValue + index * $state_count;
        $on_post
    });
}

std::map<std::string, boost::any> $class_name::GetParameters() {
    std::map<std::string, boost::any> p;
$getters
$clear_getter
    return p;
}
''')
