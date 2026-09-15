"""Generic maintained time-driven-neuron emitter for supported runtime capabilities."""
from pathlib import Path
from string import Template
from .equation_ast import emit_comparison_cpp, emit_cpp
from .model_ir import compile_time_driven_neuron, ModelIrError
from .legacy_cpu_emitter import emit_descriptor_header
from .legacy_gpu_neuron_emitter import render_legacy_gpu
from .schema import cpp_constraint

def _literal(value: float | int, storage: str) -> str:
    return str(value) if storage == "int32" else f"{float(value)!r}f"


def _statement_cpp(statement, symbols, target="cpu"):
    return (
        f"{symbols[statement.target]} {statement.operator} "
        f"{emit_cpp(statement.expression, symbols, target)};"
    )


def render(spec):
    model = compile_time_driven_neuron(spec)
    integration = spec.get("integration", {})
    if (spec.get("canonical_name", model.implementation_name) != model.implementation_name or
        spec.get("api") != {"visibility": "experimental"} or
        not model.legacy_cpu):
        raise ModelIrError("invalid generated neuron identity/backend declaration")
    if (spec.get("extends", {}).get("base_class") != "CustomTimeDrivenNeuronModel" or
        integration.get("method") != "forward_euler" or
        integration.get("step_policy") != "legacy_stride_squared"):
        raise ModelIrError("unsupported time-driven neuron runtime capability")
    parameters = {parameter.name: parameter for parameter in model.parameters}
    symbols = {name: name + "_" for name in parameters}
    symbols.update({name: f"s[{i}]" for i, name in enumerate(model.state_names)})
    differential = []
    for update, ode in zip(model.state_updates, model.odes):
        if update.method != "forward_euler":
            continue
        rhs = emit_cpp(ode.expression, symbols)
        if update.unless_refractory:
            rhs = (
                f"StateVector->LastSpike[i] > {model.refractory_parameter}_ "
                f"? {rhs} : 0.0f"
            )
        differential.append(f"        d[{update.slot}] = {rhs};")
    time_updates = [
        f"        s[{update.slot}] *= {update.name}_decay_;"
        for update in model.state_updates if update.method == "exact_exponential"
    ]
    condition = emit_comparison_cpp(model.spike_condition, symbols)
    reset = " ".join(
        _statement_cpp(statement, symbols) for statement in model.reset_statements
    )
    fields = "\n".join(
        f"    {'int' if parameter.storage == 'int32' else 'float'} "
        f"{parameter.name}_ = {_literal(parameter.default, parameter.storage)};"
        for parameter in model.parameters
    )
    decay_fields = "\n".join(
        f"    float {update.name}_decay_ = 1.0f;"
        for update in model.state_updates if update.method == "exact_exponential"
    )
    setters = "\n".join(
        f'    Read{model.implementation_name}(p, "{key}", {parameter.name}_);'
        for parameter in model.parameters
        for key in (parameter.api_name, *parameter.aliases)
    )
    getters = "\n".join(
        f'    p["{parameter.api_name}"] = {parameter.name}_;'
        for parameter in model.parameters
    )
    equal = " && ".join(
        f"{parameter.name}_ == other->{parameter.name}_"
        for parameter in model.parameters
    )
    validations = " ||\n        ".join(
        cpp_constraint(parameter, parameter.name + "_")
        for parameter in model.parameters
    )
    decays = "\n".join(
        f"    {update.name}_decay_ = std::exp(-effective_dt * "
        f"(1.0f / {update.decay_parameter}_));"
        for update in model.state_updates if update.method == "exact_exponential"
    )
    initial_symbols = {name: name + "_" for name in parameters}
    initial = ", ".join(
        f"{emit_cpp(update.initial, initial_symbols)} + init_[{update.slot}]"
        for update in model.state_updates
    )
    state_connections = sorted({
        binding.connection_type for binding in model.input_bindings
        if binding.delivery == "AddToState"
    })
    conductance_check = " || ".join(
        f"inter->type == {connection_type}" for connection_type in state_connections
    ) or "false"
    header = HEADER.substitute(class_name=model.implementation_name,
                               differential_count=model.differential_state_count,
                               differential="\n".join(differential),
                               time_updates="\n".join(time_updates),
                               condition=condition, reset=reset,
                               state_count=len(model.state_names),
                               fields=fields, decay_fields=decay_fields)
    source = SOURCE.substitute(class_name=model.implementation_name,
                               setters=setters, getters=getters, equal=equal,
                               validations=validations, decays=decays,
                               initial=initial,
                               conductance_check=conductance_check)
    descriptor = emit_descriptor_header(
        model, differential_state_count=model.differential_state_count)
    return header, source, descriptor


def render_dense(spec):
    """Lower a validated Dense capability to host metadata and CUDA code."""
    render(spec)
    model = compile_time_driven_neuron(spec)
    if not model.dense_gpu or model.dense_capability != "conductance_lif_v1":
        raise ModelIrError("no Dense lowering for this neuron capability")
    roles = dict(model.dense_roles)
    required_roles = {
        "voltage_state", "excitatory_state", "inhibitory_state", "current_state",
        "rest_parameter", "reset_parameter", "threshold_parameter",
        "resistance_parameter", "membrane_tau_parameter",
        "excitatory_tau_parameter", "inhibitory_tau_parameter",
        "excitatory_reversal_parameter", "inhibitory_reversal_parameter",
        "refractory_parameter",
    }
    if set(roles) != required_roles:
        raise ModelIrError("conductance_lif_v1 requires a complete Dense role map")
    states = set(model.state_names)
    parameter_map = {parameter.name: parameter for parameter in model.parameters}
    parameters = set(parameter_map)
    for role, value in roles.items():
        expected = states if role.endswith("_state") else parameters
        if value not in expected:
            raise ModelIrError(f"Dense role {role!r} references an unknown symbol")
    role_values = {
        "voltage_state": "v[index]",
        "excitatory_state": "next_g_exc",
        "inhibitory_state": "next_g_inh",
        "current_state": "input_current",
        "rest_parameter": "v_rest[index]",
        "reset_parameter": "v_reset[index]",
        "threshold_parameter": "v_threshold[index]",
        "resistance_parameter": "r[index]",
        "membrane_tau_parameter": "tau_m[index]",
        "excitatory_tau_parameter": "tau_exc[index]",
        "inhibitory_tau_parameter": "tau_inh[index]",
        "excitatory_reversal_parameter": "e_exc[index]",
        "inhibitory_reversal_parameter": "e_inh[index]",
        "refractory_parameter": "refractory_steps[index]",
    }
    symbols = {roles[role]: value for role, value in role_values.items()}
    derivative = emit_cpp(model.odes[model.voltage_slot].expression, symbols, "cuda")
    event_symbols = dict(symbols)
    event_symbols[roles["voltage_state"]] = "next_v"
    spike_condition = emit_comparison_cpp(model.spike_condition, event_symbols, "cuda")
    reset_targets = {statement.target for statement in model.reset_statements}
    if reset_targets != {roles["voltage_state"]}:
        raise ModelIrError(
            "conductance_lif_v1 Dense lowering only supports resetting the voltage state"
        )
    spike_reset = " ".join(
        _statement_cpp(statement, event_symbols, "cuda")
        for statement in model.reset_statements
    )
    channel_by_kind = {
        "excitatory_conductance": "PendingChannel::ExcitatoryConductance",
        "inhibitory_conductance": "PendingChannel::InhibitoryConductance",
        "current": "PendingChannel::Current",
    }
    role_kind = {
        "excitatory_state": "excitatory_conductance",
        "inhibitory_state": "inhibitory_conductance",
        "current_state": "current",
    }
    bindings_by_name = {binding.name: binding for binding in model.input_bindings}
    for role, kind in role_kind.items():
        binding = bindings_by_name.get(roles[role])
        if binding is None or binding.kind != kind:
            raise ModelIrError(
                f"conductance_lif_v1 role {role!r} requires an input of kind {kind!r}"
            )
    input_channels = "\n".join(
        f'            {{{channel_by_kind[binding.kind]}, "{binding.name}", '
        f'{"false" if binding.kind == "current" else "true"}}},'
        for binding in model.input_bindings
    )
    spike_effects = "\n".join(
        f"            {{{binding.connection_type}, {channel_by_kind[binding.kind]}, 1.0f}},"
        for binding in model.input_bindings
    )
    def dense_parameter(role):
        return parameter_map[roles[role]]

    def dense_default(role):
        return f"{float(dense_parameter(role).default)!r}f"

    def dense_keys(role, field_name):
        parameter = dense_parameter(role)
        keys = dict.fromkeys((parameter.api_name, *parameter.aliases, field_name))
        return ", ".join(f'"{key}"' for key in keys)

    parameter_keys = []
    for role, field_name in (
        ("membrane_tau_parameter", "tau_m_ms"),
        ("excitatory_tau_parameter", "tau_exc_ms"),
        ("inhibitory_tau_parameter", "tau_inh_ms"),
        ("rest_parameter", "v_rest"),
        ("reset_parameter", "v_reset"),
        ("threshold_parameter", "v_threshold"),
        ("resistance_parameter", "r"),
        ("excitatory_reversal_parameter", "e_exc"),
        ("inhibitory_reversal_parameter", "e_inh"),
        ("refractory_parameter", "t_ref"),
    ):
        parameter = dense_parameter(role)
        parameter_keys.extend((parameter.api_name, *parameter.aliases, field_name))
    parameter_keys = ", ".join(
        f'"{key}"' for key in dict.fromkeys(parameter_keys)
    )
    dense_substitutions = {
        "class_name": model.implementation_name,
        "parameter_keys": parameter_keys,
        "input_channels": input_channels,
        "spike_effects": spike_effects,
    }
    for role, field_name in (
        ("membrane_tau_parameter", "tau_m_ms"),
        ("excitatory_tau_parameter", "tau_exc_ms"),
        ("inhibitory_tau_parameter", "tau_inh_ms"),
        ("rest_parameter", "v_rest"),
        ("reset_parameter", "v_reset"),
        ("threshold_parameter", "v_threshold"),
        ("resistance_parameter", "r"),
        ("excitatory_reversal_parameter", "e_exc"),
        ("inhibitory_reversal_parameter", "e_inh"),
        ("refractory_parameter", "t_ref"),
    ):
        prefix = role.removesuffix("_parameter")
        dense_substitutions[prefix + "_default"] = dense_default(role)
        dense_substitutions[prefix + "_keys"] = dense_keys(role, field_name)
    return (DENSE_HEADER.substitute(**dense_substitutions),
            DENSE_CUDA_HEADER.substitute(class_name=model.implementation_name,
                                         derivative=derivative,
                                         spike_condition=spike_condition,
                                         spike_reset=spike_reset))

def emit_runtime(root: Path, specs):
    outputs = {}
    includes, gpu_includes, dense_includes, device_includes = [], [], [], []
    registry, gpu_registry, gpu_aggregate, dense_models, catalog, aggregate = [], [], [], [], [], []
    for spec in specs:
        model = compile_time_driven_neuron(spec)
        class_name = model.implementation_name
        header, source, descriptor = render(spec)
        outputs[f"include/neuronbridge_codegen/{class_name}.h"] = header
        outputs[f"include/neuronbridge_codegen/{class_name}Descriptor.h"] = descriptor
        outputs[f"src/{class_name}.cpp"] = source
        includes.append(f'#include "neuronbridge_codegen/{class_name}.h"')
        registry.append(f'NPGR_CUSTOM_LEGACY_NEURON("{class_name}", {class_name})')
        if model.legacy_gpu:
            gpu_header, gpu_source = render_legacy_gpu(spec)
            gpu_name = model.legacy_gpu_implementation_name
            outputs[f"include/neuronbridge_codegen/{gpu_name}.cuh"] = gpu_header
            outputs[f"src/{gpu_name}.cu"] = gpu_source
            gpu_includes.append(f'#include "neuronbridge_codegen/{gpu_name}.cuh"')
            gpu_registry.append(
                f'NPGR_CUSTOM_LEGACY_GPU_NEURON("{gpu_name}", {gpu_name})')
            gpu_aggregate.append(f'#include "{gpu_name}.cu"')
        if model.dense_gpu:
            dense_header, dense_cuda = render_dense(spec)
            outputs[f"include/neuronbridge_codegen/Dense{class_name}.h"] = dense_header
            outputs[f"include/neuronbridge_codegen/Dense{class_name}DeviceUpdate.cuh"] = dense_cuda
            dense_includes.append(f'#include "neuronbridge_codegen/Dense{class_name}.h"')
            device_includes.append(f'#include "neuronbridge_codegen/Dense{class_name}DeviceUpdate.cuh"')
            dense_models.append(
                f"NPGR_DENSE_NEURON_MODEL(kDense{class_name}ModelId, "
                f"{model.dense_factory_model_id}, Dense{class_name}, "
                f"UpdateDense{class_name}DeviceEntry)")
        backend_entries = [
            f'{{NeuronBackend::LegacyCpu, "{class_name}", true, true}}'
        ]
        if model.dense_gpu:
            backend_entries.append(
                f'{{NeuronBackend::DenseGpu, "{class_name}", true, true}}'
            )
        if model.legacy_gpu:
            backend_entries.append(
                f'{{NeuronBackend::LegacyGpu, "{model.legacy_gpu_implementation_name}", true, true}}'
            )
        aliases = (
            f'{{"{model.legacy_gpu_implementation_name}"}}'
            if model.legacy_gpu else '{}'
        )
        catalog.append(
            f'entries->push_back({{"{class_name}", {aliases}, '
            f'{{{", ".join(backend_entries)}}}, '
            'NeuronModelCategory::Core, NeuronModelRole::OrdinaryCore, false});')
        aggregate.append(f'#include "{class_name}.cpp"')
    outputs.update({
        "include/neuronbridge_codegen/CustomGeneratedModels.h": "#pragma once\n" + "\n".join(includes) + "\n",
        "include/neuronbridge_codegen/CustomGeneratedLegacyGpuModels.cuh": "#pragma once\n" + "\n".join(gpu_includes) + "\n",
        "include/neuronbridge_codegen/CustomGeneratedDenseNeuronModels.h": "#pragma once\n" + "\n".join(dense_includes) + "\n",
        "include/neuronbridge_codegen/CustomGeneratedDenseNeuronDeviceUpdates.cuh": "#pragma once\n" + "\n".join(device_includes) + "\n",
        "include/neuronbridge_codegen/CustomLegacyNeuronRegistry.inc": "\n".join(registry) + "\n",
        "include/neuronbridge_codegen/CustomLegacyGpuNeuronRegistry.inc": "\n".join(gpu_registry) + "\n",
        "include/neuronbridge_codegen/CustomDenseNeuronModelList.inc": "\n".join(dense_models) + "\n",
        "include/neuronbridge_codegen/CustomNeuronCatalog.inc": "\n".join(catalog) + "\n",
        "src/GeneratedNeuronModels.cpp": "\n".join(aggregate) + "\n",
        "src/GeneratedLegacyGpuNeuronModels.cu": "\n".join(gpu_aggregate) + "\n",
        "src/GeneratedNeuronCatalogEntries.cpp": '#include "neuron_model/GeneratedNeuronCatalog.h"\nnamespace npgr {\nvoid RegisterGeneratedNeuronCatalogEntries(std::vector<NeuronModelCatalogEntry>* entries) {\n#include "neuronbridge_codegen/CustomNeuronCatalog.inc"\n}\n}\n',
    })
    for relative, content in outputs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

HEADER = Template('''// Generated time-driven neuron. Do not edit.
#pragma once
#include "source_file_realtime_v1_async/NeuralModel/inc/Custom/CustomTimeDrivenNeuronModel.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include <array>
#include <cmath>
class ${class_name} final : public CustomTimeDrivenNeuronModel {
public:
    static constexpr int N_DifferentialStates = $differential_count;
    explicit ${class_name}(int stride);
    ~${class_name}() override;
    void SetParameters(std::map<std::string, boost::any> p, float base_dt);
    std::map<std::string, boost::any> getParameters() override;
    bool compare(NeuronModel* model) override;
    void InitStateVector(int count, int gpu) override;
    void CheckType(Interconnections* inter) override;
    void UpdateState(int index, int time, Simulation* simulation) override;
    void CaculateDifferentialEquation(float* s, float* d, int i) {
$differential
    }
    void CaculateTimeDependentEquation(float* s, int, float) {
        if (conductance_enabled_) {
$time_updates
        }
    }
    void CaculateSpike(float, float* s, int i) {
        if ($condition) { $reset MarkSpike(i); }
    }
private:
$fields
$decay_fields
    bool conductance_enabled_ = false;
    std::array<float, $state_count> init_{}, sigma_{};
};
''')

SOURCE = Template('''// Generated time-driven neuron. Do not edit.
#include "neuronbridge_codegen/${class_name}.h"
#include "neuronbridge_codegen/${class_name}Descriptor.h"
#include "source_file_realtime_v1_async/Intergration/inc/FixStep/ForwardEulerMethod.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include <cmath>
#include <stdexcept>
namespace {
template<class T> void Read${class_name}(std::map<std::string, boost::any>& p, const char* key, T& value) {
    auto it = p.find(key); if (it != p.end()) { value = boost::any_cast<T>(it->second); p.erase(it); }
}
}
${class_name}::${class_name}(int stride)
    : CustomTimeDrivenNeuronModel(npgr_generated::k${class_name}Descriptor, stride) {}
${class_name}::~${class_name}() {
    // IntegerationMethod has a nonvirtual destructor in the legacy ABI.
    delete static_cast<ForwardEulerMethod<${class_name}>*>(integrationMethod);
    integrationMethod = nullptr;
}
void ${class_name}::SetParameters(std::map<std::string, boost::any> p, float base_dt) {
    if (StateVector->NumberofNeuron) throw std::logic_error("configure before initialization");
$setters
    Read${class_name}(p, "random_mu", init_); Read${class_name}(p, "random_sigma", sigma_);
    ModelDescription method; method.ModelName = "ForwardEulerMethod";
    Read${class_name}(p, "int_method", method);
    if (!p.empty() || method.ModelName != "ForwardEulerMethod") throw std::invalid_argument("unsupported generated parameter/integrator");
    if (!(base_dt > 0) || !std::isfinite(base_dt) || getTimestepSize() <= 0 ||
        $validations)
        throw std::invalid_argument("invalid generated neuron parameters");
    float step = base_dt * getTimestepSize();
    method.ModelParameter["step"] = step;
    delete static_cast<ForwardEulerMethod<${class_name}>*>(integrationMethod);
    integrationMethod = nullptr;
    integrationMethod = ForwardEulerMethod<${class_name}>::CreateIntegerationMethod(method.ModelParameter, this);
    const float effective_dt = step * getTimestepSize();
$decays
}
void ${class_name}::InitStateVector(int count, int) {
    if (!integrationMethod || StateVector->NumberofNeuron) throw std::logic_error("invalid generated initialization order");
    float initial[] = {$initial};
    InitializeStateVector(count, initial, sigma_.data());
}
void ${class_name}::CheckType(Interconnections* inter) {
    CustomTimeDrivenNeuronModel::CheckType(inter);
    if ($conductance_check) conductance_enabled_ = true;
}
void ${class_name}::UpdateState(int, int time, Simulation* simulation) {
    if (!integrationMethod) throw std::logic_error("generated integrator missing");
    StateVector->NumberofSpike = 0;
    integrationMethod->CaculateIncreament(simulation, time);
}
std::map<std::string, boost::any> ${class_name}::getParameters() {
    std::map<std::string, boost::any> p;
$getters
    p["random_mu"] = init_; p["random_sigma"] = sigma_;
    if (integrationMethod) { ModelDescription m; m.ModelName = "ForwardEulerMethod";
        m.ModelParameter = integrationMethod->getParameters(); p["int_method"] = m; }
    return p;
}
bool ${class_name}::compare(NeuronModel* model) {
    auto other = dynamic_cast<${class_name}*>(model);
    return other && TimeDrivenModel::compare(model) && $equal &&
        t_ref_ == other->t_ref_ && init_ == other->init_ && sigma_ == other->sigma_;
}
''')


DENSE_HEADER = Template('''// Generated dense LIF host metadata. Do not edit.
#pragma once
#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

enum Dense${class_name}FieldSlot {
    kDense${class_name}V = 0,
    kDense${class_name}GExc,
    kDense${class_name}GInh,
    kDense${class_name}Fired,
    kDense${class_name}StepsSinceLastSpike,
    kDense${class_name}VRest,
    kDense${class_name}VReset,
    kDense${class_name}VThreshold,
    kDense${class_name}R,
    kDense${class_name}TauM,
    kDense${class_name}EExc,
    kDense${class_name}EInh,
    kDense${class_name}ExcDecay,
    kDense${class_name}InhDecay,
    kDense${class_name}RefractorySteps,
    kDense${class_name}FieldCount,
};

class Dense${class_name} final : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override {
        return DenseNeuronModelFactory::kDense${class_name}ModelId;
    }
    const char* CanonicalName() const override { return "${class_name}"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "${class_name}";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"v", DenseFieldStorage::Float32, DenseFieldRole::State, $rest_default},
            {"gexc", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"ginh", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
            {"steps_since_last_spike", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 10000},
            {"tau_m_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $membrane_tau_default},
            {"tau_exc_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $excitatory_tau_default},
            {"tau_inh_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $inhibitory_tau_default},
            {"v_rest", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $rest_default},
            {"v_reset", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $reset_default},
            {"v_threshold", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $threshold_default},
            {"r", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $resistance_default},
            {"e_exc", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $excitatory_reversal_default},
            {"e_inh", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $inhibitory_reversal_default},
            {"t_ref", DenseFieldStorage::Float32, DenseFieldRole::Parameter, $refractory_default},
            {"exc_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"inh_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"refractory_steps", DenseFieldStorage::Int32, DenseFieldRole::Derived, 0.0f, 0},
        };
    }
    int FieldSlotCount() const override { return kDense${class_name}FieldCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kDense${class_name}V, "v"},
            {kDense${class_name}GExc, "gexc"},
            {kDense${class_name}GInh, "ginh"},
            {kDense${class_name}Fired, "fired"},
            {kDense${class_name}StepsSinceLastSpike, "steps_since_last_spike"},
            {kDense${class_name}VRest, "v_rest"},
            {kDense${class_name}VReset, "v_reset"},
            {kDense${class_name}VThreshold, "v_threshold"},
            {kDense${class_name}R, "r"},
            {kDense${class_name}TauM, "tau_m_ms"},
            {kDense${class_name}EExc, "e_exc"},
            {kDense${class_name}EInh, "e_inh"},
            {kDense${class_name}ExcDecay, "exc_decay"},
            {kDense${class_name}InhDecay, "inh_decay"},
            {kDense${class_name}RefractorySteps, "refractory_steps"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        return {
$input_channels
        };
    }
    std::vector<DenseSpikeEffectBinding> SpikeEffects() const override {
        return {
$spike_effects
        };
    }
    bool FillInitialFieldValues(DenseNeuronHostFieldTable* table,
                                const DenseNeuronModelSpec& spec,
                                std::string*) const override {
        FillFloatField(table, spec, "tau_m_ms", {$membrane_tau_keys}, $membrane_tau_default);
        FillFloatField(table, spec, "tau_exc_ms", {$excitatory_tau_keys}, $excitatory_tau_default);
        FillFloatField(table, spec, "tau_inh_ms", {$inhibitory_tau_keys}, $inhibitory_tau_default);
        FillFloatField(table, spec, "v_rest", {$rest_keys}, $rest_default);
        FillFloatField(table, spec, "v_reset", {$reset_keys}, $reset_default);
        FillFloatField(table, spec, "v_threshold", {$threshold_keys}, $threshold_default);
        FillFloatField(table, spec, "r", {$resistance_keys}, $resistance_default);
        FillFloatField(table, spec, "e_exc", {$excitatory_reversal_keys}, $excitatory_reversal_default);
        FillFloatField(table, spec, "e_inh", {$inhibitory_reversal_keys}, $inhibitory_reversal_default);
        FillFloatField(table, spec, "t_ref", {$refractory_keys}, $refractory_default);
        return ResetStateFields(table, spec, nullptr);
    }
    bool BuildDerivedFields(DenseNeuronHostFieldTable* table,
                            const DenseNeuronModelSpec& spec,
                            float dt_ms,
                            std::string* reason) const override {
        const float* tau_m = FloatField(*table, "tau_m_ms");
        const float* tau_exc = FloatField(*table, "tau_exc_ms");
        const float* tau_inh = FloatField(*table, "tau_inh_ms");
        const float* t_ref = FloatField(*table, "t_ref");
        float* exc_decay = FloatField(table, "exc_decay");
        float* inh_decay = FloatField(table, "inh_decay");
        int* refractory_steps = IntField(table, "refractory_steps");
        for (int offset = 0; offset < spec.range.count; ++offset) {
            const int index = spec.range.begin + offset;
            if (tau_m[index] <= 0.0f || tau_exc[index] <= 0.0f || tau_inh[index] <= 0.0f) {
                if (reason != nullptr) *reason = "generated dense LIF time constants must be positive";
                return false;
            }
            exc_decay[index] = std::exp(-dt_ms / tau_exc[index]);
            inh_decay[index] = std::exp(-dt_ms / tau_inh[index]);
            refractory_steps[index] = static_cast<int>(t_ref[index]);
        }
        return true;
    }
    bool ResetStateFields(DenseNeuronHostFieldTable* table,
                          const DenseNeuronModelSpec& spec,
                          std::string*) const override {
        const float* v_rest = FloatField(*table, "v_rest");
        float* v = FloatField(table, "v");
        for (int offset = 0; offset < spec.range.count; ++offset) {
            const int index = spec.range.begin + offset;
            v[index] = v_rest[index];
        }
        FillFloatConstant(table, spec, "gexc", 0.0f);
        FillFloatConstant(table, spec, "ginh", 0.0f);
        FillByteConstant(table, spec, "fired", 0);
        FillIntConstant(table, spec, "steps_since_last_spike", 10000);
        return true;
    }
protected:
    std::vector<const char*> ParameterKeys() const override {
        return {$parameter_keys};
    }
};

}  // namespace
}  // namespace npgr
''')


DENSE_CUDA_HEADER = Template('''// Generated dense LIF CUDA update. Do not edit.
#pragma once
#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"
#include "neuronbridge_codegen/Dense${class_name}.h"

namespace npgr {

__device__ inline unsigned char UpdateDense${class_name}DeviceEntry(
    DenseNeuronDeviceFieldTable fields, const int* field_ids, int index, int,
    DensePendingChannelDeviceView pending, float dt_ms, int* fired_field_id) {
    *fired_field_id = field_ids[kDense${class_name}Fired];
    const float arrival_exc = ReadAndClearPendingChannel(
        pending, PendingChannel::ExcitatoryConductance);
    const float arrival_inh = ReadAndClearPendingChannel(
        pending, PendingChannel::InhibitoryConductance);
    const float input_current = ReadAndClearPendingChannel(pending, PendingChannel::Current);
    float* v = DeviceFloatField(fields, field_ids[kDense${class_name}V]);
    float* g_exc = DeviceFloatField(fields, field_ids[kDense${class_name}GExc]);
    float* g_inh = DeviceFloatField(fields, field_ids[kDense${class_name}GInh]);
    int* steps = DeviceIntField(fields, field_ids[kDense${class_name}StepsSinceLastSpike]);
    const float* v_rest = DeviceFloatField(fields, field_ids[kDense${class_name}VRest]);
    const float* v_reset = DeviceFloatField(fields, field_ids[kDense${class_name}VReset]);
    const float* v_threshold = DeviceFloatField(fields, field_ids[kDense${class_name}VThreshold]);
    const float* r = DeviceFloatField(fields, field_ids[kDense${class_name}R]);
    const float* tau_m = DeviceFloatField(fields, field_ids[kDense${class_name}TauM]);
    const float* e_exc = DeviceFloatField(fields, field_ids[kDense${class_name}EExc]);
    const float* e_inh = DeviceFloatField(fields, field_ids[kDense${class_name}EInh]);
    const float* exc_decay = DeviceFloatField(fields, field_ids[kDense${class_name}ExcDecay]);
    const float* inh_decay = DeviceFloatField(fields, field_ids[kDense${class_name}InhDecay]);
    const int* refractory_steps = DeviceIntField(fields, field_ids[kDense${class_name}RefractorySteps]);

    const float next_g_exc = DenseDecayAndAdd(g_exc[index], exc_decay[index], arrival_exc);
    const float next_g_inh = DenseDecayAndAdd(g_inh[index], inh_decay[index], arrival_inh);
    float next_v = v[index];
    if (steps[index] > refractory_steps[index]) {
        next_v = DenseLifEulerStep(v[index], dt_ms, $derivative);
    }
    int next_steps = steps[index] + 1;
    unsigned char did_fire = 0;
    if ($spike_condition) {
        did_fire = 1;
        $spike_reset
        next_steps = 0;
    }
    v[index] = next_v;
    g_exc[index] = next_g_exc;
    g_inh[index] = next_g_inh;
    steps[index] = next_steps;
    return did_fire;
}

}  // namespace npgr
''')
