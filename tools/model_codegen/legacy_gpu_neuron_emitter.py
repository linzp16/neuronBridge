"""CUDA emitter for generated neurons hosted by the legacy main network."""

from string import Template

from .equation_ast import emit_comparison_cpp, emit_cpp
from .model_ir import ModelIrError, compile_time_driven_neuron
from .schema import cpp_constraint


REQUIRED_ROLES = {
    "voltage_state", "excitatory_state", "inhibitory_state", "current_state",
    "rest_parameter", "reset_parameter", "threshold_parameter",
    "resistance_parameter", "membrane_tau_parameter",
    "excitatory_tau_parameter", "inhibitory_tau_parameter",
    "excitatory_reversal_parameter", "inhibitory_reversal_parameter",
    "refractory_parameter",
}


def _literal(value, storage):
    return str(value) if storage == "int32" else f"{float(value)!r}f"


def render_legacy_gpu(spec):
    model = compile_time_driven_neuron(spec)
    if (not model.legacy_gpu or
            model.legacy_gpu_capability != "conductance_lif_v1" or
            not model.legacy_gpu_implementation_name):
        raise ModelIrError("no legacy GPU lowering for this neuron capability")
    roles = dict(model.legacy_gpu_roles)
    if set(roles) != REQUIRED_ROLES:
        raise ModelIrError("conductance_lif_v1 requires a complete legacy GPU role map")
    states = set(model.state_names)
    parameter_map = {parameter.name: parameter for parameter in model.parameters}
    for role, value in roles.items():
        expected = states if role.endswith("_state") else set(parameter_map)
        if value not in expected:
            raise ModelIrError(f"legacy GPU role {role!r} references an unknown symbol")

    class_name = model.legacy_gpu_implementation_name
    state_slots = {name: index for index, name in enumerate(model.state_names)}
    parameter_symbols = {name: name + "_" for name in parameter_map}
    device_symbols = {
        name: f"state[{slot} * neuron_count + index]"
        for name, slot in state_slots.items()
    }
    device_symbols.update(parameter_symbols)
    voltage_update = model.state_updates[model.voltage_slot]
    voltage_ode = model.odes[model.voltage_slot]
    if voltage_update.method != "forward_euler":
        raise ModelIrError("legacy GPU conductance LIF requires Euler voltage state")
    derivative = emit_cpp(voltage_ode.expression, device_symbols, "cuda")
    if voltage_update.unless_refractory:
        derivative = (
            f"last_spike[index] > {roles['refractory_parameter']}_ "
            f"? ({derivative}) : 0.0f"
        )
    decays = []
    for update in model.state_updates:
        if update.method == "exact_exponential":
            decays.append(
                f"    state[{update.slot} * neuron_count + index] *= "
                f"expf(-dt / {update.decay_parameter}_);"
            )
    event_symbols = dict(device_symbols)
    condition = emit_comparison_cpp(model.spike_condition, event_symbols, "cuda")
    reset = []
    for statement in model.reset_statements:
        target = device_symbols[statement.target]
        expression = emit_cpp(statement.expression, event_symbols, "cuda")
        reset.append(f"        {target} {statement.operator} {expression};")

    fields = "\n".join(
        f"    {'int' if p.storage == 'int32' else 'float'} {p.name}_ = "
        f"{_literal(p.default, p.storage)};"
        for p in model.parameters
    )
    setters = []
    for parameter in model.parameters:
        for key in (parameter.api_name, *parameter.aliases):
            setters.append(
                f'    ReadGeneratedGpu(p, "{key}", {parameter.name}_);'
            )
    getters = "\n".join(
        f'    p["{parameter.api_name}"] = {parameter.name}_;'
        for parameter in model.parameters
    )
    comparisons = " && ".join(
        f"{p.name}_ == other->{p.name}_" for p in model.parameters
    )
    validations = " ||\n        ".join(
        cpp_constraint(parameter, parameter.name + "_")
        for parameter in model.parameters
    )
    initial_symbols = dict(parameter_symbols)
    initial = ", ".join(
        emit_cpp(update.initial, initial_symbols, "cpu") + f" + init_[{update.slot}]"
        for update in model.state_updates
    )
    substitutions = dict(
        class_name=class_name,
        state_count=len(model.state_names),
        voltage_slot=model.voltage_slot,
        exc_slot=state_slots[roles["excitatory_state"]],
        inh_slot=state_slots[roles["inhibitory_state"]],
        current_slot=state_slots[roles["current_state"]],
        fields=fields,
        setters="\n".join(setters),
        getters=getters,
        comparisons=comparisons,
        validations=validations,
        initial=initial,
        derivative=derivative,
        decays="\n".join(decays),
        condition=condition,
        reset="\n".join(reset),
        parameter_arguments=", ".join(p.name + "_" for p in model.parameters),
        kernel_parameters=", ".join(
            ("int " if p.storage == "int32" else "float ") + p.name + "_"
            for p in model.parameters
        ),
    )
    return GPU_HEADER.substitute(**substitutions), GPU_SOURCE.substitute(**substitutions)


GPU_HEADER = Template(r'''// Generated legacy main-network CUDA neuron. Do not edit.
#pragma once
#include "source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenNeuronModelGPU_Interface.cuh"
#include <array>
class CurrentSynapse;
class ${class_name} final : public TimeDrivenNeuronModelGPU_Interface {
public:
    explicit ${class_name}(int stride);
    ~${class_name}() override;
    float timestepdouble = 0.0f;
    void SetParameters(std::map<std::string, boost::any> p, float base_dt);
    std::map<std::string, boost::any> getParameters() override;
    bool compare(NeuronModel* model) override;
    Neuron_State_Vector* InitState() override;
    InternalSpike* ProcessSpike(Interconnections* inter, int time) override;
    void ProcessCurrent(Interconnections* inter, Neuron* target, float current) override;
    void UpdateState(int index, int time, Simulation* simulation) override;
    void InitStateVector(int count, int gpu_index) override;
    void InitializeInputCurrentSynapseStructure() override;
    void InitializeClassGPU2(int) override {}
    void InitializeVectorNeuronState_GPU2() override {}
    void DestroyGPUModel() override {}
    void CheckType(Interconnections* inter) override;
    int getV_index() override { return $voltage_slot; }
    int get_NumberOfState() override { return $state_count; }
    NeuronModelType getNeuronModelType() override { return NEURAL_LAYER; }
private:
$fields
    std::array<float, $state_count> init_{}, sigma_{};
    CurrentSynapse* current_synapse_ = nullptr;
    bool excited_ = false;
    bool inhibitory_ = false;
    bool current_enabled_ = false;
};
''')


GPU_SOURCE = Template(r'''// Generated legacy main-network CUDA neuron. Do not edit.
#include "neuronbridge_codegen/${class_name}.cuh"
#include "source_file_realtime_v1_async/Current/inc/CurrentSynapse.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector_Interface.cuh"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "source_file_realtime_v1_async/error/cudaerror.h"
#include "source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
template<class T> void ReadGeneratedGpu(std::map<std::string, boost::any>& p,
                                        const char* key, T& value) {
    auto it = p.find(key);
    if (it != p.end()) { value = boost::any_cast<T>(it->second); p.erase(it); }
}

__global__ void Update${class_name}(
    float* state, float* aux, int* last_update, int* last_spike, bool* fired,
    int neuron_count, int time, float dt, bool excited, bool inhibitory,
    bool current_enabled, $kernel_parameters) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < neuron_count) {
        if (excited) state[$exc_slot * neuron_count + index] += aux[index];
        if (inhibitory) state[$inh_slot * neuron_count + index] += aux[neuron_count + index];
        if (current_enabled) state[$current_slot * neuron_count + index] = aux[2 * neuron_count + index];
        last_update[index] = time;
        state[$voltage_slot * neuron_count + index] += dt * ($derivative);
$decays
        ++last_spike[index];
        if ($condition) {
$reset
            last_spike[index] = 0;
            fired[index] = true;
        }
        index += blockDim.x * gridDim.x;
    }
}
}  // namespace

${class_name}::${class_name}(int stride) : TimeDrivenNeuronModelGPU_Interface(stride) {
    StateVector = new Neuron_State_Vector_Interface($state_count);
    setModelName("${class_name}");
}

${class_name}::~${class_name}() {
    delete current_synapse_;
    current_synapse_ = nullptr;
}

void ${class_name}::SetParameters(std::map<std::string, boost::any> p, float base_dt) {
$setters
    ReadGeneratedGpu(p, "random_mu", init_);
    ReadGeneratedGpu(p, "random_sigma", sigma_);
    auto method = p.find("int_method");
    if (method != p.end()) {
        ModelDescription description = boost::any_cast<ModelDescription>(method->second);
        if (description.ModelName != "ForwardEulerMethod")
            throw std::invalid_argument("generated legacy GPU supports ForwardEulerMethod only");
        p.erase(method);
    }
    p.erase("basetimestep");
    if (!p.empty() || !(base_dt > 0.0f) || !std::isfinite(base_dt) ||
        $validations)
        throw std::invalid_argument("invalid generated legacy GPU parameters");
    timestepdouble = base_dt;
}

std::map<std::string, boost::any> ${class_name}::getParameters() {
    std::map<std::string, boost::any> p;
$getters
    p["basetimestep"] = timestepdouble;
    p["random_mu"] = init_;
    p["random_sigma"] = sigma_;
    return p;
}

bool ${class_name}::compare(NeuronModel* model) {
    auto* other = dynamic_cast<${class_name}*>(model);
    return other && TimeDrivenNeuronModelGPU_Interface::compare(model) &&
        $comparisons && init_ == other->init_ && sigma_ == other->sigma_;
}

Neuron_State_Vector* ${class_name}::InitState() { return StateVector; }

InternalSpike* ${class_name}::ProcessSpike(Interconnections* inter, int) {
    if (inter->type == 0 && excited_)
        State_GPU->AuxStateCPU[inter->TargetNeuronModelIndex] += inter->weight;
    else if (inter->type == 1 && inhibitory_)
        State_GPU->AuxStateCPU[State_GPU->NumberofNeuron + inter->TargetNeuronModelIndex] += inter->weight;
    return nullptr;
}

void ${class_name}::ProcessCurrent(Interconnections* inter, Neuron* target, float current) {
    current_synapse_->SetInputCurrentPerSynapse(target->index_in_NeuronModel,
                                                inter->subindex_type, current);
    State_GPU->AuxStateCPU[2 * State_GPU->NumberofNeuron + target->index_in_NeuronModel] =
        current_synapse_->GetTotalInputCurrentPerNeuron(target->index_in_NeuronModel);
}

void ${class_name}::UpdateState(int, int time, Simulation*) {
    const int count = State_GPU->NumberofNeuron;
    const size_t aux_bytes = sizeof(float) * count * 3;
    if (!deviceProp.canMapHostMemory)
        HANDLE_ERROR(cudaMemcpyAsync(State_GPU->AuxStateGPU, State_GPU->AuxStateCPU,
                                     aux_bytes, cudaMemcpyHostToDevice, computeStream));
    HANDLE_ERROR(cudaMemsetAsync(State_GPU->InternalSpikeGPU, 0, sizeof(bool) * count, computeStream));
    Update${class_name}<<<gridsize, blocksize, 0, computeStream>>>(
        State_GPU->Vector_of_StateVariableGPU, State_GPU->AuxStateGPU,
        State_GPU->LastUpdateGPU, State_GPU->LastSpikingGPU, State_GPU->InternalSpikeGPU,
        count, time, timestepdouble, excited_, inhibitory_, current_enabled_, $parameter_arguments);
    HANDLE_ERROR(cudaGetLastError());
    if (!deviceProp.canMapHostMemory)
        HANDLE_ERROR(cudaMemcpyAsync(State_GPU->InternalSpikeCPU, State_GPU->InternalSpikeGPU,
                                     sizeof(bool) * count, cudaMemcpyDeviceToHost, computeStream));
    if (StateVector->IsMonitored) {
        HANDLE_ERROR(cudaMemcpyAsync(State_GPU->Vector_of_StateVariable,
            State_GPU->Vector_of_StateVariableGPU, sizeof(float) * count * $state_count,
            cudaMemcpyDeviceToHost, computeStream));
        HANDLE_ERROR(cudaMemcpyAsync(State_GPU->LastSpike, State_GPU->LastSpikingGPU,
            sizeof(int) * count, cudaMemcpyDeviceToHost, computeStream));
    }
    HANDLE_ERROR(cudaEventRecord(sync_event, computeStream));
    HANDLE_ERROR(cudaEventSynchronize(sync_event));
    std::memset(State_GPU->AuxStateCPU, 0, sizeof(float) * count * 2);
}

void ${class_name}::InitStateVector(int count, int gpu_index) {
    NumberOfNeuron = count;
    GPU_ID = gpu_index % NumberOfGPU;
    HANDLE_ERROR(cudaSetDevice(GPUIndex[GPU_ID]));
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, GPUIndex[GPU_ID]));
    HANDLE_ERROR(cudaEventCreate(&sync_event));
    HANDLE_ERROR(cudaStreamCreate(&copyStream));
    HANDLE_ERROR(cudaStreamCreate(&computeStream));
    State_GPU = static_cast<Neuron_State_Vector_Interface*>(StateVector);
    float initial[] = {$initial};
    State_GPU->InitNeuronStateGPU(count, initial, sigma_.data(), 3, deviceProp);
    blocksize = 128;
    gridsize = (count + blocksize - 1) / blocksize;
    if (gridsize < 1) gridsize = 1;
    current_synapse_ = new CurrentSynapse(count);
    InitializeInputCurrentSynapseStructure();
}

void ${class_name}::InitializeInputCurrentSynapseStructure() {
    if (current_synapse_) current_synapse_->InitializeInputCurrentPerSynapseStructure();
}

void ${class_name}::CheckType(Interconnections* inter) {
    if (inter->type == 0) excited_ = true;
    else if (inter->type == 1) inhibitory_ = true;
    else if (inter->type == 3) {
        current_enabled_ = true;
        inter->subindex_type = current_synapse_->N_connections[inter->TargetNeuron->index_in_NeuronModel];
        current_synapse_->IncrementNInputCurrentSynapsesPerNeuron(inter->TargetNeuron->index_in_NeuronModel);
    } else if (inter->type == 2) {
        throw std::invalid_argument("NMDA is not supported by generated conductance_lif_v1");
    } else {
        throw std::invalid_argument("unsupported generated legacy GPU synapse type");
    }
}
''')
