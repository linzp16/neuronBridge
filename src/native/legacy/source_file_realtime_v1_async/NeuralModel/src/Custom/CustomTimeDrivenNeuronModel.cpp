#include "../source_file_realtime_v1_async/NeuralModel/inc/Custom/CustomTimeDrivenNeuronModel.h"

#include "../source_file_realtime_v1_async/Current/inc/CurrentSynapse.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <stdexcept>
#include <vector>

CustomTimeDrivenNeuronModel::CustomTimeDrivenNeuronModel(
    const npgr::CustomEquationDescriptor& descriptor,
    int timestep_size)
    : TimeDrivenModel(timestep_size),
      descriptor_(descriptor),
      current_synapse_model_(0) {
    if (descriptor_.implementation_name == 0 || descriptor_.state_count <= 0 ||
        descriptor_.differential_state_count < 0 ||
        descriptor_.differential_state_count > descriptor_.state_count ||
        descriptor_.voltage_state_slot < 0 || descriptor_.input_binding_count < 0 ||
        (descriptor_.input_binding_count > 0 && descriptor_.input_bindings == 0) ||
        descriptor_.voltage_state_slot >= descriptor_.state_count) {
        throw std::invalid_argument("invalid generated custom neuron descriptor");
    }
    this->StateVector = new Neuron_State_Vector(descriptor_.state_count, true);
    this->setModelName(descriptor_.implementation_name);
}

CustomTimeDrivenNeuronModel::~CustomTimeDrivenNeuronModel() {
    if (current_synapse_model_ != 0) {
        delete current_synapse_model_;
        current_synapse_model_ = 0;
    }
}

void CustomTimeDrivenNeuronModel::InitStateVector(int neuron_count, int) {
    std::vector<float> zero_state(descriptor_.state_count, 0.0f);
    InitializeStateVector(neuron_count, zero_state.data(), zero_state.data());
}

Neuron_State_Vector* CustomTimeDrivenNeuronModel::InitState() {
    return this->StateVector;
}

InternalSpike* CustomTimeDrivenNeuronModel::ProcessSpike(Interconnections* inter, int) {
    if (inter == 0) {
        throw std::invalid_argument("custom neuron received a null interconnection");
    }
    const npgr::CustomInputBinding* binding =
        FindInputBinding(inter->type, npgr::CustomInputDelivery::AddToState);
    if (binding == 0) {
        throw std::invalid_argument("custom neuron does not declare the arriving spike type");
    }
    this->StateVector->SetNeuronStateIncrement(
        inter->TargetNeuronModelIndex,
        binding->target_state_slot,
        inter->weight * binding->scale);
    return 0;
}

void CustomTimeDrivenNeuronModel::ProcessCurrent(
    Interconnections* inter,
    Neuron* target,
    float current) {
    if (inter == 0 || target == 0 || current_synapse_model_ == 0) {
        throw std::logic_error("custom neuron current input is not initialized");
    }
    const npgr::CustomInputBinding* binding = FindInputBinding(
        inter->type, npgr::CustomInputDelivery::AddToCurrentAccumulator);
    if (binding == 0) {
        throw std::invalid_argument("custom neuron does not declare the arriving current type");
    }
    const int neuron_index = target->index_in_NeuronModel;
    current_synapse_model_->SetInputCurrentPerSynapse(
        neuron_index, inter->subindex_type, current);
    this->StateVector->SetNeuronState(
        neuron_index,
        binding->target_state_slot,
        current_synapse_model_->GetTotalInputCurrentPerNeuron(neuron_index));
}

void CustomTimeDrivenNeuronModel::InitializeInputCurrentSynapseStructure() {
    if (current_synapse_model_ != 0) {
        current_synapse_model_->InitializeInputCurrentPerSynapseStructure();
    }
}

void CustomTimeDrivenNeuronModel::CheckType(Interconnections* inter) {
    if (inter == 0) {
        throw std::invalid_argument("custom neuron received a null interconnection");
    }
    if (inter->type == 3) {
        const npgr::CustomInputBinding* binding = FindInputBinding(
            inter->type, npgr::CustomInputDelivery::AddToCurrentAccumulator);
        if (binding == 0 || current_synapse_model_ == 0) {
            throw std::invalid_argument("custom neuron does not declare current input type 3");
        }
        if (inter->TargetNeuron == 0) {
            throw std::invalid_argument("custom neuron current input has no target neuron");
        }
        const int neuron_index = inter->TargetNeuron->index_in_NeuronModel;
        inter->subindex_type = current_synapse_model_->N_connections[neuron_index];
        current_synapse_model_->IncrementNInputCurrentSynapsesPerNeuron(neuron_index);
        return;
    }
    if (FindInputBinding(inter->type, npgr::CustomInputDelivery::AddToState) == 0) {
        throw std::invalid_argument("custom neuron does not declare this synapse type");
    }
}

int CustomTimeDrivenNeuronModel::getV_index() {
    return descriptor_.voltage_state_slot;
}

int CustomTimeDrivenNeuronModel::get_NumberOfState() {
    return descriptor_.state_count;
}

NeuronModelType CustomTimeDrivenNeuronModel::getNeuronModelType() {
    return NEURAL_LAYER;
}

const npgr::CustomEquationDescriptor& CustomTimeDrivenNeuronModel::descriptor() const {
    return descriptor_;
}

void CustomTimeDrivenNeuronModel::InitializeStateVector(
    int neuron_count,
    float* initial_state,
    float* initial_sigma) {
    if (neuron_count < 0 || initial_state == 0 || initial_sigma == 0) {
        throw std::invalid_argument("invalid custom neuron state initialization");
    }
    this->StateVector->InitNeuronState(neuron_count, initial_state, initial_sigma);
    if (current_synapse_model_ != 0) {
        delete current_synapse_model_;
    }
    current_synapse_model_ = new CurrentSynapse(neuron_count);
}

void CustomTimeDrivenNeuronModel::MarkSpike(int neuron_index) {
    this->StateVector->SpikeIndex[this->StateVector->NumberofSpike] = neuron_index;
    this->StateVector->NumberofSpike += 1;
    this->StateVector->LastSpike[neuron_index] = 0;
}

const npgr::CustomInputBinding* CustomTimeDrivenNeuronModel::FindInputBinding(
    int connection_type,
    npgr::CustomInputDelivery delivery) const {
    for (int index = 0; index < descriptor_.input_binding_count; ++index) {
        const npgr::CustomInputBinding& binding = descriptor_.input_bindings[index];
        if (binding.connection_type == connection_type && binding.delivery == delivery) {
            return &binding;
        }
    }
    return 0;
}
