#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/OuterDynamicInterfaceNeuronModel.h"

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "../source_file_realtime_v1_async/Neuron/inc/Neuron.h"

namespace {

std::vector<int> GetVectorIntParameter(const std::map<std::string, boost::any>& parameters,
                                       const char* name) {
    std::map<std::string, boost::any>::const_iterator it = parameters.find(name);
    if (it == parameters.end()) {
        return std::vector<int>();
    }
    return boost::any_cast<std::vector<int> >(it->second);
}

}  // namespace

OuterDynamicInterfaceNeuronModel::OuterDynamicInterfaceNeuronModel()
    : EventDrivenInputDevice(), simulation_(NULL), endpoint_global_start_(-1) {
    std::string name = "OuterDynamicInterfaceNeuronModel";
    this->setModelName(name);
    this->TimeDriven = false;
}

OuterDynamicInterfaceNeuronModel::OuterDynamicInterfaceNeuronModel(int timestep)
    : EventDrivenInputDevice(timestep), simulation_(NULL), endpoint_global_start_(-1) {
    std::string name = "OuterDynamicInterfaceNeuronModel";
    this->setModelName(name);
    this->TimeDriven = false;
}

OuterDynamicInterfaceNeuronModel::~OuterDynamicInterfaceNeuronModel() {}

void OuterDynamicInterfaceNeuronModel::BindSimulation(Simulation* simulation) {
    this->simulation_ = simulation;
}

void OuterDynamicInterfaceNeuronModel::ConfigureEndpointBindings(
    const std::vector<int>& outer_dynamic_ids,
    const std::vector<int>& joint_ids) {
    this->outer_dynamic_id_by_neuron_ = outer_dynamic_ids;
    this->joint_id_by_neuron_ = joint_ids;
}

void OuterDynamicInterfaceNeuronModel::ConfigureFromParameters(
    const std::map<std::string, boost::any>& parameters) {
    this->ConfigureEndpointBindings(
        GetVectorIntParameter(parameters, "outer_dynamic_ids_by_neuron"),
        GetVectorIntParameter(parameters, "outer_dynamic_joint_ids_by_neuron"));
    std::map<std::string, boost::any>::const_iterator start_it =
        parameters.find("outer_dynamic_endpoint_global_start");
    if (start_it != parameters.end()) {
        this->endpoint_global_start_ = boost::any_cast<int>(start_it->second);
    }
}

InternalSpike* OuterDynamicInterfaceNeuronModel::ProcessSpike(Interconnections* inter, int time) {
    if (this->simulation_ == NULL || inter == NULL || inter->TargetNeuron == NULL) {
        return NULL;
    }
    // Network creates one model instance per OpenMP queue, so
    // index_in_NeuronModel is queue-local and repeats from zero.  Resolve the
    // endpoint through its global neuron id to avoid routing every queue's
    // local prefix to the same OuterDynamic slots.
    const int local_index = this->endpoint_global_start_ >= 0
                                ? inter->TargetNeuron->Neuron_index - this->endpoint_global_start_
                                : inter->TargetNeuron->index_in_NeuronModel;
    if (local_index < 0 ||
        local_index >= static_cast<int>(this->outer_dynamic_id_by_neuron_.size()) ||
        local_index >= static_cast<int>(this->joint_id_by_neuron_.size())) {
        return NULL;
    }
    const int outer_dynamic_id = this->outer_dynamic_id_by_neuron_[local_index];
    const int joint_id = this->joint_id_by_neuron_[local_index];
    if (outer_dynamic_id < 0 ||
        outer_dynamic_id >= static_cast<int>(this->simulation_->OuterDynamicModelList.size()) ||
        this->simulation_->OuterDynamicModelList[static_cast<std::size_t>(outer_dynamic_id)] == NULL) {
        return NULL;
    }
    this->simulation_->OuterDynamicModelList[static_cast<std::size_t>(outer_dynamic_id)]
        ->AccumulateInputSpike(joint_id, inter->type, inter->weight, time);
    return NULL;
}

bool OuterDynamicInterfaceNeuronModel::compare(NeuronModel* neuronModel) {
    if (!EventDrivenInputDevice::compare(neuronModel)) {
        return false;
    }
    OuterDynamicInterfaceNeuronModel* other =
        dynamic_cast<OuterDynamicInterfaceNeuronModel*>(neuronModel);
    if (other == NULL) {
        return false;
    }
    return this->outer_dynamic_id_by_neuron_ == other->outer_dynamic_id_by_neuron_ &&
           this->joint_id_by_neuron_ == other->joint_id_by_neuron_;
}

std::map<std::string, boost::any> OuterDynamicInterfaceNeuronModel::getParameters() {
    std::map<std::string, boost::any> parameters;
    parameters["outer_dynamic_ids_by_neuron"] = this->outer_dynamic_id_by_neuron_;
    parameters["outer_dynamic_joint_ids_by_neuron"] = this->joint_id_by_neuron_;
    parameters["outer_dynamic_endpoint_global_start"] = this->endpoint_global_start_;
    return parameters;
}
