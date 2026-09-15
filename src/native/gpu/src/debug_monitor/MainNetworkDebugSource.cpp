#include "debug_monitor/MainNetworkDebugSource.h"

#include "source_file_realtime_v1_async/Network/inc/Network.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "source_file_realtime_v1_async/Neuron/inc/Neuron.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"

#include <algorithm>

namespace npgr {

MainNetworkDebugSource::MainNetworkDebugSource(const Network* network,
                                               const std::vector<int>& original_to_main_neuron_id)
    : network_(network),
      name_("main_network") {
    if (network_ == nullptr || network_->neurons == nullptr || network_->neuronsNum <= 0) {
        return;
    }
    main_to_original_ids_.assign(static_cast<std::size_t>(network_->neuronsNum), -1);
    if (!original_to_main_neuron_id.empty()) {
        for (std::size_t original = 0; original < original_to_main_neuron_id.size(); ++original) {
            const int main_id = original_to_main_neuron_id[original];
            if (main_id >= 0 && main_id < network_->neuronsNum) {
                main_to_original_ids_[static_cast<std::size_t>(main_id)] = static_cast<int>(original);
            }
        }
    }
    for (int main_id = 0; main_id < network_->neuronsNum; ++main_id) {
        if (main_to_original_ids_[static_cast<std::size_t>(main_id)] < 0) {
            main_to_original_ids_[static_cast<std::size_t>(main_id)] = main_id;
        }
        if (network_->neurons[main_id].IsMonitor) {
            monitored_main_ids_.push_back(main_id);
        }
    }
}

DebugComponentKind MainNetworkDebugSource::kind() const {
    return DebugComponentKind::MainNetwork;
}

int MainNetworkDebugSource::component_index() const {
    return 0;
}

const std::string& MainNetworkDebugSource::component_name() const {
    return name_;
}

bool MainNetworkDebugSource::HasAnyTarget(const DebugMonitorConfig& config) const {
    if (network_ == nullptr || network_->neurons == nullptr) {
        return false;
    }
    return config.all_neurons ||
           config.record_spikes ||
           config.record_weights ||
           !config.neuron_ids.empty() ||
           !monitored_main_ids_.empty();
}

bool MainNetworkDebugSource::Capture(int time_step,
                                     const DebugMonitorConfig& config,
                                     DebugMonitorFrame* frame,
                                     std::string* reason) {
    if (frame == nullptr) {
        if (reason != nullptr) {
            *reason = "main debug source requires an output frame";
        }
        return false;
    }
    if ((!config.record_state && !config.record_spikes && !config.record_weights) ||
        network_ == nullptr ||
        network_->neurons == nullptr) {
        return true;
    }
    if (config.record_weights && network_->wordination != nullptr) {
        // Main-network weights are exported in original ConnectionDescription
        // order, matching Simulation::GetConnectionWeight(index).
        for (int synapse_id = 0; synapse_id < network_->intersNum; ++synapse_id) {
            const Interconnections* connection = network_->wordination[synapse_id];
            if (connection == nullptr) {
                continue;
            }
            DebugWeightRecord record;
            record.time_step = time_step;
            record.component_kind = DebugComponentKind::MainNetwork;
            record.component_index = 0;
            record.component_name = name_;
            record.synapse_id = synapse_id;
            record.value = connection->weight;
            frame->weights.push_back(record);
        }
    }
    if (!config.record_state && !config.record_spikes) {
        return true;
    }
    const int neuron_count = network_->neuronsNum;
    for (int main_id = 0; main_id < neuron_count; ++main_id) {
        const int global_id = main_id < static_cast<int>(main_to_original_ids_.size())
                                  ? main_to_original_ids_[static_cast<std::size_t>(main_id)]
                                  : main_id;
        if (!config.neuron_ids.empty() && !this->IsSelected(global_id, config)) {
            continue;
        }
        if (!config.all_neurons &&
            config.neuron_ids.empty() &&
            !network_->neurons[main_id].IsMonitor) {
            continue;
        }
        const Neuron& neuron = network_->neurons[main_id];
        Neuron_State_Vector* state = neuron.neuron_state_vector;
        if (state == nullptr || state->Vector_of_StateVariable == nullptr ||
            neuron.index_in_NeuronModel < 0 ||
            neuron.index_in_NeuronModel >= state->NumberofNeuron) {
            continue;
        }
        if (config.record_state) {
            const int printable_count = state->GetNumberOfPrintableValues();
            for (int field = 0; field < printable_count; ++field) {
                DebugNeuronStateRecord record;
                record.time_step = time_step;
                record.component_kind = DebugComponentKind::MainNetwork;
                record.component_index = 0;
                record.component_name = name_;
                record.global_neuron_id = global_id;
                record.local_neuron_id = main_id;
                record.field_name = field == 0 ? "v" : ("state_" + std::to_string(field));
                record.value = state->GetPrintableValuesAt(neuron.index_in_NeuronModel, field);
                frame->neuron_states.push_back(record);
            }
        }
        bool* internal_spike = state->getInternalSpike();
        if (config.record_spikes &&
            internal_spike != nullptr &&
            internal_spike[neuron.index_in_NeuronModel]) {
            DebugSpikeRecord spike;
            spike.time_step = time_step;
            spike.component_kind = DebugComponentKind::MainNetwork;
            spike.component_index = 0;
            spike.component_name = name_;
            spike.global_neuron_id = global_id;
            spike.local_neuron_id = main_id;
            frame->spikes.push_back(spike);
        }
    }
    return true;
}

bool MainNetworkDebugSource::IsSelected(int global_neuron_id, const DebugMonitorConfig& config) const {
    if (config.all_neurons) {
        return true;
    }
    if (config.neuron_ids.empty()) {
        return false;
    }
    return std::find(config.neuron_ids.begin(), config.neuron_ids.end(), global_neuron_id) != config.neuron_ids.end();
}

}  // namespace npgr
