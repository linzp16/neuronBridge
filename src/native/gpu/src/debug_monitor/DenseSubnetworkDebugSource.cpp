#include "debug_monitor/DenseSubnetworkDebugSource.h"

#include "simulation_dense/DenseSubnetworkModel.h"

#include <algorithm>

namespace npgr {
namespace {

bool IsDenseNeuronSelected(const DenseSubnetworkModel* subnetwork,
                           const DebugMonitorConfig& config,
                           const std::vector<int>* selected_local,
                           int local_id) {
    if (subnetwork == nullptr || local_id < 0) {
        return false;
    }
    if (selected_local != nullptr) {
        return std::find(selected_local->begin(), selected_local->end(), local_id) != selected_local->end();
    }
    const int global_id = subnetwork->OriginalGlobalNeuronIdForLocal(local_id);
    if (!config.neuron_ids.empty()) {
        return std::find(config.neuron_ids.begin(), config.neuron_ids.end(), global_id) !=
               config.neuron_ids.end();
    }
    return config.all_neurons || subnetwork->IsMonitorCandidateLocalNeuron(local_id);
}

bool IsActiveInSnapshot(const DenseNeuronDebugSnapshot& snapshot, int local_id) {
    std::map<std::string, std::vector<unsigned char> >::const_iterator active =
        snapshot.byte_state_vectors.find("active_mask");
    if (active == snapshot.byte_state_vectors.end()) {
        return true;
    }
    if (local_id < 0 || local_id >= static_cast<int>(active->second.size())) {
        return false;
    }
    return active->second[static_cast<std::size_t>(local_id)] != 0;
}

}  // namespace

DenseSubnetworkDebugSource::DenseSubnetworkDebugSource(DenseSubnetworkModel* subnetwork,
                                                       int subnetwork_index)
    : subnetwork_(subnetwork),
      subnetwork_index_(subnetwork_index),
      name_(subnetwork != nullptr ? subnetwork->name() : std::string()) {}

DebugComponentKind DenseSubnetworkDebugSource::kind() const {
    return DebugComponentKind::DenseSubnetwork;
}

int DenseSubnetworkDebugSource::component_index() const {
    return subnetwork_index_;
}

const std::string& DenseSubnetworkDebugSource::component_name() const {
    return name_;
}

bool DenseSubnetworkDebugSource::HasAnyTarget(const DebugMonitorConfig& config) const {
    if (subnetwork_ == nullptr) {
        return false;
    }
    return config.all_neurons ||
           config.record_weights ||
           config.record_spikes ||
           config.record_pending_channels ||
           config.dense_local_neuron_ids.find(name_) != config.dense_local_neuron_ids.end() ||
           !config.neuron_ids.empty() ||
           subnetwork_->HasMonitorTargets();
}

bool DenseSubnetworkDebugSource::Capture(int time_step,
                                         const DebugMonitorConfig& config,
                                         DebugMonitorFrame* frame,
                                         std::string* reason) {
    if (frame == nullptr) {
        if (reason != nullptr) {
            *reason = "dense debug source requires an output frame";
        }
        return false;
    }
    if (subnetwork_ == nullptr) {
        return true;
    }
    DenseSubnetworkMonitorState monitor_state;
    if (!subnetwork_->ExportMonitorState(config.record_state,
                                         config.record_spikes,
                                         config.record_pending_channels,
                                         config.record_weights,
                                         &monitor_state,
                                         reason)) {
        return false;
    }
    const std::vector<int>* selected_local = nullptr;
    std::map<std::string, std::vector<int> >::const_iterator found =
        config.dense_local_neuron_ids.find(name_);
    if (found != config.dense_local_neuron_ids.end()) {
        selected_local = &found->second;
    }

    if (config.record_state) {
        for (std::size_t snapshot_index = 0;
             snapshot_index < monitor_state.model_debug_states.size();
             ++snapshot_index) {
            const DenseNeuronDebugSnapshot& snapshot =
                monitor_state.model_debug_states[snapshot_index];
            for (std::map<std::string, std::vector<float> >::const_iterator field =
                     snapshot.float_state_vectors.begin();
                 field != snapshot.float_state_vectors.end();
                 ++field) {
                for (int local_id = 0; local_id < static_cast<int>(field->second.size()); ++local_id) {
                    if (!IsActiveInSnapshot(snapshot, local_id) ||
                        !IsDenseNeuronSelected(subnetwork_, config, selected_local, local_id)) {
                        continue;
                    }
                    DebugNeuronStateRecord record;
                    record.time_step = time_step;
                    record.component_kind = DebugComponentKind::DenseSubnetwork;
                    record.component_index = subnetwork_index_;
                    record.component_name = name_;
                    record.global_neuron_id = subnetwork_->OriginalGlobalNeuronIdForLocal(local_id);
                    record.local_neuron_id = local_id;
                    record.field_name = field->first;
                    record.value = field->second[static_cast<std::size_t>(local_id)];
                    frame->neuron_states.push_back(record);
                }
            }
            for (std::map<std::string, std::vector<unsigned char> >::const_iterator field =
                     snapshot.byte_state_vectors.begin();
                 field != snapshot.byte_state_vectors.end();
                 ++field) {
                for (int local_id = 0; local_id < static_cast<int>(field->second.size()); ++local_id) {
                    if (!IsActiveInSnapshot(snapshot, local_id) ||
                        !IsDenseNeuronSelected(subnetwork_, config, selected_local, local_id)) {
                        continue;
                    }
                    DebugNeuronStateRecord record;
                    record.time_step = time_step;
                    record.component_kind = DebugComponentKind::DenseSubnetwork;
                    record.component_index = subnetwork_index_;
                    record.component_name = name_;
                    record.global_neuron_id = subnetwork_->OriginalGlobalNeuronIdForLocal(local_id);
                    record.local_neuron_id = local_id;
                    record.field_name = field->first;
                    record.value = static_cast<float>(field->second[static_cast<std::size_t>(local_id)]);
                    frame->neuron_states.push_back(record);
                }
            }
        }
    }
    if (config.record_spikes) {
        for (std::size_t index = 0; index < monitor_state.current_firing_ids.size(); ++index) {
            const int local_id = monitor_state.current_firing_ids[index];
            if (!IsDenseNeuronSelected(subnetwork_, config, selected_local, local_id)) {
                continue;
            }
            DebugSpikeRecord spike;
            spike.time_step = time_step;
            spike.component_kind = DebugComponentKind::DenseSubnetwork;
            spike.component_index = subnetwork_index_;
            spike.component_name = name_;
            spike.global_neuron_id = subnetwork_->OriginalGlobalNeuronIdForLocal(local_id);
            spike.local_neuron_id = local_id;
            frame->spikes.push_back(spike);
        }
    }
    if (config.record_pending_channels && !monitor_state.pending_channels.empty()) {
        const int channel_count = monitor_state.pending_channel_count;
        const int channel_stride = monitor_state.pending_channel_stride;
        if (channel_count > 0 && channel_stride > 0) {
            for (int channel = 0; channel < channel_count; ++channel) {
                for (int local_id = 0; local_id < channel_stride; ++local_id) {
                    const std::size_t offset =
                        static_cast<std::size_t>(channel) * static_cast<std::size_t>(channel_stride) +
                        static_cast<std::size_t>(local_id);
                    if (offset >= monitor_state.pending_channels.size()) {
                        continue;
                    }
                    if (!IsDenseNeuronSelected(subnetwork_, config, selected_local, local_id)) {
                        continue;
                    }
                    DebugPendingChannelRecord record;
                    record.time_step = time_step;
                    record.subnetwork_index = subnetwork_index_;
                    record.subnetwork_name = name_;
                    record.channel = channel;
                    record.local_neuron_id = local_id;
                    record.global_neuron_id = subnetwork_->OriginalGlobalNeuronIdForLocal(local_id);
                    record.value = monitor_state.pending_channels[offset];
                    frame->pending_channels.push_back(record);
                }
            }
        }
    }
    if (config.record_weights) {
        for (std::size_t index = 0; index < monitor_state.synaptic_weights.size(); ++index) {
            DebugWeightRecord record;
            record.time_step = time_step;
            record.component_kind = DebugComponentKind::DenseSubnetwork;
            record.component_index = subnetwork_index_;
            record.component_name = name_;
            record.synapse_id = static_cast<int>(index);
            record.value = monitor_state.synaptic_weights[index];
            frame->weights.push_back(record);
        }
    }
    return true;
}

}  // namespace npgr
