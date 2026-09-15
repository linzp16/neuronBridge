#include "bridge/LegacyNetworkBridge.h"

#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "source_file_realtime_v1_async/EventQueue/inc/EventQueue.h"
#include "source_file_realtime_v1_async/Event/inc/Spike/PropogatedSpikeGroup.h"

#include <cstddef>
#include <vector>

namespace npgr {

LegacyNetworkBridge::LegacyNetworkBridge() = default;
LegacyNetworkBridge::~LegacyNetworkBridge() = default;

bool LegacyNetworkBridge::ScheduleDenseOutputSpikesToLegacy(const DenseSubnetworkEventResult& result,
                                                            const LegacyNetworkBinding& binding,
                                                            EventQueue* event_queue,
                                                            std::string* reason) const {
    if (event_queue == nullptr) {
        if (reason != nullptr) {
            *reason = "event_queue must not be null";
        }
        return false;
    }

    if (result.output_firing_ids.empty() && !result.output_spikes.empty()) {
        for (std::size_t output_index = 0; output_index < result.output_spikes.size(); ++output_index) {
            const DenseOutputSpike& spike = result.output_spikes[output_index];
            if (spike.output_index < 0 ||
                spike.output_index >= static_cast<int>(binding.output_connections.size())) {
                if (reason != nullptr) {
                    *reason = "dense output spike contains invalid output_index";
                }
                return false;
            }
            Interconnections* output_connection =
                binding.output_connections[static_cast<std::size_t>(spike.output_index)];
            if (output_connection == nullptr) {
                if (reason != nullptr) {
                    *reason = "dense output binding contains null interconnection";
                }
                return false;
            }
            const int output_queue_index =
                binding.output_queue_indices[static_cast<std::size_t>(spike.output_index)];

            PropogatedSpikeGroup* group =
                new PropogatedSpikeGroup(spike.time_step + spike.delay, output_queue_index);
            group->IncludeNewSourceNeuron(1, output_connection);
            group->SourceNeuron = output_connection->SourceNeuron;
            event_queue->Insert_a_Event(group, output_queue_index);
        }
        return true;
    }

    for (std::size_t firing_index = 0; firing_index < result.output_firing_ids.size(); ++firing_index) {
        const int source_neuron = result.output_firing_ids[firing_index];
        if (source_neuron < 0 ||
            source_neuron + 1 >= static_cast<int>(binding.output_route_start_by_source_neuron.size())) {
            continue;
        }
        const int route_begin =
            binding.output_route_start_by_source_neuron[static_cast<std::size_t>(source_neuron)];
        const int route_end =
            binding.output_route_start_by_source_neuron[static_cast<std::size_t>(source_neuron + 1)];
        for (int route_index = route_begin; route_index < route_end; ++route_index) {
            if (route_index < 0 ||
                route_index >= static_cast<int>(binding.output_route_binding_indices.size())) {
                if (reason != nullptr) {
                    *reason = "dense output route contains invalid CSR offset";
                }
                return false;
            }
            const int output_index =
                binding.output_route_binding_indices[static_cast<std::size_t>(route_index)];
            if (output_index < 0 ||
                output_index >= static_cast<int>(binding.output_connections.size())) {
                if (reason != nullptr) {
                    *reason = "dense output route contains invalid binding index";
                }
                return false;
            }
            Interconnections* output_connection =
                binding.output_connections[static_cast<std::size_t>(output_index)];
            if (output_connection == nullptr) {
                if (reason != nullptr) {
                    *reason = "dense output binding contains null interconnection";
                }
                return false;
            }
            const int output_queue_index = binding.output_queue_indices[static_cast<std::size_t>(output_index)];

            PropogatedSpikeGroup* group =
                new PropogatedSpikeGroup(result.time_step + output_connection->delay, output_queue_index);
            group->IncludeNewSourceNeuron(1, output_connection);
            group->SourceNeuron = output_connection->SourceNeuron;
            event_queue->Insert_a_Event(group, output_queue_index);
        }
    }
    return true;
}

bool LegacyNetworkBridge::ApplyDenseOutputSpikesDirect(const DenseSubnetworkEventResult& result,
                                                       const LegacyNetworkBinding& binding,
                                                       std::string* reason) const {
    if (result.output_firing_ids.empty() && !result.output_spikes.empty()) {
        for (std::size_t output_index = 0; output_index < result.output_spikes.size(); ++output_index) {
            const DenseOutputSpike& spike = result.output_spikes[output_index];
            if (spike.output_index < 0 ||
                spike.output_index >= static_cast<int>(binding.output_connections.size())) {
                if (reason != nullptr) {
                    *reason = "dense output spike contains invalid output_index";
                }
                return false;
            }
            Interconnections* connection =
                binding.output_connections[static_cast<std::size_t>(spike.output_index)];
            if (connection == nullptr || connection->TargetNeuronModel == nullptr) {
                if (reason != nullptr) {
                    *reason = "dense output binding contains null target";
                }
                return false;
            }
            connection->TargetNeuronModel->ProcessSpike(connection, spike.time_step + spike.delay);
        }
        return true;
    }

    for (std::size_t firing_index = 0; firing_index < result.output_firing_ids.size(); ++firing_index) {
        const int source_neuron = result.output_firing_ids[firing_index];
        if (source_neuron < 0 ||
            source_neuron + 1 >= static_cast<int>(binding.output_route_start_by_source_neuron.size())) {
            continue;
        }
        const int route_begin =
            binding.output_route_start_by_source_neuron[static_cast<std::size_t>(source_neuron)];
        const int route_end =
            binding.output_route_start_by_source_neuron[static_cast<std::size_t>(source_neuron + 1)];
        for (int route_index = route_begin; route_index < route_end; ++route_index) {
            if (route_index < 0 ||
                route_index >= static_cast<int>(binding.output_route_binding_indices.size())) {
                if (reason != nullptr) {
                    *reason = "dense output route contains invalid CSR offset";
                }
                return false;
            }
            const int output_index =
                binding.output_route_binding_indices[static_cast<std::size_t>(route_index)];
            if (output_index < 0 ||
                output_index >= static_cast<int>(binding.output_connections.size())) {
                if (reason != nullptr) {
                    *reason = "dense output route contains invalid binding index";
                }
                return false;
            }
            Interconnections* connection =
                binding.output_connections[static_cast<std::size_t>(output_index)];
            if (connection == nullptr || connection->TargetNeuronModel == nullptr) {
                if (reason != nullptr) {
                    *reason = "dense output binding contains null target";
                }
                return false;
            }
            connection->TargetNeuronModel->ProcessSpike(connection, result.time_step + connection->delay);
        }
    }
    return true;
}

}  // namespace npgr
