/*
 * LegacyNetworkBridge.h
 *
 * Routes spikes from a dense subnetwork back into the legacy event queue.
 */
#ifndef NPGR_LEGACY_NETWORK_BRIDGE_H
#define NPGR_LEGACY_NETWORK_BRIDGE_H

#include "dense_subnetwork/DenseSubnetworkEvent.h"

#include <string>
#include <vector>

class Interconnections;
class EventQueue;

namespace npgr {

struct LegacyNetworkBinding {
    // Legacy-to-dense input connections and their dense external-input slot indices.
    std::vector<Interconnections*> input_connections;
    std::vector<int> input_slot_indices;
    // Dense-to-legacy output connections and destination event queues.
    std::vector<Interconnections*> output_connections;
    std::vector<int> output_queue_indices;
    // CSR starts for dense source neuron -> output binding indices.
    std::vector<int> output_route_start_by_source_neuron;
    // CSR payload storing output binding indices for every routed source neuron.
    std::vector<int> output_route_binding_indices;
};

class LegacyNetworkBridge {
public:
    LegacyNetworkBridge();
    ~LegacyNetworkBridge();

    // Schedules dense output spikes back into the legacy EventQueue.
    bool ScheduleDenseOutputSpikesToLegacy(const DenseSubnetworkEventResult& result,
                                           const LegacyNetworkBinding& binding,
                                           EventQueue* event_queue,
                                           std::string* reason = nullptr) const;

    // Applies dense output directly through legacy connections without queueing.
    bool ApplyDenseOutputSpikesDirect(const DenseSubnetworkEventResult& result,
                                      const LegacyNetworkBinding& binding,
                                      std::string* reason = nullptr) const;
};

}  // namespace npgr

#endif
