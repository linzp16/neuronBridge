#include "dense_subnetwork/DenseSubnetworkOutputExtractor.h"

#include <algorithm>

namespace npgr {

namespace {

bool IsOutputMasked(const sim_support::DenseSubnetworkBuildSpec& spec, int source_neuron) {
    if (source_neuron < 0 ||
        source_neuron >= static_cast<int>(spec.output_neuron_mask.size())) {
        return false;
    }
    return spec.output_neuron_mask[static_cast<std::size_t>(source_neuron)] != 0;
}

}  // namespace

std::vector<DenseOutputRouteEntry> DenseSubnetworkOutputExtractor::BuildOutputRoutesBySourceNeuron(
    const sim_support::DenseSubnetworkBuildSpec& spec,
    int neuron_count) {
    const int safe_neuron_count = neuron_count > 0 ? neuron_count : 0;
    std::vector<DenseOutputRouteEntry> output_routes_by_source_neuron;
    // scan all the output connection
    for (std::size_t output_index = 0; output_index < spec.output_source_local_ids.size(); ++output_index) {
        const int source_local_id = spec.output_source_local_ids[output_index];
        if (source_local_id < 0 ||
            source_local_id >= safe_neuron_count ||
            !IsOutputMasked(spec, source_local_id)) {
            continue;
        }
        const auto found = std::lower_bound(
            output_routes_by_source_neuron.begin(),
            output_routes_by_source_neuron.end(),
            source_local_id,
            [](const DenseOutputRouteEntry& entry, int source_neuron) {
                return entry.source_neuron < source_neuron;
            });
        // if this is a new source neuron
        if (found == output_routes_by_source_neuron.end() ||
            found->source_neuron != source_local_id) {
            // create a new entry
            DenseOutputRouteEntry entry;
            entry.source_neuron = source_local_id;
            DenseOutputSpike spike;
            spike.output_index = static_cast<int>(output_index);
            spike.source_neuron = source_local_id;
            spike.external_target_id = spec.output_target_main_ids[output_index];
            spike.delay = spec.output_delays[output_index];
            spike.weight = spec.output_weights[output_index];
            spike.synapse_type = spec.output_types[output_index];
            entry.output_spikes.push_back(spike);
            output_routes_by_source_neuron.insert(found, entry);
        } else {
            // add the output spike to the existing entry
            DenseOutputSpike spike;
            spike.output_index = static_cast<int>(output_index);
            spike.source_neuron = source_local_id;
            spike.external_target_id = spec.output_target_main_ids[output_index];
            spike.delay = spec.output_delays[output_index];
            spike.weight = spec.output_weights[output_index];
            spike.synapse_type = spec.output_types[output_index];
            found->output_spikes.push_back(spike);
        }
    }
    return output_routes_by_source_neuron;
}

std::vector<DenseOutputSpike> DenseSubnetworkOutputExtractor::Extract(
    const std::vector<DenseOutputRouteEntry>& output_routes_by_source_neuron,
    const std::vector<int>& firing_ids,
    int time_step) {
    std::vector<DenseOutputSpike> emitted;
    for (std::size_t firing_index = 0; firing_index < firing_ids.size(); ++firing_index) {
        const int source_neuron = firing_ids[firing_index];
        const auto found = std::lower_bound(
            output_routes_by_source_neuron.begin(),
            output_routes_by_source_neuron.end(),
            source_neuron,
            [](const DenseOutputRouteEntry& entry, int source) {
                return entry.source_neuron < source;
            });
        if (found == output_routes_by_source_neuron.end() ||
            found->source_neuron != source_neuron) {
            continue;
        }
        for (std::size_t local_index = 0; local_index < found->output_spikes.size(); ++local_index) {
            DenseOutputSpike spike = found->output_spikes[local_index];
            spike.time_step = time_step;
            emitted.push_back(spike);
        }
    }
    return emitted;
}

}  // namespace npgr
