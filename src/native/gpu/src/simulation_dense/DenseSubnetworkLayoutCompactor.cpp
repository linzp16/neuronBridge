#include "simulation_dense/DenseSubnetworkLayoutCompactor.h"

#include <limits>
#include <map>

namespace npgr {
namespace sim_support {
namespace {

bool RemapNeuronId(int* value, const std::vector<int>& old_to_new, std::string* reason) {
    if (value == nullptr) {
        return true;
    }
    if (*value < 0 || *value >= static_cast<int>(old_to_new.size())) {
        if (reason != nullptr) {
            *reason = "dense local neuron id is out of range during compaction";
        }
        return false;
    }
    *value = old_to_new[static_cast<std::size_t>(*value)];
    return true;
}

std::vector<float> RemapParamVector(const std::vector<float>& old_values,
                                    const std::vector<int>& old_to_new,
                                    int neuron_count) {
    if (old_values.size() == 1) {
        return std::vector<float>(static_cast<std::size_t>(neuron_count), old_values.front());
    }
    std::vector<float> remapped(static_cast<std::size_t>(neuron_count), 0.0f);
    for (int old_id = 0; old_id < neuron_count; ++old_id) {
        const int new_id = old_to_new[static_cast<std::size_t>(old_id)];
        if (new_id < 0 || new_id >= neuron_count) {
            continue;
        }
        // add all the old parameter values to the new parameter values
        if (old_id < static_cast<int>(old_values.size())) {
            remapped[static_cast<std::size_t>(new_id)] = old_values[static_cast<std::size_t>(old_id)];
        }
    }
    return remapped;
}

}  // namespace

bool CompactDenseSubnetworkByModel(DenseSubnetworkBuildSpec* spec,
                                   std::vector<int>* old_to_new_local,
                                   std::string* reason) {
    if (spec == nullptr) {
        if (reason != nullptr) {
            *reason = "dense build spec must not be null";
        }
        return false;
    }
    // get the neuron count of the subnetwork
    const int neuron_count = spec->layout.stats.neuron_count;
    if (neuron_count <= 0 || spec->neuron_models.empty()) {
        if (old_to_new_local != nullptr) {
            old_to_new_local->clear();
        }
        return true;
    }
    // Build one compact range for each factory model id.
    std::map<int, std::vector<int> > group_to_model_indices;
    for (std::size_t model_index = 0; model_index < spec->neuron_models.size(); ++model_index) {
        group_to_model_indices[spec->neuron_models[model_index].factory_model_id]
            .push_back(static_cast<int>(model_index));
    }
    // build the remapping of the original neuron ids to the new ids
    std::vector<int> old_to_new(static_cast<std::size_t>(neuron_count), -1);
    // build the remapping of the original neuronmodel id per neuron to the new ids
    std::vector<int> old_model_id_to_new_model_id(spec->neuron_models.size(), -1);
    std::vector<DenseNeuronModelSpec> compacted_models;
    int next_neuron = 0;
    // scan all neuron model factory ids
    for (std::map<int, std::vector<int> >::const_iterator group = group_to_model_indices.begin();
         group != group_to_model_indices.end();
         ++group) {
        const int group_begin = next_neuron;
        DenseNeuronModelSpec merged;
        // get the first model id in a specific kind
        const DenseNeuronModelSpec& first =
            spec->neuron_models[static_cast<std::size_t>(group->second.front())];
        merged.factory_model_id = first.factory_model_id;
        merged.legacy_model_name = first.legacy_model_name;
        merged.model_id = static_cast<int>(compacted_models.size());
        // scan all the same kind model
        for (std::size_t index = 0; index < group->second.size(); ++index) {
            // get the old model index
            const int old_model_index = group->second[index];
            
            const DenseNeuronModelSpec& old_model =
                spec->neuron_models[static_cast<std::size_t>(old_model_index)];
            // map the old model id to the new merged model id
            old_model_id_to_new_model_id[static_cast<std::size_t>(old_model_index)] = merged.model_id;
            // build the old to new neuron id map
            const DenseNeuronRange& range = old_model.range;
            for (int offset = 0; offset < range.count; ++offset) {
                const int old_id = range.begin + offset;
                if (old_id < 0 || old_id >= neuron_count) {
                    if (reason != nullptr) {
                        *reason = "dense neuron range is out of range during compaction";
                    }
                    return false;
                }
                old_to_new[static_cast<std::size_t>(old_id)] = next_neuron++;
            }
        }
        // update the merged model range
        merged.range = DenseNeuronRange{group_begin, next_neuron - group_begin};
        compacted_models.push_back(merged);
    }

    for (int old_id = 0; old_id < neuron_count; ++old_id) {
        if (old_to_new[static_cast<std::size_t>(old_id)] < 0) {
            if (reason != nullptr) {
                *reason = "dense compaction did not assign every neuron";
            }
            return false;
        }
    }
    // scan all the old neuron models
    for (std::size_t old_model_index = 0; old_model_index < spec->neuron_models.size(); ++old_model_index) {
        const int new_model_id = old_model_id_to_new_model_id[old_model_index];
        if (new_model_id < 0 || new_model_id >= static_cast<int>(compacted_models.size())) {
            if (reason != nullptr) {
                *reason = "dense compaction produced an invalid model mapping";
            }
            return false;
        }
        // merge the user defined model parameters
        const DenseNeuronModelSpec& old_model = spec->neuron_models[old_model_index];
        DenseNeuronModelSpec& merged = compacted_models[static_cast<std::size_t>(new_model_id)];
        // scan all the user defined old model parameters
        for (std::map<std::string, std::vector<float> >::const_iterator param = old_model.per_neuron_params.begin();
             param != old_model.per_neuron_params.end();
             ++param) {
            // get the float value of a specific parameter
            std::vector<float>& merged_values = merged.per_neuron_params[param->first];
            if (merged_values.empty()) {
                // NaN marks neurons that did not receive this explicit layer
                // parameter; model factories fall back to their defaults.
                merged_values.assign(static_cast<std::size_t>(neuron_count),
                                     std::numeric_limits<float>::quiet_NaN());
            }
            const std::vector<float> remapped = RemapParamVector(param->second, old_to_new, neuron_count);
            const DenseNeuronRange& range = old_model.range;
            // add the old parameter value to the new merged model parameters
            for (int offset = 0; offset < range.count; ++offset) {
                const int old_id = range.begin + offset;
                const int new_id = old_to_new[static_cast<std::size_t>(old_id)];
                merged_values[static_cast<std::size_t>(new_id)] =
                    remapped[static_cast<std::size_t>(new_id)];
            }
        }
    }

    for (std::size_t synapse_id = 0; synapse_id < spec->layout.synapses.post_neuron.size(); ++synapse_id) {
        if (!RemapNeuronId(&spec->layout.synapses.post_neuron[synapse_id], old_to_new, reason)) {
            return false;
        }
    }
    for (std::size_t key_index = 0; key_index < spec->synapse_source_local_ids.size(); ++key_index) {
        if (!RemapNeuronId(&spec->synapse_source_local_ids[key_index], old_to_new, reason)) {
            return false;
        }
    }
    for (std::size_t input_index = 0; input_index < spec->input_target_local_ids.size(); ++input_index) {
        if (!RemapNeuronId(&spec->input_target_local_ids[input_index], old_to_new, reason)) {
            return false;
        }
    }
    for (std::size_t output_index = 0; output_index < spec->output_source_local_ids.size(); ++output_index) {
        if (!RemapNeuronId(&spec->output_source_local_ids[output_index], old_to_new, reason)) {
            return false;
        }
    }
    for (std::size_t output_index = 0;
         output_index < spec->output_dense_source_local_ids.size();
         ++output_index) {
        if (!RemapNeuronId(&spec->output_dense_source_local_ids[output_index], old_to_new, reason)) {
            return false;
        }
    }

    std::vector<int> compacted_model_by_neuron(static_cast<std::size_t>(neuron_count), -1);
    for (std::size_t old_id = 0; old_id < spec->neuron_model_id_by_neuron.size(); ++old_id) {
        const int old_model_id = spec->neuron_model_id_by_neuron[old_id];
        if (old_model_id < 0 || old_model_id >= static_cast<int>(old_model_id_to_new_model_id.size())) {
            continue;
        }
        const int new_id = old_to_new[old_id];
        // assign the old model id to the new model id
        compacted_model_by_neuron[static_cast<std::size_t>(new_id)] =
            old_model_id_to_new_model_id[static_cast<std::size_t>(old_model_id)];
    }
    spec->neuron_model_id_by_neuron.swap(compacted_model_by_neuron);
    // remap the post neuron id
    for (std::size_t synapse_id = 0; synapse_id < spec->layout.synapses.post_neuron.size(); ++synapse_id) {
        const int post_neuron = spec->layout.synapses.post_neuron[synapse_id];
        if (post_neuron >= 0 &&
            post_neuron < static_cast<int>(spec->neuron_model_id_by_neuron.size()) &&
            synapse_id < spec->layout.synapses.post_model.size()) {
            spec->layout.synapses.post_model[synapse_id] =
                spec->neuron_model_id_by_neuron[static_cast<std::size_t>(post_neuron)];
        }
    }
    // remap the output neuron mask
    if (!spec->output_neuron_mask.empty()) {
        std::vector<unsigned char> compacted_mask(static_cast<std::size_t>(neuron_count), 0);
        for (std::size_t old_id = 0; old_id < spec->output_neuron_mask.size(); ++old_id) {
            const int new_id = old_to_new[old_id];
            compacted_mask[static_cast<std::size_t>(new_id)] = spec->output_neuron_mask[old_id];
        }
        spec->output_neuron_mask.swap(compacted_mask);
    }
    if (!spec->monitor_local_to_original_ids.empty()) {
        std::vector<int> compacted_original_ids(static_cast<std::size_t>(neuron_count), -1);
        for (std::size_t old_id = 0; old_id < spec->monitor_local_to_original_ids.size(); ++old_id) {
            const int new_id = old_to_new[old_id];
            compacted_original_ids[static_cast<std::size_t>(new_id)] =
                spec->monitor_local_to_original_ids[old_id];
        }
        spec->monitor_local_to_original_ids.swap(compacted_original_ids);
    }
    if (!spec->monitor_candidate_mask.empty()) {
        std::vector<unsigned char> compacted_monitor_mask(static_cast<std::size_t>(neuron_count), 0);
        for (std::size_t old_id = 0; old_id < spec->monitor_candidate_mask.size(); ++old_id) {
            const int new_id = old_to_new[old_id];
            compacted_monitor_mask[static_cast<std::size_t>(new_id)] =
                spec->monitor_candidate_mask[old_id];
        }
        spec->monitor_candidate_mask.swap(compacted_monitor_mask);
    }

    spec->neuron_models.swap(compacted_models);
    spec->internal_model_ids.clear();
    for (std::size_t index = 0; index < spec->neuron_models.size(); ++index) {
        spec->internal_model_ids.push_back(static_cast<int>(index));
    }
    spec->layout.stats.model_count = static_cast<int>(spec->neuron_models.size());
    spec->layout.pre_delay_slices = PreDelaySlices{};

    if (old_to_new_local != nullptr) {
        *old_to_new_local = old_to_new;
    }
    return true;
}

}  // namespace sim_support
}  // namespace npgr
