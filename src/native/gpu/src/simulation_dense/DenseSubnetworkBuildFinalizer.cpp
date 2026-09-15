#include "simulation_dense/DenseSubnetworkBuildFinalizer.h"

#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <map>
#include <vector>

namespace npgr {
namespace sim_support {
namespace {

bool Fail(std::string* reason, const std::string& message) {
    if (reason != nullptr) {
        *reason = message;
    }
    return false;
}

}  // namespace

bool FinalizeDenseSubnetworkBuildSpec(DenseSubnetworkBuildSpec* spec,
                                      std::string* reason) {
    if (spec == nullptr) {
        return Fail(reason, "dense subnetwork spec must not be null");
    }
    const int neuron_count = spec->layout.stats.neuron_count;
    if (neuron_count <= 0) {
        return Fail(reason, "dense subnetwork spec does not contain neurons");
    }
    // get the total synapse count of the subnetwork
    const int synapse_count = static_cast<int>(spec->layout.synapses.post_neuron.size());
    if (spec->synapse_source_local_ids.size() != static_cast<std::size_t>(synapse_count) ||
        spec->synapse_delay_slots.size() != static_cast<std::size_t>(synapse_count)) {
        return Fail(reason, "dense synapse source/delay arrays must be parallel to layout.synapses");
    }

    int max_delay = 0;
    // scan the total subnetwork and get the max delay
    for (std::size_t index = 0; index < spec->synapse_source_local_ids.size(); ++index) {
        const int source_local_id = spec->synapse_source_local_ids[index];
        const int delay = spec->synapse_delay_slots[index];
        if (source_local_id < 0 || source_local_id >= neuron_count) {
            return Fail(reason, "dense synapse slice key contains invalid source neuron id");
        }
        if (delay < 0) {
            return Fail(reason, "dense synapse slice key contains negative delay");
        }
        const int post_neuron = spec->layout.synapses.post_neuron[index];
        if (post_neuron < 0 || post_neuron >= neuron_count) {
            return Fail(reason, "dense synapse layout contains invalid post neuron id");
        }
        max_delay = std::max(max_delay, delay);
    }

    GpuPropagationLayout& layout = spec->layout;
    layout.stats.synapse_count = synapse_count;
    // init the delay slot count
    layout.stats.delay_slot_count = max_delay + 1;
    // count the number of neuron models
    layout.stats.model_count = static_cast<int>(spec->neuron_models.size());
    // get the maximum pending_channel_count
    layout.stats.pending_channel_count =
        std::max(3, DenseNeuronModelFactory::Instance().RequiredPendingChannelCount());
    if (synapse_count > 0 &&
        layout.synapses.effect_channel.size() != static_cast<std::size_t>(synapse_count)) {
        return Fail(reason, "dense synapse effect_channel must be compiled for every synapse");
    }
    if (synapse_count > 0 &&
        layout.synapses.effect_scale.size() != static_cast<std::size_t>(synapse_count)) {
        return Fail(reason, "dense synapse effect_scale must be compiled for every synapse");
    }
    if (layout.synapses.max_weight.empty()) {
        layout.synapses.max_weight = layout.synapses.weight;
    }
    if (layout.synapses.plastic_rule_id.empty()) {
        layout.synapses.plastic_rule_id.assign(static_cast<std::size_t>(synapse_count), -1);
    }
    if (layout.synapses.plastic_state_index.empty()) {
        layout.synapses.plastic_state_index.assign(static_cast<std::size_t>(synapse_count), -1);
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            if (layout.synapses.plastic_rule_id[static_cast<std::size_t>(synapse_id)] >= 0) {
                layout.synapses.plastic_state_index[static_cast<std::size_t>(synapse_id)] = synapse_id;
            }
        }
    }
    if (layout.synapses.trigger_rule_id.empty()) {
        layout.synapses.trigger_rule_id.assign(static_cast<std::size_t>(synapse_count), -1);
    }
    const bool has_learning_rule_list = !spec->learning_rule_table.empty();
    if (!has_learning_rule_list) {
        // An empty dense learning rule table means the user did not define
        // dense plasticity models. Rule references in synapses are therefore
        // configuration errors instead of implicit requests for default rules.
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            if (layout.synapses.plastic_rule_id[static_cast<std::size_t>(synapse_id)] >= 0 ||
                layout.synapses.trigger_rule_id[static_cast<std::size_t>(synapse_id)] >= 0) {
                return Fail(reason, "dense synapse references learning rules but learning_rule_list is empty");
            }
        }
        layout.synapses.plastic_model_id.assign(static_cast<std::size_t>(synapse_count), -1);
        layout.synapses.plastic_flags.assign(static_cast<std::size_t>(synapse_count), 0);
        layout.synapses.trigger_model_id.assign(static_cast<std::size_t>(synapse_count), -1);
        layout.synapses.trigger_flags.assign(static_cast<std::size_t>(synapse_count), 0);
        layout.learning_fields = DenseLearningHostFieldTable{};
        layout.learning_model_indices = DenseLearningModelFieldIndexTable{};
        layout.learning_trigger_routes = DenseLearningTriggerRouteTable{};
        layout.learning_spike_buffers = DenseLearningSpikeBufferTable{};
    } else {
        // Runtime model ids and hook flags are compiled from rule ids by the
        // learning factory. Build callers only need to provide rule bindings; the
        // concrete rule parameters stay in spec->learning_rule_fields.
        if (!DenseLearningRuleFactory::Instance().CompileSynapseLearningBindings(
                synapse_count,
                layout.synapses.plastic_rule_id,
                layout.synapses.trigger_rule_id,
                spec->learning_rule_table,
                &layout.synapses.plastic_model_id,
                &layout.synapses.plastic_flags,
                &layout.synapses.trigger_model_id,
                &layout.synapses.trigger_flags,
                reason)) {
            return false;
        }
    }
    if (layout.synapses.max_weight.size() != static_cast<std::size_t>(synapse_count) ||
        layout.synapses.plastic_rule_id.size() != static_cast<std::size_t>(synapse_count) ||
        layout.synapses.plastic_model_id.size() != static_cast<std::size_t>(synapse_count) ||
        layout.synapses.plastic_flags.size() != static_cast<std::size_t>(synapse_count) ||
        layout.synapses.plastic_state_index.size() != static_cast<std::size_t>(synapse_count) ||
        layout.synapses.trigger_rule_id.size() != static_cast<std::size_t>(synapse_count) ||
        layout.synapses.trigger_model_id.size() != static_cast<std::size_t>(synapse_count) ||
        layout.synapses.trigger_flags.size() != static_cast<std::size_t>(synapse_count)) {
        return Fail(reason, "dense synapse learning arrays must be parallel to layout.synapses");
    }
    if (has_learning_rule_list) {
        // build the registeredfieldtable of the learning rule model
        if (!DenseLearningRuleFactory::Instance().BuildRegisteredFieldTable(
                synapse_count,
                &layout.learning_fields,
                &layout.learning_model_indices,
                reason)) {
            return false;
        }
        // Learning-rule-specific field materialization is owned by the learning
        // factory. It expands rule-indexed parameters into synapse-indexed runtime
        // fields that can be uploaded directly to the GPU.
        // construct the learning field table's flat pool
        if (!DenseLearningRuleFactory::Instance().PopulateFieldTableFromRuleBindings(
                synapse_count,
                layout.synapses.plastic_rule_id,
                layout.synapses.plastic_model_id,
                layout.synapses.trigger_rule_id,
                layout.synapses.trigger_model_id,
                layout.synapses.max_weight,
                spec->learning_rule_table,
                spec->learning_rule_fields,
                &layout.learning_fields,
                reason)) {
            return false;
        }
        layout.learning_trigger_routes.start.assign(static_cast<std::size_t>(synapse_count), 0);
        layout.learning_trigger_routes.count.assign(static_cast<std::size_t>(synapse_count), 0);
        layout.learning_trigger_routes.synapse_ids.clear();
        layout.learning_spike_buffers = DenseLearningSpikeBufferTable{};
        layout.learning_spike_buffers.synapse_bucket_id.assign(static_cast<std::size_t>(synapse_count), -1);
        std::map<std::pair<int, int>, std::vector<int> > plastic_by_target_and_rule;
        // scan all synapses to build plastic_rule pair
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            // get the rule id
            const int plastic_rule = layout.synapses.plastic_rule_id[static_cast<std::size_t>(synapse_id)];
            // skip if it isn't a plastic id but a trigger id
            if (plastic_rule < 0 ||
                layout.synapses.trigger_rule_id[static_cast<std::size_t>(synapse_id)] >= 0) {
                continue;
            }
            // get the target of the post neuron
            const int target = layout.synapses.post_neuron[static_cast<std::size_t>(synapse_id)];
            // add the synapse to the list of plastic synapses for the target and rule
            plastic_by_target_and_rule[std::make_pair(target, plastic_rule)].push_back(synapse_id);
        }
        // again scan all the connections to build the trigger routes
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            // get the trigger rule index
            const int trigger_rule = layout.synapses.trigger_rule_id[static_cast<std::size_t>(synapse_id)];
            // skip if it doesn't have the trigger rule
            if (trigger_rule < 0) {
                continue;
            }
            // get the target of the post neuron
            const int target = layout.synapses.post_neuron[static_cast<std::size_t>(synapse_id)];
            // make the key pair
            const std::pair<int, int> key = std::make_pair(target, trigger_rule);
            // get the connections which this trigger connection can affect
            const std::vector<int>& routed = plastic_by_target_and_rule[key];
            // get the start index of the trigger synapse
            layout.learning_trigger_routes.start[static_cast<std::size_t>(synapse_id)] =
                static_cast<int>(layout.learning_trigger_routes.synapse_ids.size());
            // get the routed size
            layout.learning_trigger_routes.count[static_cast<std::size_t>(synapse_id)] =
                static_cast<int>(routed.size());
            // add the learning trigger ids to the  total trigger routes
            layout.learning_trigger_routes.synapse_ids.insert(
                layout.learning_trigger_routes.synapse_ids.end(), routed.begin(), routed.end());
        }
        // Build learning spike-buffer buckets after trigger routes are known.
        // Each bucket represents one legacy BufferedActivityTime owner:
        // (target_dense_neuron, plastic_rule_id).
        std::map<std::pair<int, int>, int> spike_bucket_by_target_and_rule;
        // scan all the target and rule table
        for (std::map<std::pair<int, int>, std::vector<int> >::const_iterator it =
                 plastic_by_target_and_rule.begin();
             it != plastic_by_target_and_rule.end();
             ++it) {
            // get the synapses that use the same target and rule
            const std::vector<int>& plastic_synapses = it->second;
            bool has_buffered_plastic = false;
            float max_time_measured = 0.0f;
            // scan all the synapses that use the same target and rule
            for (int synapse_id : plastic_synapses) {
                // get the buffertime according to learning rule model
                const float buffered_time =
                    DenseLearningRuleFactory::Instance().MaxBufferedTimeForBinding(
                        layout.synapses.plastic_rule_id[static_cast<std::size_t>(synapse_id)],
                        layout.synapses.plastic_model_id[static_cast<std::size_t>(synapse_id)],
                        spec->learning_rule_table,
                        spec->learning_rule_fields);
                if (buffered_time > 0.0f) {
                    has_buffered_plastic = true;
                    max_time_measured = std::max(max_time_measured, buffered_time);
                }
            }
            if (!has_buffered_plastic) {
                continue;
            }
            // get the new bucket id
            const int bucket_id = static_cast<int>(layout.learning_spike_buffers.bucket_start.size());
            // bind the bucket_id to the target and rule
            spike_bucket_by_target_and_rule[it->first] = bucket_id;
            // Capacity is sized from the measurement window and the number of
            // ordinary plastic synapses that can write into this target/rule bucket.
            // get the max bucket window steps
            const int window_steps =
                std::max(1, static_cast<int>(std::ceil(max_time_measured / std::max(1.0e-6f, spec->config.dt_ms))) + 1);
            //caculate the max buffer capcity
            const int bucket_capacity =
                std::max(1, static_cast<int>(plastic_synapses.size()) * window_steps);
            // get the current bucket start spike index in the spike time buffer
            const int bucket_start =
                static_cast<int>(layout.learning_spike_buffers.spike_time.size());
            // add the barket start marker
            layout.learning_spike_buffers.bucket_start.push_back(bucket_start);
            // add the bucket capcity marker
            layout.learning_spike_buffers.bucket_capacity.push_back(bucket_capacity);
            // add the backet head marker
            layout.learning_spike_buffers.bucket_head.push_back(0);
            // add the bucket cout storage
            layout.learning_spike_buffers.bucket_count.push_back(0);
            // insert the time storage to the end of the buffer to set up the 
            layout.learning_spike_buffers.spike_time.insert(
                layout.learning_spike_buffers.spike_time.end(),
                static_cast<std::size_t>(bucket_capacity),
                0.0f);
            // insert the spiking synapse id to the end of the buffer
            layout.learning_spike_buffers.spike_synapse_id.insert(
                layout.learning_spike_buffers.spike_synapse_id.end(),
                static_cast<std::size_t>(bucket_capacity),
                -1);
            // insert the valid mark to the end of the buffer
            layout.learning_spike_buffers.valid.insert(
                layout.learning_spike_buffers.valid.end(),
                static_cast<std::size_t>(bucket_capacity),
                0u);
            // mark the synapse to the bucket 
            for (int synapse_id : plastic_synapses) {
                layout.learning_spike_buffers.synapse_bucket_id[static_cast<std::size_t>(synapse_id)] =
                    bucket_id;
            }
        }
        // Trigger synapses read the same bucket as the ordinary plastic synapses
        // they route to, but they never write pre-spike records themselves.
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            // buffer_time > 0 means the model need the spike buffer
            const float buffered_time =
                DenseLearningRuleFactory::Instance().MaxBufferedTimeForBinding(
                    layout.synapses.trigger_rule_id[static_cast<std::size_t>(synapse_id)],
                    layout.synapses.trigger_model_id[static_cast<std::size_t>(synapse_id)],
                    spec->learning_rule_table,
                    spec->learning_rule_fields);
            // skip if the model doesn't need a buffer
            if (buffered_time <= 0.0f) {
                continue;
            }
            const int trigger_rule = layout.synapses.trigger_rule_id[static_cast<std::size_t>(synapse_id)];
            const int target = layout.synapses.post_neuron[static_cast<std::size_t>(synapse_id)];
            // find the target and rule bucket
            const std::map<std::pair<int, int>, int>::const_iterator found =
                spike_bucket_by_target_and_rule.find(std::make_pair(target, trigger_rule));
            if (found != spike_bucket_by_target_and_rule.end()) {
                // map the trigger synapse to the bucket
                layout.learning_spike_buffers.synapse_bucket_id[static_cast<std::size_t>(synapse_id)] =
                    found->second;
            }
        }
    }
    layout.pre_delay_slices.neuron_count = neuron_count;
    layout.pre_delay_slices.delay_slot_count = layout.stats.delay_slot_count;

    const int flat_count = neuron_count * layout.stats.delay_slot_count;
    layout.pre_delay_slices.start.assign(static_cast<std::size_t>(flat_count), 0);
    layout.pre_delay_slices.count.assign(static_cast<std::size_t>(flat_count), 0);
    layout.pre_delay_slices.synapse_ids.clear();

    std::vector<std::vector<int> > slices(static_cast<std::size_t>(flat_count));
    for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
        const int source_local_id = spec->synapse_source_local_ids[static_cast<std::size_t>(synapse_id)];
        const int delay = spec->synapse_delay_slots[static_cast<std::size_t>(synapse_id)];
        const int flat = source_local_id * layout.stats.delay_slot_count + delay;
        slices[static_cast<std::size_t>(flat)].push_back(synapse_id);
    }

    int running_start = 0;
    for (int pre_neuron = 0; pre_neuron < neuron_count; ++pre_neuron) {
        for (int delay_slot = 0; delay_slot < layout.stats.delay_slot_count; ++delay_slot) {
            const int flat = pre_neuron * layout.stats.delay_slot_count + delay_slot;
            const std::vector<int>& ids = slices[static_cast<std::size_t>(flat)];
            layout.pre_delay_slices.start[static_cast<std::size_t>(flat)] = running_start;
            layout.pre_delay_slices.count[static_cast<std::size_t>(flat)] = static_cast<int>(ids.size());
            layout.pre_delay_slices.synapse_ids.insert(
                layout.pre_delay_slices.synapse_ids.end(), ids.begin(), ids.end());
            running_start += static_cast<int>(ids.size());
        }
    }
    // mask the output neurons
    if (spec->output_neuron_mask.empty()) {
        spec->output_neuron_mask.assign(static_cast<std::size_t>(neuron_count), 0);
        for (std::size_t index = 0; index < spec->output_source_local_ids.size(); ++index) {
            const int source_local_id = spec->output_source_local_ids[index];
            if (source_local_id >= 0 && source_local_id < neuron_count) {
                spec->output_neuron_mask[static_cast<std::size_t>(source_local_id)] = 1;
            }
        }
        for (std::size_t index = 0; index < spec->output_dense_source_local_ids.size(); ++index) {
            const int source_local_id = spec->output_dense_source_local_ids[index];
            if (source_local_id >= 0 && source_local_id < neuron_count) {
                spec->output_neuron_mask[static_cast<std::size_t>(source_local_id)] = 1;
            }
        }
    }

    return spec->IsValid(reason);
}

}  // namespace sim_support
}  // namespace npgr
