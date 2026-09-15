#include "gpu_runtime/GpuPropagationLayout.h"

#include <sstream>

namespace npgr {

int PreDelaySlices::FlatIndex(int pre_neuron, int delay_slot) const {
    return pre_neuron * delay_slot_count + delay_slot;
}

bool GpuPropagationLayout::IsValid(std::string* reason) const {
    auto fail = [&](const std::string& message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    if (stats.neuron_count < 0 || stats.synapse_count < 0 || stats.delay_slot_count < 0) {
        return fail("layout stats contain negative counts");
    }
    if (static_cast<int>(synapses.post_neuron.size()) != stats.synapse_count) {
        return fail("synapses.post_neuron size does not match synapse_count");
    }
    if (static_cast<int>(synapses.weight.size()) != stats.synapse_count) {
        return fail("synapses.weight size does not match synapse_count");
    }
    if (!synapses.max_weight.empty() &&
        static_cast<int>(synapses.max_weight.size()) != stats.synapse_count) {
        return fail("synapses.max_weight size does not match synapse_count");
    }
    if (static_cast<int>(synapses.effect_channel.size()) != stats.synapse_count) {
        return fail("synapses.effect_channel size does not match synapse_count");
    }
    if (static_cast<int>(synapses.effect_scale.size()) != stats.synapse_count) {
        return fail("synapses.effect_scale size does not match synapse_count");
    }
    auto check_optional_synapse_ints = [&](const std::vector<int>& values, const char* name) {
        if (!values.empty() && static_cast<int>(values.size()) != stats.synapse_count) {
            std::ostringstream oss;
            oss << name << " size does not match synapse_count";
            return fail(oss.str());
        }
        return true;
    };
    auto check_optional_synapse_bytes = [&](const std::vector<unsigned char>& values, const char* name) {
        if (!values.empty() && static_cast<int>(values.size()) != stats.synapse_count) {
            std::ostringstream oss;
            oss << name << " size does not match synapse_count";
            return fail(oss.str());
        }
        return true;
    };
    if (!check_optional_synapse_ints(synapses.plastic_rule_id, "synapses.plastic_rule_id") ||
        !check_optional_synapse_ints(synapses.plastic_model_id, "synapses.plastic_model_id") ||
        !check_optional_synapse_bytes(synapses.plastic_flags, "synapses.plastic_flags") ||
        !check_optional_synapse_ints(synapses.plastic_state_index, "synapses.plastic_state_index") ||
        !check_optional_synapse_ints(synapses.trigger_rule_id, "synapses.trigger_rule_id") ||
        !check_optional_synapse_ints(synapses.trigger_model_id, "synapses.trigger_model_id") ||
        !check_optional_synapse_bytes(synapses.trigger_flags, "synapses.trigger_flags")) {
        return false;
    }
    if (!learning_trigger_routes.start.empty() &&
        static_cast<int>(learning_trigger_routes.start.size()) != stats.synapse_count) {
        return fail("learning_trigger_routes.start size does not match synapse_count");
    }
    if (!learning_trigger_routes.count.empty() &&
        static_cast<int>(learning_trigger_routes.count.size()) != stats.synapse_count) {
        return fail("learning_trigger_routes.count size does not match synapse_count");
    }
    for (int synapse_id = 0;
         synapse_id < static_cast<int>(learning_trigger_routes.start.size());
         ++synapse_id) {
        const int start = learning_trigger_routes.start[static_cast<std::size_t>(synapse_id)];
        const int count = learning_trigger_routes.count[static_cast<std::size_t>(synapse_id)];
        if (start < 0 || count < 0) {
            return fail("learning trigger route contains a negative range");
        }
        if (start + count > static_cast<int>(learning_trigger_routes.synapse_ids.size())) {
            return fail("learning trigger route range exceeds synapse_ids bounds");
        }
    }
    for (int routed_synapse : learning_trigger_routes.synapse_ids) {
        if (routed_synapse < 0 || routed_synapse >= stats.synapse_count) {
            return fail("learning trigger route contains an invalid synapse id");
        }
    }
    if (!learning_spike_buffers.synapse_bucket_id.empty() &&
        static_cast<int>(learning_spike_buffers.synapse_bucket_id.size()) != stats.synapse_count) {
        return fail("learning_spike_buffers.synapse_bucket_id size does not match synapse_count");
    }
    if (learning_spike_buffers.bucket_start.size() != learning_spike_buffers.bucket_capacity.size() ||
        learning_spike_buffers.bucket_start.size() != learning_spike_buffers.bucket_head.size() ||
        learning_spike_buffers.bucket_start.size() != learning_spike_buffers.bucket_count.size()) {
        return fail("learning spike buffer bucket metadata sizes do not match");
    }
    if (learning_spike_buffers.spike_time.size() != learning_spike_buffers.spike_synapse_id.size() ||
        learning_spike_buffers.spike_time.size() != learning_spike_buffers.valid.size()) {
        return fail("learning spike buffer payload sizes do not match");
    }
    for (std::size_t bucket = 0; bucket < learning_spike_buffers.bucket_start.size(); ++bucket) {
        const int start = learning_spike_buffers.bucket_start[bucket];
        const int capacity = learning_spike_buffers.bucket_capacity[bucket];
        if (start < 0 || capacity < 0 ||
            start + capacity > static_cast<int>(learning_spike_buffers.spike_time.size())) {
            return fail("learning spike buffer bucket range exceeds payload bounds");
        }
    }
    for (int bucket_id : learning_spike_buffers.synapse_bucket_id) {
        if (bucket_id < -1 ||
            bucket_id >= static_cast<int>(learning_spike_buffers.bucket_start.size())) {
            return fail("learning spike buffer contains an invalid bucket id");
        }
    }
    if (stats.pending_channel_count < 0) {
        return fail("layout stats contain a negative pending_channel_count");
    }
    if (pre_delay_slices.neuron_count != stats.neuron_count) {
        return fail("pre_delay_slices.neuron_count does not match stats.neuron_count");
    }
    if (pre_delay_slices.delay_slot_count != stats.delay_slot_count) {
        return fail("pre_delay_slices.delay_slot_count does not match stats.delay_slot_count");
    }

    const int expected_slice_cells = stats.neuron_count * stats.delay_slot_count;
    if (static_cast<int>(pre_delay_slices.start.size()) != expected_slice_cells) {
        return fail("pre_delay_slices.start size does not match neuron_count * delay_slot_count");
    }
    if (static_cast<int>(pre_delay_slices.count.size()) != expected_slice_cells) {
        return fail("pre_delay_slices.count size does not match neuron_count * delay_slot_count");
    }

    for (int pre = 0; pre < stats.neuron_count; ++pre) {
        for (int delay = 0; delay < stats.delay_slot_count; ++delay) {
            const int flat = pre_delay_slices.FlatIndex(pre, delay);
            const int start = pre_delay_slices.start[flat];
            const int count = pre_delay_slices.count[flat];
            if (start < 0 || count < 0) {
                std::ostringstream oss;
                oss << "negative pre-delay slice at pre=" << pre << ", delay=" << delay;
                return fail(oss.str());
            }
            if (start + count > static_cast<int>(pre_delay_slices.synapse_ids.size())) {
                std::ostringstream oss;
                oss << "pre-delay slice exceeds synapse_ids bounds at pre=" << pre << ", delay=" << delay;
                return fail(oss.str());
            }
        }
    }

    return true;
}

}  // namespace npgr
