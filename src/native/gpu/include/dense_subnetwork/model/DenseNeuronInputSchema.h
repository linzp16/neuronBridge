#ifndef NPGR_DENSE_NEURON_INPUT_SCHEMA_H
#define NPGR_DENSE_NEURON_INPUT_SCHEMA_H

#include "gpu_runtime/GpuPropagationLayout.h"

namespace npgr {

// Channel consumed by a neuron model update equation.
struct DenseInputChannelBinding {
    PendingChannel channel = PendingChannel::ExcitatoryConductance;
    const char* semantic_name = "";
    bool required = false;
};

// Compile-time rule: a synapse type targeting this model writes a pending channel.
struct DenseSpikeEffectBinding {
    int synapse_type = 0;
    PendingChannel channel = PendingChannel::ExcitatoryConductance;
    float scale = 1.0f;
};

}  // namespace npgr

#endif
