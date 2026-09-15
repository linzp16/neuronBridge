#ifndef NPGR_SIMULATION_CONNECTION_WEIGHT_REF_H
#define NPGR_SIMULATION_CONNECTION_WEIGHT_REF_H

namespace npgr {

enum class RuntimeConnectionWeightOwner {
    // The original connection became a non-owned boundary route. Its weight is
    // immutable at runtime, so queries return the construction-time value.
    InitialOnly,
    // The original connection is owned by the legacy main Network.
    MainNetwork,
    // The original connection is owned by one GPU dense subnetwork.
    DenseInternal,
};

struct RuntimeConnectionWeightRef {
    // Identifies which runtime storage, if any, owns the live weight.
    RuntimeConnectionWeightOwner owner = RuntimeConnectionWeightOwner::InitialOnly;
    // Construction-time weight used for immutable boundary routes.
    float initial_weight = 0.0f;
    // Dense subnetwork index when owner is DenseInternal.
    int dense_subnetwork_index = -1;
    // Main wordination index or dense local synapse index, depending on owner.
    int runtime_weight_index = -1;
};

}  // namespace npgr

#endif
