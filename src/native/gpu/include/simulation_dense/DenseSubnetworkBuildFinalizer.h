#ifndef NPGR_DENSE_SUBNETWORK_BUILD_FINALIZER_H
#define NPGR_DENSE_SUBNETWORK_BUILD_FINALIZER_H

#include "simulation_dense/DenseBuildShared.h"

#include <string>

namespace npgr {
namespace sim_support {

// Completes the derived GPU scheduling tables inside an already prepared dense
// spec. The spec owns the dense graph; this function only generates
// layout.pre_delay_slices from synapse_source_local_ids/synapse_delay_slots.
bool FinalizeDenseSubnetworkBuildSpec(DenseSubnetworkBuildSpec* spec,
                                      std::string* reason = nullptr);

}  // namespace sim_support
}  // namespace npgr

#endif
