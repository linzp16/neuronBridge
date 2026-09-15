#ifndef NPGR_DENSE_SUBNETWORK_LAYOUT_COMPACTOR_H
#define NPGR_DENSE_SUBNETWORK_LAYOUT_COMPACTOR_H

#include "simulation_dense/DenseBuildShared.h"

#include <string>
#include <vector>

namespace npgr {
namespace sim_support {

// Groups neuron ranges with the same factory model id into flatter dense-local id spans.
// The build spec is still a construction artifact; runtime code consumes the
// compacted result and does not retain this object after initialization.
bool CompactDenseSubnetworkByModel(DenseSubnetworkBuildSpec* spec,
                                   std::vector<int>* old_to_new_local = nullptr,
                                   std::string* reason = nullptr);

}  // namespace sim_support
}  // namespace npgr

#endif
