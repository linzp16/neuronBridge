#ifndef NPGR_DENSE_SUBNETWORK_OUTPUT_EXTRACTOR_H
#define NPGR_DENSE_SUBNETWORK_OUTPUT_EXTRACTOR_H

#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"

#include <vector>

namespace npgr {

class DenseSubnetworkOutputExtractor {
public:
    static std::vector<DenseOutputRouteEntry> BuildOutputRoutesBySourceNeuron(
        const sim_support::DenseSubnetworkBuildSpec& spec,
        int neuron_count);

    static std::vector<DenseOutputSpike> Extract(const std::vector<DenseOutputRouteEntry>& output_routes_by_source_neuron,
                                                 const std::vector<int>& firing_ids,
                                                 int time_step);
};

}  // namespace npgr

#endif
