#include "simulation_dense/SimulationWeightIO.h"

#include "simulation_dense/DenseSubnetworkModel.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "source_file_realtime_v1_async/Network/inc/Network.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace npgr {
namespace sim_support {
namespace {

// One dense block stores weights in the dense runtime synapse order. The name
// and index are both validated on load so a snapshot cannot silently target a
// different dense subnetwork layout.
struct DenseWeightBlock {
    std::string name;
    int index = -1;
    std::vector<float> weights;
};

// Whole-simulation snapshot. Main-network weights are stored separately from
// dense blocks because they live in different runtime owners.
struct WeightSnapshot {
    int time_step = 0;
    bool has_main = false;
    std::vector<float> main_weights;
    std::vector<DenseWeightBlock> dense_blocks;
};

bool Fail(std::string* reason, const std::string& message) {
    if (reason != nullptr) {
        *reason = message;
    }
    return false;
}

bool CollectMainNetworkWeights(const Network* network,
                               std::vector<float>* weights,
                               std::string* reason) {
    if (network == nullptr || weights == nullptr) {
        return Fail(reason, "main network weight collection received a null pointer");
    }
    weights->clear();
    if (network->intersNum < 0) {
        return Fail(reason, "main network contains a negative interconnection count");
    }
    // Legacy Network stores the live connection weights in wordination order;
    // this is the same order used by main-network propagation.
    weights->reserve(static_cast<std::size_t>(network->intersNum));
    for (int index = 0; index < network->intersNum; ++index) {
        if (network->wordination == nullptr || network->wordination[index] == nullptr) {
            return Fail(reason, "main network contains a null weight storage entry");
        }
        weights->push_back(network->wordination[index]->weight);
    }
    return true;
}

bool ApplyMainNetworkWeights(Network* network,
                             const std::vector<float>& weights,
                             std::string* reason) {
    if (network == nullptr) {
        return Fail(reason, "main network weight apply received a null network");
    }
    if (weights.size() != static_cast<std::size_t>(network->intersNum)) {
        return Fail(reason, "main network weight count does not match intersNum");
    }
    // Validate the whole main-network target first so LoadWeight does not
    // partially apply a malformed snapshot.
    for (int index = 0; index < network->intersNum; ++index) {
        if (network->wordination == nullptr || network->wordination[index] == nullptr) {
            return Fail(reason, "main network contains a null weight storage entry");
        }
    }
    for (int index = 0; index < network->intersNum; ++index) {
        network->wordination[index]->weight = weights[static_cast<std::size_t>(index)];
    }
    return true;
}

bool CollectDenseWeights(const Simulation* simulation,
                         std::vector<DenseWeightBlock>* blocks,
                         std::string* reason) {
    if (simulation == nullptr || blocks == nullptr) {
        return Fail(reason, "dense weight collection received a null pointer");
    }
    blocks->clear();
    blocks->reserve(simulation->dense_subnetworks.size());
    for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
        const DenseSubnetworkModel* dense = simulation->dense_subnetworks[index];
        if (dense == nullptr) {
            return Fail(reason, "simulation contains a null dense subnetwork");
        }
        DenseWeightBlock block;
        block.index = static_cast<int>(index);
        block.name = dense->name();
        // Dense learning updates happen on the GPU, so save from the live
        // device weight buffer instead of the host debug mirror.
        if (!dense->DownloadLiveSynapticWeights(&block.weights, reason)) {
            return false;
        }
        blocks->push_back(block);
    }
    return true;
}

void WriteRleWeights(std::ostream& out, const std::vector<float>& weights) {
    if (weights.empty()) {
        return;
    }
    // Keep exact-equality RLE semantics, matching the legacy weight file
    // behavior while nesting it inside explicit main/dense blocks.
    float previous = weights[0];
    int run_count = 1;
    for (std::size_t index = 1; index <= weights.size(); ++index) {
        if (index < weights.size() && weights[index] == previous) {
            ++run_count;
            continue;
        }
        out << run_count << ' ' << std::setprecision(9) << previous << '\n';
        if (index < weights.size()) {
            previous = weights[index];
            run_count = 1;
        }
    }
}

bool ReadRleWeights(std::istream& in,
                    int expected_count,
                    std::vector<float>* weights,
                    std::string* reason) {
    if (weights == nullptr) {
        return Fail(reason, "RLE weight output vector must not be null");
    }
    if (expected_count < 0) {
        return Fail(reason, "RLE expected_count must not be negative");
    }
    weights->clear();
    weights->reserve(static_cast<std::size_t>(expected_count));
    // RLE blocks terminate only with end_block. Expanding beyond the declared
    // weight count is treated as a hard format error.
    std::string token;
    while (in >> token) {
        if (token == "end_block") {
            return weights->size() == static_cast<std::size_t>(expected_count)
                       ? true
                       : Fail(reason, "RLE block ended with an unexpected weight count");
        }
        int run_count = 0;
        try {
            run_count = std::stoi(token);
        } catch (const std::exception&) {
            return Fail(reason, "RLE block contains an invalid run count");
        }
        float weight = 0.0f;
        if (!(in >> weight)) {
            return Fail(reason, "RLE block is missing a weight value");
        }
        if (run_count <= 0) {
            return Fail(reason, "RLE block contains a non-positive run count");
        }
        if (weights->size() + static_cast<std::size_t>(run_count) >
            static_cast<std::size_t>(expected_count)) {
            return Fail(reason, "RLE block expands beyond its declared weight count");
        }
        weights->insert(weights->end(), static_cast<std::size_t>(run_count), weight);
    }
    return Fail(reason, "RLE block reached end-of-file before end_block");
}

bool WriteSnapshot(const WeightSnapshot& snapshot,
                   const char* filename,
                   std::string* reason) {
    std::ofstream out(filename);
    if (!out) {
        return Fail(reason, "failed to open weight snapshot for writing");
    }
    // New unified format: a required header, one main block, zero or more dense
    // blocks, then end. Old Network-only files are intentionally not emitted.
    out << "NPGR_WEIGHT_SNAPSHOT 1\n";
    out << "time_step " << snapshot.time_step << "\n\n";
    out << "block main 0 " << snapshot.main_weights.size() << '\n';
    out << "rle\n";
    WriteRleWeights(out, snapshot.main_weights);
    out << "end_block\n\n";
    for (const DenseWeightBlock& block : snapshot.dense_blocks) {
        out << "block dense " << block.index << ' ' << std::quoted(block.name) << ' '
            << block.weights.size() << '\n';
        out << "rle\n";
        WriteRleWeights(out, block.weights);
        out << "end_block\n\n";
    }
    out << "end\n";
    return static_cast<bool>(out) ? true : Fail(reason, "failed while writing weight snapshot");
}

bool ReadSnapshot(const char* filename,
                  WeightSnapshot* snapshot,
                  std::string* reason) {
    if (snapshot == nullptr) {
        return Fail(reason, "weight snapshot output must not be null");
    }
    std::ifstream in(filename);
    if (!in) {
        return Fail(reason, "failed to open weight snapshot for reading");
    }
    std::string magic;
    int version = 0;
    // No compatibility probing: LoadWeight now requires the unified snapshot
    // header so main and dense weights cannot be confused.
    if (!(in >> magic >> version) || magic != "NPGR_WEIGHT_SNAPSHOT" || version != 1) {
        return Fail(reason, "weight file is not an NPGR_WEIGHT_SNAPSHOT v1 file");
    }
    std::string label;
    if (!(in >> label >> snapshot->time_step) || label != "time_step") {
        return Fail(reason, "weight snapshot is missing time_step");
    }
    snapshot->has_main = false;
    snapshot->main_weights.clear();
    snapshot->dense_blocks.clear();
    std::string token;
    while (in >> token) {
        if (token == "end") {
            return snapshot->has_main ? true : Fail(reason, "weight snapshot is missing main block");
        }
        if (token != "block") {
            return Fail(reason, "weight snapshot contains an unexpected token");
        }
        std::string block_kind;
        if (!(in >> block_kind)) {
            return Fail(reason, "weight snapshot block is missing a kind");
        }
        if (block_kind == "main") {
            int block_index = -1;
            int weight_count = -1;
            if (!(in >> block_index >> weight_count) || block_index != 0) {
                return Fail(reason, "main weight block header is invalid");
            }
            if (snapshot->has_main) {
                return Fail(reason, "weight snapshot contains duplicate main blocks");
            }
            if (!(in >> label) || label != "rle") {
                return Fail(reason, "main weight block is missing rle marker");
            }
            if (!ReadRleWeights(in, weight_count, &snapshot->main_weights, reason)) {
                return false;
            }
            snapshot->has_main = true;
        } else if (block_kind == "dense") {
            DenseWeightBlock block;
            int weight_count = -1;
            if (!(in >> block.index >> std::quoted(block.name) >> weight_count)) {
                return Fail(reason, "dense weight block header is invalid");
            }
            if (!(in >> label) || label != "rle") {
                return Fail(reason, "dense weight block is missing rle marker");
            }
            if (!ReadRleWeights(in, weight_count, &block.weights, reason)) {
                return false;
            }
            snapshot->dense_blocks.push_back(block);
        } else {
            return Fail(reason, "weight snapshot contains an unknown block kind");
        }
    }
    return Fail(reason, "weight snapshot reached end-of-file before end marker");
}

bool ValidateSnapshotForSimulation(const WeightSnapshot& snapshot,
                                   const Simulation* simulation,
                                   std::string* reason) {
    if (simulation == nullptr || simulation->network == nullptr) {
        return Fail(reason, "simulation or main network is null");
    }
    if (!snapshot.has_main) {
        return Fail(reason, "weight snapshot is missing main block");
    }
    if (snapshot.main_weights.size() != static_cast<std::size_t>(simulation->network->intersNum)) {
        return Fail(reason, "main weight block count does not match simulation network");
    }
    if (snapshot.dense_blocks.size() != simulation->dense_subnetworks.size()) {
        return Fail(reason, "dense weight block count does not match simulation dense subnetworks");
    }
    // Validate dense identity and counts before applying any main or dense
    // weights. This makes LoadWeight all-or-nothing at the structural level.
    for (std::size_t index = 0; index < snapshot.dense_blocks.size(); ++index) {
        const DenseWeightBlock& block = snapshot.dense_blocks[index];
        if (block.index != static_cast<int>(index)) {
            return Fail(reason, "dense weight block index does not match block order");
        }
        const DenseSubnetworkModel* dense = simulation->dense_subnetworks[index];
        if (dense == nullptr) {
            return Fail(reason, "simulation contains a null dense subnetwork");
        }
        if (block.name != dense->name()) {
            return Fail(reason, "dense weight block name does not match simulation dense subnetwork");
        }
        std::vector<float> current_weights;
        // Download only to learn the live runtime synapse count. The contents
        // are not used for comparison because the snapshot is allowed to change weights.
        if (!dense->DownloadLiveSynapticWeights(&current_weights, reason)) {
            return false;
        }
        if (block.weights.size() != current_weights.size()) {
            return Fail(reason, "dense weight block count does not match runtime synapse_count");
        }
    }
    return true;
}

bool ApplyDenseWeights(Simulation* simulation,
                       const std::vector<DenseWeightBlock>& blocks,
                       std::string* reason) {
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        DenseSubnetworkModel* dense = simulation->dense_subnetworks[index];
        if (dense == nullptr) {
            return Fail(reason, "simulation contains a null dense subnetwork");
        }
        if (!dense->UploadLiveSynapticWeights(blocks[index].weights, reason)) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool SaveSimulationWeights(const Simulation* simulation,
                           const char* filename,
                           std::string* reason) {
    if (simulation == nullptr || filename == nullptr || filename[0] == '\0') {
        return Fail(reason, "save weight requires a simulation and a non-empty filename");
    }
    WeightSnapshot snapshot;
    snapshot.time_step = simulation->currenttime != nullptr ? simulation->currenttime[0] : 0;
    snapshot.has_main = true;
    // Collect both owners first, then write a single snapshot file.
    if (!CollectMainNetworkWeights(simulation->network, &snapshot.main_weights, reason) ||
        !CollectDenseWeights(simulation, &snapshot.dense_blocks, reason)) {
        return false;
    }
    return WriteSnapshot(snapshot, filename, reason);
}

bool LoadSimulationWeights(Simulation* simulation,
                           const char* filename,
                           std::string* reason) {
    if (simulation == nullptr || filename == nullptr || filename[0] == '\0') {
        return Fail(reason, "load weight requires a simulation and a non-empty filename");
    }
    WeightSnapshot snapshot;
    // Parse and validate first. Only after that do we mutate main-network or
    // dense GPU weights.
    if (!ReadSnapshot(filename, &snapshot, reason) ||
        !ValidateSnapshotForSimulation(snapshot, simulation, reason)) {
        return false;
    }
    if (!ApplyMainNetworkWeights(simulation->network, snapshot.main_weights, reason)) {
        return false;
    }
    return ApplyDenseWeights(simulation, snapshot.dense_blocks, reason);
}

}  // namespace sim_support
}  // namespace npgr
