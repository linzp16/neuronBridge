#ifndef NPGR_SIMULATION_WEIGHT_IO_H
#define NPGR_SIMULATION_WEIGHT_IO_H

#include <string>

class Simulation;

namespace npgr {
namespace sim_support {

// Saves a complete v1 weight snapshot for the legacy main network and all dense subnetworks.
// The file format intentionally does not accept the old Network-only RLE files.
bool SaveSimulationWeights(const Simulation* simulation,
                           const char* filename,
                           std::string* reason = nullptr);

// Loads a complete v1 weight snapshot into the legacy main network and all dense subnetworks.
// The snapshot is fully parsed and validated before any runtime weight is modified.
bool LoadSimulationWeights(Simulation* simulation,
                           const char* filename,
                           std::string* reason = nullptr);

}  // namespace sim_support
}  // namespace npgr

#endif
