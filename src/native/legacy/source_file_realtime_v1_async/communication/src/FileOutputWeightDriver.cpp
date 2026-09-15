#include "../source_file_realtime_v1_async/communication/inc/FileOutputWeightDriver.h"

#include "simulation_dense/SimulationWeightIO.h"

#include <iostream>
#include <string>

FileOutputWeightDriver::FileOutputWeightDriver(const char* newfilename) : OutputWeightDriver() {
    this->filename = newfilename;
}

FileOutputWeightDriver::~FileOutputWeightDriver() {}

void FileOutputWeightDriver::WriteWeight(Simulation* simulation, int time) {
    // Preserve the legacy event-driver naming convention: insert the timestep
    // before the extension, then write the new unified main+dense snapshot.
    std::string name = this->filename;
    const std::size_t dot_pos = name.find_last_of('.');
    const std::string time_str = std::to_string(time);
    if (dot_pos != std::string::npos) {
        name.insert(dot_pos, time_str);
    } else {
        name += time_str;
    }

    std::string reason;
    if (!npgr::sim_support::SaveSimulationWeights(simulation, name.c_str(), &reason)) {
        std::cerr << "SaveWeight failed: " << reason << std::endl;
    }
}
