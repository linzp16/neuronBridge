#pragma once

#include "../source_file_realtime_v1_async/Handwriting/inc/PlanarArm2DStage1.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace handwriting {

struct Stage2Config {
    int n_ext = 100;
    int n_cm = 400;
    int n_pop = 8;
    int total_time_ms = 4000;
    double dt_ms = 0.1;
    int refractory_ms = 2;
    float g_ext_cm = 10.0f;
    float g_l = 25.0f;
    float cm_membrane_capacitance = 500.0f;
    float e_ampa = 0.0f;
    float e_leak = -70.0f;
    float v_th = -50.0f;
    float v_reset = -60.0f;
    float v_spike = 20.0f;
    float tau_ampa = 2.0f;
};

struct Shape2D {
    int rows = 0;
    int cols = 0;
};

struct Shape3D {
    int dim0 = 0;
    int dim1 = 0;
    int dim2 = 0;
};

struct BinarySpikeSeries {
    Shape2D shape;
    std::vector<std::uint8_t> values;
};

struct BinaryWeightWindows {
    Shape3D shape;
    std::vector<float> values;
};

struct Stage2StrokeResult {
    std::vector<int> population_spikes;
    TorqueTrajectory decoded_torque;
    SimulatedTrajectory replay;
};

struct CmForwardResult {
    std::vector<std::uint8_t> cm_spikes;
    std::vector<int> population_counts;
};

Shape2D LoadShape2D(const std::filesystem::path& path);
Shape3D LoadShape3D(const std::filesystem::path& path);
BinarySpikeSeries LoadBinarySpikeSeries(const std::filesystem::path& data_path, const std::filesystem::path& shape_path);
BinaryWeightWindows LoadBinaryWeightWindows(const std::filesystem::path& data_path, const std::filesystem::path& shape_path);

std::vector<int> BuildExternalSpikeTimes(const BinarySpikeSeries& spikes);
std::vector<int> BuildExternalSpikeNeurons(const BinarySpikeSeries& spikes);
std::vector<std::array<double, 2> > PopulationDirections(int n_pop);
TorqueTrajectory DecodePopulationSpikes(const std::vector<int>& counts_by_window_and_population, int n_pop);
CmForwardResult SimulateCmForwardExact(
    const Stage2Config& cfg,
    const BinarySpikeSeries& ext_spikes,
    const BinaryWeightWindows& weights);

std::filesystem::path FindExistingStage2Input(const std::filesystem::path& directory, const std::string& stem, const std::string& extension);

}  // namespace handwriting
