#pragma once

#include "../source_file_realtime_v1_async/Handwriting/inc/PlanarArm2DStage1.h"
#include "../source_file_realtime_v1_async/Handwriting/inc/HandwritingStage2.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace handwriting {

enum class Stage3Mode {
    Test,
    TestD,
    TrainG,
    TrainD
};

struct Stage3Config {
    int n_ext = 100;
    int n_cm = 400;
    int n_bg = 10;
    int n_bg_total = 60;
    int n_sm_group = 10;
    int n_sm_total = 20;
    int n_mm = 500;
    int n_e = 1000;
    int n_i = 250;
    int n_sig = 20;
    int n_pop = 8;
    int n_rcv = 20;
    int total_time_ms = 9000;
    double dt_ms = 0.1;
    double verify_dt_s = 0.01;
    int refractory_ms = 2;
    double v_th = -50.0;
    double v_reset = -60.0;
    double v_spike = 20.0;
    double e_ampa = 0.0;
    double e_nmda = 0.0;
    double e_gaba = -70.0;
    double e_leak = -70.0;
    double cm = 500.0;
    double g_l = 25.0;
    double tau_ampa = 2.0;
    double tau_nmda = 100.0;
    double tau_gaba = 10.0;
    double g_ext_cm = 10.0;
    double g_bg_cm = 550.0;
    double g_cm_bg = 4.0;
    double g_sm_bg = 6.0;
    double g_mm_bg = 8.0;
    double g_ext_mm = 20.0;
    double g_e_mm = 20.0;
    double g_ext_e = 42.2;
    double g_ext_i = 40.0;
    double g_ee = 18.2;
    double g_ei = 18.2;
    double g_ie = 16.7;
    double g_ii = 16.7;
    double g_sig_e = 40.0;
    double g_cm_e = 40.0;
    double hebb_increment = 0.1;
};

struct MatrixF32 {
    int rows = 0;
    int cols = 0;
    std::vector<float> values;
};

struct Stage3Data {
    BinarySpikeSeries sspk_ext;
    BinarySpikeSeries sspk_ext_e;
    BinarySpikeSeries sspk_ext_i;
    BinarySpikeSeries sspk_ext_mm;
    BinarySpikeSeries sspk_sig;
    BinarySpikeSeries sspk_sm;
    BinaryWeightWindows ww_extcm1;
    BinaryWeightWindows ww_extcm2;
    BinaryWeightWindows ww_extcm3;
    MatrixF32 c_emm1;
    MatrixF32 c_emm2;
    MatrixF32 c_mmbg1;
    MatrixF32 c_mmbg2;
    MatrixF32 w_mmbg1;
    MatrixF32 w_mmbg2;
    MatrixF32 c_cm1e;
    MatrixF32 c_cm2e;
    MatrixF32 c_cm3e;
    MatrixF32 c_sig_e;
    MatrixF32 w_ee;
    MatrixF32 c_emm_d;
    MatrixF32 c_mmbg_d;
    MatrixF32 w_mmbg_d;
    std::vector<double> a1_flat;
    std::vector<double> a2_flat;
    std::vector<double> a3_flat;
    StrokePath x1y1;
    StrokePath x2y2;
    StrokePath x3y3;
};

struct Stage3Result {
    TorqueTrajectory qq;
    SimulatedTrajectory replay;
    std::vector<int> pop_spk;
    MatrixF32 w_mmbg1;
    MatrixF32 w_mmbg2;
};

Stage3Mode ParseStage3Mode(const std::string& text);
Stage3Data LoadStage3Data(const std::filesystem::path& input_directory, bool require_test_weights);
Stage3Result RunStage3(const Stage3Config& cfg, const Stage3Data& data, Stage3Mode mode);
void WriteStage3Outputs(const std::filesystem::path& output_directory, const Stage3Result& result, const StrokePath& target_path);

}  // namespace handwriting
