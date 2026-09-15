#include "../source_file_realtime_v1_async/Handwriting/inc/HandwritingStage3.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>

namespace handwriting {

namespace {

using SpikeVector = std::vector<std::uint8_t>;

struct LayerState {
    std::vector<double> vm;
    std::vector<int> last_spike_step;
};

struct NetworkState {
    LayerState cm1;
    LayerState cm2;
    LayerState cm3;
    LayerState bg;
    LayerState mm1;
    LayerState mm2;
    LayerState e;
    LayerState i;

    std::vector<double> s_cm1;
    std::vector<double> s_cm2;
    std::vector<double> s_cm3;
    std::vector<double> s_bg;
    std::vector<double> s_mm1;
    std::vector<double> s_mm2;
    std::vector<double> s_e;
    std::vector<double> s_i;
};

std::filesystem::path RequirePath(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Required input file not found: " + path.string());
    }
    return path;
}

bool Exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path);
}

MatrixF32 LoadMatrixF32(const std::filesystem::path& data_path, const std::filesystem::path& shape_path) {
    const Shape2D shape = LoadShape2D(shape_path);
    const std::size_t count = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    std::ifstream in(data_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open binary matrix " + data_path.string());
    }
    MatrixF32 matrix;
    matrix.rows = shape.rows;
    matrix.cols = shape.cols;
    matrix.values.resize(count);
    in.read(reinterpret_cast<char*>(matrix.values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error("Failed to read matrix " + data_path.string());
    }
    return matrix;
}

std::vector<double> LoadMatrixTextFirstColumn(const std::filesystem::path& path) {
    return LoadNumericSeries(path);
}

std::vector<double> LoadMatrixTextAll(const std::filesystem::path& path) {
    return LoadNumericSeries(path);
}

inline std::uint8_t SpikeAt(const BinarySpikeSeries& spikes, int row, int col) {
    return spikes.values[static_cast<std::size_t>(col) * static_cast<std::size_t>(spikes.shape.rows) + static_cast<std::size_t>(row)];
}

inline float WeightWindowAt(const BinaryWeightWindows& weights, int source, int window, int target) {
    const std::size_t block = static_cast<std::size_t>(weights.shape.dim0) * static_cast<std::size_t>(weights.shape.dim2);
    const std::size_t offset = static_cast<std::size_t>(window) * block;
    const std::size_t local = static_cast<std::size_t>(target) * static_cast<std::size_t>(weights.shape.dim0) + static_cast<std::size_t>(source);
    return weights.values[offset + local];
}

inline float MatrixAt(const MatrixF32& matrix, int row, int col) {
    return matrix.values[static_cast<std::size_t>(col) * static_cast<std::size_t>(matrix.rows) + static_cast<std::size_t>(row)];
}

inline float& MatrixAtRef(MatrixF32& matrix, int row, int col) {
    return matrix.values[static_cast<std::size_t>(col) * static_cast<std::size_t>(matrix.rows) + static_cast<std::size_t>(row)];
}

template <typename T>
std::vector<T> Filled(int count, T value) {
    return std::vector<T>(static_cast<std::size_t>(count), value);
}

void UpdateAMPA(std::vector<double>& s, const std::vector<double>& input, double dt, double tau) {
    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] += (-s[i] / tau + input[i]) * dt;
    }
}

void UpdateGABA(std::vector<double>& s, const std::vector<double>& input, double dt, double tau) {
    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] += (-s[i] / tau + input[i]) * dt;
    }
}

void UpdateNMDA(std::vector<double>& s, const std::vector<double>& input, double dt, double tau) {
    constexpr double alpha = 0.63;
    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] += (-s[i] / tau + alpha * (1.0 - s[i]) * input[i]) * dt;
    }
}

void ApplyIFAndRefractory(
    LayerState& layer,
    SpikeVector& spikes_out,
    int step,
    double v_th,
    double v_reset,
    double v_spike,
    int refractory_steps) {
    for (std::size_t i = 0; i < layer.vm.size(); ++i) {
        double v = layer.vm[i];
        std::uint8_t spike = 0;
        if (v >= v_th) {
            v = v_spike;
            layer.vm[i] = v_reset;
            spike = 1;
        }
        if (layer.last_spike_step[i] >= 0 && step - layer.last_spike_step[i] <= refractory_steps) {
            spike = 0;
            v = v_reset;
            layer.vm[i] = v_reset;
        }
        if (spike != 0) {
            layer.last_spike_step[i] = step;
        }
        spikes_out[i] = spike;
        (void)v;
    }
}

MatrixF32 RandomBinaryConnectivity(int rows, int cols, double probability, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    MatrixF32 matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.values.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0.0f);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            MatrixAtRef(matrix, row, col) = dist(rng) <= probability ? 1.0f : 0.0f;
        }
    }
    return matrix;
}

MatrixF32 StructuredConnectivityEE(int n_e) {
    MatrixF32 matrix;
    matrix.rows = n_e;
    matrix.cols = n_e;
    matrix.values.resize(static_cast<std::size_t>(n_e) * static_cast<std::size_t>(n_e), 0.0f);
    for (int col = 0; col < n_e; ++col) {
        for (int row = 0; row < n_e; ++row) {
            MatrixAtRef(matrix, row, col) = static_cast<float>(std::exp(-static_cast<double>((col - row) * (col - row)) / 3000.0));
        }
    }
    return matrix;
}

MatrixF32 ConnectivityBgCM(int n_cm, int n_bg, int which) {
    MatrixF32 m;
    m.rows = n_cm;
    m.cols = 6 * n_bg;
    m.values.resize(static_cast<std::size_t>(m.rows) * static_cast<std::size_t>(m.cols), 0.0f);
    for (int row = 0; row < n_cm; ++row) {
        for (int block = 0; block < 6; ++block) {
            const bool on =
                (which == 1 && (block == 1 || block == 2 || block == 4 || block == 5)) ||
                (which == 2 && (block == 0 || block == 2 || block == 3 || block == 5)) ||
                (which == 3 && (block == 0 || block == 1 || block == 3 || block == 4));
            for (int j = 0; j < n_bg; ++j) {
                MatrixAtRef(m, row, block * n_bg + j) = on ? 1.0f : 0.0f;
            }
        }
    }
    return m;
}

MatrixF32 ConnectivityCMbg(int n_cm, int n_bg, int which) {
    MatrixF32 m;
    m.rows = 6 * n_bg;
    m.cols = n_cm;
    m.values.resize(static_cast<std::size_t>(m.rows) * static_cast<std::size_t>(m.cols), 0.0f);
    for (int block = 0; block < 6; ++block) {
        const bool on =
            (which == 1 && (block == 0 || block == 3)) ||
            (which == 2 && (block == 1 || block == 4)) ||
            (which == 3 && (block == 2 || block == 5));
        for (int r = 0; r < n_bg; ++r) {
            for (int c = 0; c < n_cm; ++c) {
                MatrixAtRef(m, block * n_bg + r, c) = on ? 1.0f : 0.0f;
            }
        }
    }
    return m;
}

MatrixF32 ConnectivitySMbg(int n_sm_group, int n_bg) {
    MatrixF32 m;
    m.rows = 6 * n_bg;
    m.cols = 2 * n_sm_group;
    m.values.resize(static_cast<std::size_t>(m.rows) * static_cast<std::size_t>(m.cols), 0.0f);
    for (int r = 0; r < n_bg; ++r) {
        if (r < n_sm_group) {
            MatrixAtRef(m, 0 * n_bg + r, r) = 1.0f;
            MatrixAtRef(m, 1 * n_bg + r, r) = 1.0f;
            MatrixAtRef(m, 2 * n_bg + r, r) = 1.0f;
            MatrixAtRef(m, 3 * n_bg + r, n_sm_group + r) = 1.0f;
            MatrixAtRef(m, 4 * n_bg + r, n_sm_group + r) = 1.0f;
            MatrixAtRef(m, 5 * n_bg + r, n_sm_group + r) = 1.0f;
        }
    }
    return m;
}

MatrixF32 ConnectivityCME(int n_cm, int n_e, int n_rcv, int center) {
    MatrixF32 m;
    m.rows = n_e;
    m.cols = n_cm;
    m.values.resize(static_cast<std::size_t>(m.rows) * static_cast<std::size_t>(m.cols), 0.0f);
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int local_row = 0; local_row < n_rcv; ++local_row) {
        const int row = center - n_rcv / 2 + local_row;
        for (int col = 0; col < n_cm; ++col) {
            MatrixAtRef(m, row, col) = dist(rng) <= 0.5 ? 1.0f : 0.0f;
        }
    }
    return m;
}

MatrixF32 ConnectivitySignalE(int n_sig, int n_e, int n_rcv) {
    MatrixF32 m;
    m.rows = n_e;
    m.cols = n_sig;
    m.values.resize(static_cast<std::size_t>(m.rows) * static_cast<std::size_t>(m.cols), 0.0f);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int local_row = 0; local_row < n_rcv; ++local_row) {
        const int row = 800 - n_rcv / 2 + local_row;
        for (int col = 0; col < n_sig; ++col) {
            MatrixAtRef(m, row, col) = dist(rng) <= 0.5 ? 1.0f : 0.0f;
        }
    }
    return m;
}

std::vector<int> BuildWindowIndexSequence(int window_count) {
    std::vector<int> w;
    w.reserve(static_cast<std::size_t>(window_count));
    for (int i = 0; i < 50; ++i) {
        w.push_back(0);
    }
    for (int i = 0; i < 400; ++i) {
        w.push_back(i);
    }
    for (int i = 0; i < 50; ++i) {
        w.push_back(0);
    }
    for (int i = 0; i < 400; ++i) {
        w.push_back(i);
    }
    return w;
}

bool ExtActiveForTrain(Stage3Mode mode, int channel, int step, int ext_steps, int total_steps) {
    const int offset_short = 5000;
    const int offset_long = 50000;
    if (channel == 0) {
        return step >= offset_short && step < offset_short + ext_steps;
    }
    if (mode == Stage3Mode::TrainG) {
        if (channel == 1) {
            return step >= offset_long && step < offset_long + ext_steps;
        }
        return false;
    }
    if (mode == Stage3Mode::TrainD) {
        if (channel == 2) {
            return step >= offset_long && step < std::min(total_steps, offset_long + ext_steps);
        }
        return false;
    }
    return false;
}

int LocalExtStepForTrain(Stage3Mode mode, int channel, int step) {
    if (channel == 0) {
        return step - 5000;
    }
    if (mode == Stage3Mode::TrainG && channel == 1) {
        return step - 50000;
    }
    if (mode == Stage3Mode::TrainD && channel == 2) {
        return step - 50000;
    }
    return -1;
}

int LocalExtStepForTestG(int step) {
    if (step >= 5000 && step < 45000) {
        return step - 5000;
    }
    if (step >= 50000 && step < 90000) {
        return step - 50000;
    }
    return -1;
}

int LocalExtStepForTestD(Stage3Mode mode, int channel, int step) {
    if (mode != Stage3Mode::TestD) {
        return -1;
    }
    if (channel == 0 && step >= 5000 && step < 45000) {
        return step - 5000;
    }
    if (channel == 2 && step >= 50000 && step < 90000) {
        return step - 50000;
    }
    return -1;
}

SpikeVector BuildCmExternalSpike(
    const BinarySpikeSeries& ext,
    Stage3Mode mode,
    bool is_test,
    int channel,
    int step,
    int n_ext) {
    SpikeVector out(static_cast<std::size_t>(n_ext), 0);
    int local_step = -1;
    if (is_test) {
        if (mode == Stage3Mode::TestD) {
            local_step = LocalExtStepForTestD(mode, channel, step);
        } else {
            local_step = LocalExtStepForTestG(step);
        }
    } else {
        local_step = LocalExtStepForTrain(mode, channel, step);
    }
    if (local_step < 0 || local_step >= ext.shape.cols) {
        return out;
    }
    for (int i = 0; i < n_ext; ++i) {
        out[static_cast<std::size_t>(i)] = SpikeAt(ext, i, local_step);
    }
    return out;
}

void UpdateCM(
    const Stage3Config& cfg,
    const BinaryWeightWindows& weights,
    int window_index,
    const MatrixF32& c_bgcm,
    const SpikeVector& ext_spike,
    const SpikeVector& bg_spike,
    int step,
    LayerState& layer,
    std::vector<double>& s_state,
    SpikeVector& out_spike) {
    std::vector<double> input_ext(static_cast<std::size_t>(cfg.n_cm), 0.0);
    for (int source = 0; source < cfg.n_ext; ++source) {
        if (ext_spike[static_cast<std::size_t>(source)] == 0) {
            continue;
        }
        for (int target = 0; target < cfg.n_cm; ++target) {
            input_ext[static_cast<std::size_t>(target)] += WeightWindowAt(weights, source, window_index, target);
        }
    }

    std::vector<double> input_bg(static_cast<std::size_t>(cfg.n_cm), 0.0);
    for (int target = 0; target < cfg.n_cm; ++target) {
        double total = 0.0;
        for (int source = 0; source < cfg.n_bg_total; ++source) {
            total += static_cast<double>(MatrixAt(c_bgcm, target, source)) * static_cast<double>(bg_spike[static_cast<std::size_t>(source)]);
        }
        input_bg[static_cast<std::size_t>(target)] = total;
    }

    std::vector<double> s_ampa_ext(s_state.begin(), s_state.begin() + cfg.n_cm);
    std::vector<double> s_gaba_bg(s_state.begin() + cfg.n_cm, s_state.end());
    UpdateAMPA(s_ampa_ext, input_ext, cfg.dt_ms, cfg.tau_ampa);
    UpdateGABA(s_gaba_bg, input_bg, cfg.dt_ms, cfg.tau_gaba);

    for (int i = 0; i < cfg.n_cm; ++i) {
        const double vm = layer.vm[static_cast<std::size_t>(i)];
        const double i_ex_ext = -cfg.g_ext_cm * s_ampa_ext[static_cast<std::size_t>(i)] * (vm - cfg.e_ampa);
        const double i_ex_bg = -cfg.g_bg_cm * s_gaba_bg[static_cast<std::size_t>(i)] * (vm - cfg.e_gaba);
        const double i_in = -cfg.g_l * (vm - cfg.e_leak);
        layer.vm[static_cast<std::size_t>(i)] += (i_in + i_ex_ext + i_ex_bg) / cfg.cm * cfg.dt_ms;
    }

    ApplyIFAndRefractory(layer, out_spike, step, cfg.v_th, cfg.v_reset, cfg.v_spike, static_cast<int>(cfg.refractory_ms / cfg.dt_ms));
    std::copy(s_ampa_ext.begin(), s_ampa_ext.end(), s_state.begin());
    std::copy(s_gaba_bg.begin(), s_gaba_bg.end(), s_state.begin() + cfg.n_cm);
}

void UpdateBG(
    const Stage3Config& cfg,
    const MatrixF32& c_cm1bg,
    const MatrixF32& c_cm2bg,
    const MatrixF32& c_cm3bg,
    const MatrixF32& c_smbg,
    const MatrixF32& c_mmbg1,
    MatrixF32& w_mmbg1,
    const MatrixF32& c_mmbg2,
    MatrixF32& w_mmbg2,
    const SpikeVector& spk_cm1,
    const SpikeVector& spk_cm2,
    const SpikeVector& spk_cm3,
    const SpikeVector& spk_sm,
    const SpikeVector& spk_mm1,
    const SpikeVector& spk_mm2,
    int step,
    LayerState& layer,
    std::vector<double>& s_state,
    SpikeVector& out_spike) {
    const int n = cfg.n_bg_total;
    std::vector<double> input_cm1(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_cm2(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_cm3(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_sm(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_mm1(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_mm2(static_cast<std::size_t>(n), 0.0);

    for (int row = 0; row < n; ++row) {
        double total1 = 0.0, total2 = 0.0, total3 = 0.0, totalsm = 0.0, totalm1 = 0.0, totalm2 = 0.0;
        for (int col = 0; col < cfg.n_cm; ++col) {
            total1 += MatrixAt(c_cm1bg, row, col) * spk_cm1[static_cast<std::size_t>(col)];
            total2 += MatrixAt(c_cm2bg, row, col) * spk_cm2[static_cast<std::size_t>(col)];
            total3 += MatrixAt(c_cm3bg, row, col) * spk_cm3[static_cast<std::size_t>(col)];
        }
        for (int col = 0; col < cfg.n_sm_total; ++col) {
            totalsm += MatrixAt(c_smbg, row, col) * spk_sm[static_cast<std::size_t>(col)];
        }
        for (int col = 0; col < cfg.n_mm; ++col) {
            totalm1 += MatrixAt(c_mmbg1, row, col) * MatrixAt(w_mmbg1, row, col) * spk_mm1[static_cast<std::size_t>(col)];
            totalm2 += MatrixAt(c_mmbg2, row, col) * MatrixAt(w_mmbg2, row, col) * spk_mm2[static_cast<std::size_t>(col)];
        }
        input_cm1[static_cast<std::size_t>(row)] = total1;
        input_cm2[static_cast<std::size_t>(row)] = total2;
        input_cm3[static_cast<std::size_t>(row)] = total3;
        input_sm[static_cast<std::size_t>(row)] = totalsm;
        input_mm1[static_cast<std::size_t>(row)] = totalm1;
        input_mm2[static_cast<std::size_t>(row)] = totalm2;
    }

    auto seg = [&](int start) { return std::vector<double>(s_state.begin() + start * n, s_state.begin() + (start + 1) * n); };
    std::vector<double> s_ampa_cm1 = seg(0), s_nmda_cm1 = seg(1), s_ampa_cm2 = seg(2), s_nmda_cm2 = seg(3),
                        s_ampa_cm3 = seg(4), s_nmda_cm3 = seg(5), s_ampa_sm = seg(6), s_nmda_sm = seg(7),
                        s_ampa_mm1 = seg(8), s_nmda_mm1 = seg(9), s_ampa_mm2 = seg(10), s_nmda_mm2 = seg(11);

    UpdateAMPA(s_ampa_cm1, input_cm1, cfg.dt_ms, cfg.tau_ampa); UpdateNMDA(s_nmda_cm1, input_cm1, cfg.dt_ms, cfg.tau_nmda);
    UpdateAMPA(s_ampa_cm2, input_cm2, cfg.dt_ms, cfg.tau_ampa); UpdateNMDA(s_nmda_cm2, input_cm2, cfg.dt_ms, cfg.tau_nmda);
    UpdateAMPA(s_ampa_cm3, input_cm3, cfg.dt_ms, cfg.tau_ampa); UpdateNMDA(s_nmda_cm3, input_cm3, cfg.dt_ms, cfg.tau_nmda);
    UpdateAMPA(s_ampa_sm, input_sm, cfg.dt_ms, cfg.tau_ampa); UpdateNMDA(s_nmda_sm, input_sm, cfg.dt_ms, cfg.tau_nmda);
    UpdateAMPA(s_ampa_mm1, input_mm1, cfg.dt_ms, cfg.tau_ampa); UpdateNMDA(s_nmda_mm1, input_mm1, cfg.dt_ms, cfg.tau_nmda);
    UpdateAMPA(s_ampa_mm2, input_mm2, cfg.dt_ms, cfg.tau_ampa); UpdateNMDA(s_nmda_mm2, input_mm2, cfg.dt_ms, cfg.tau_nmda);

    for (int i = 0; i < n; ++i) {
        const double vm = layer.vm[static_cast<std::size_t>(i)];
        const auto nmda = [&](double g, double sa, double sn) {
            return -g * sa * (vm - cfg.e_ampa) - g * sn * (vm - cfg.e_nmda) / (1.0 + std::exp(-0.062 * vm / 3.57));
        };
        const double i_ex =
            nmda(cfg.g_cm_bg, s_ampa_cm1[static_cast<std::size_t>(i)], s_nmda_cm1[static_cast<std::size_t>(i)]) +
            nmda(cfg.g_cm_bg, s_ampa_cm2[static_cast<std::size_t>(i)], s_nmda_cm2[static_cast<std::size_t>(i)]) +
            nmda(cfg.g_cm_bg, s_ampa_cm3[static_cast<std::size_t>(i)], s_nmda_cm3[static_cast<std::size_t>(i)]) +
            nmda(cfg.g_sm_bg, s_ampa_sm[static_cast<std::size_t>(i)], s_nmda_sm[static_cast<std::size_t>(i)]) +
            nmda(cfg.g_mm_bg, s_ampa_mm1[static_cast<std::size_t>(i)], s_nmda_mm1[static_cast<std::size_t>(i)]) +
            nmda(cfg.g_mm_bg, s_ampa_mm2[static_cast<std::size_t>(i)], s_nmda_mm2[static_cast<std::size_t>(i)]);
        const double i_in = -cfg.g_l * (vm - cfg.e_leak);
        layer.vm[static_cast<std::size_t>(i)] += (i_in + i_ex) / cfg.cm * cfg.dt_ms;
    }

    ApplyIFAndRefractory(layer, out_spike, step, cfg.v_th, cfg.v_reset, cfg.v_spike, static_cast<int>(cfg.refractory_ms / cfg.dt_ms));

    auto put = [&](const std::vector<double>& src, int start) { std::copy(src.begin(), src.end(), s_state.begin() + start * n); };
    put(s_ampa_cm1, 0); put(s_nmda_cm1, 1); put(s_ampa_cm2, 2); put(s_nmda_cm2, 3); put(s_ampa_cm3, 4); put(s_nmda_cm3, 5);
    put(s_ampa_sm, 6); put(s_nmda_sm, 7); put(s_ampa_mm1, 8); put(s_nmda_mm1, 9); put(s_ampa_mm2, 10); put(s_nmda_mm2, 11);
}

void UpdateMM(
    const Stage3Config& cfg,
    const MatrixF32& c_emm,
    const SpikeVector& spk_ext,
    const SpikeVector& spk_e,
    int step,
    LayerState& layer,
    std::vector<double>& s_state,
    SpikeVector& out_spike) {
    const int n = cfg.n_mm;
    std::vector<double> input_ext(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_e(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        input_ext[static_cast<std::size_t>(i)] = static_cast<double>(spk_ext[static_cast<std::size_t>(i)]);
    }
    for (int row = 0; row < n; ++row) {
        double total = 0.0;
        for (int col = 0; col < cfg.n_e; ++col) {
            total += MatrixAt(c_emm, row, col) * spk_e[static_cast<std::size_t>(col)];
        }
        input_e[static_cast<std::size_t>(row)] = total;
    }

    auto seg = [&](int start) { return std::vector<double>(s_state.begin() + start * n, s_state.begin() + (start + 1) * n); };
    std::vector<double> s_ampa_ext = seg(0), s_ampa_e = seg(1), s_nmda_e = seg(2);
    UpdateAMPA(s_ampa_ext, input_ext, cfg.dt_ms, cfg.tau_ampa);
    UpdateAMPA(s_ampa_e, input_e, cfg.dt_ms, cfg.tau_ampa);
    UpdateNMDA(s_nmda_e, input_e, cfg.dt_ms, cfg.tau_nmda);

    for (int i = 0; i < n; ++i) {
        const double vm = layer.vm[static_cast<std::size_t>(i)];
        const double i_ex_ext = -cfg.g_ext_mm * s_ampa_ext[static_cast<std::size_t>(i)] * (vm - cfg.e_ampa);
        const double i_ex_e = -cfg.g_e_mm * s_ampa_e[static_cast<std::size_t>(i)] * (vm - cfg.e_ampa)
                            - cfg.g_e_mm * s_nmda_e[static_cast<std::size_t>(i)] * (vm - cfg.e_nmda) / (1.0 + std::exp(-0.062 * vm / 3.57));
        const double i_in = -cfg.g_l * (vm - cfg.e_leak);
        layer.vm[static_cast<std::size_t>(i)] += (i_in + i_ex_ext + i_ex_e) / cfg.cm * cfg.dt_ms;
    }

    ApplyIFAndRefractory(layer, out_spike, step, cfg.v_th, cfg.v_reset, cfg.v_spike, static_cast<int>(cfg.refractory_ms / cfg.dt_ms));
    std::copy(s_ampa_ext.begin(), s_ampa_ext.end(), s_state.begin());
    std::copy(s_ampa_e.begin(), s_ampa_e.end(), s_state.begin() + n);
    std::copy(s_nmda_e.begin(), s_nmda_e.end(), s_state.begin() + 2 * n);
}

void UpdateE(
    const Stage3Config& cfg,
    const MatrixF32& w_ee,
    const MatrixF32& c_sig_e,
    const MatrixF32& c_cm1e,
    const MatrixF32& c_cm2e,
    const MatrixF32& c_cm3e,
    const SpikeVector& spk_ext,
    const SpikeVector& spk_e,
    const SpikeVector& spk_i,
    const SpikeVector& spk_sig,
    const SpikeVector& spk_cm1,
    const SpikeVector& spk_cm2,
    const SpikeVector& spk_cm3,
    int step,
    LayerState& layer,
    std::vector<double>& s_state,
    SpikeVector& out_spike) {
    const int n = cfg.n_e;
    std::vector<double> input_ext(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_e(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_i(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_sig(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_cm1(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_cm2(static_cast<std::size_t>(n), 0.0);
    std::vector<double> input_cm3(static_cast<std::size_t>(n), 0.0);

    int sum_i = 0;
    for (int i = 0; i < cfg.n_i; ++i) sum_i += spk_i[static_cast<std::size_t>(i)];
    for (int row = 0; row < n; ++row) {
        input_ext[static_cast<std::size_t>(row)] = static_cast<double>(spk_ext[static_cast<std::size_t>(row)]);
        input_i[static_cast<std::size_t>(row)] = static_cast<double>(sum_i);
        double total_ee = 0.0, total_sig = 0.0, total_cm1 = 0.0, total_cm2 = 0.0, total_cm3 = 0.0;
        for (int col = 0; col < n; ++col) total_ee += MatrixAt(w_ee, row, col) * spk_e[static_cast<std::size_t>(col)];
        for (int col = 0; col < cfg.n_sig; ++col) total_sig += MatrixAt(c_sig_e, row, col) * spk_sig[static_cast<std::size_t>(col)];
        for (int col = 0; col < cfg.n_cm; ++col) {
            total_cm1 += MatrixAt(c_cm1e, row, col) * spk_cm1[static_cast<std::size_t>(col)];
            total_cm2 += MatrixAt(c_cm2e, row, col) * spk_cm2[static_cast<std::size_t>(col)];
            total_cm3 += MatrixAt(c_cm3e, row, col) * spk_cm3[static_cast<std::size_t>(col)];
        }
        input_e[static_cast<std::size_t>(row)] = total_ee;
        input_sig[static_cast<std::size_t>(row)] = total_sig;
        input_cm1[static_cast<std::size_t>(row)] = total_cm1;
        input_cm2[static_cast<std::size_t>(row)] = total_cm2;
        input_cm3[static_cast<std::size_t>(row)] = total_cm3;
    }

    auto seg = [&](int start) { return std::vector<double>(s_state.begin() + start * n, s_state.begin() + (start + 1) * n); };
    std::vector<double> s_ampa_ext = seg(0), s_ampa_e = seg(1), s_nmda_e = seg(2), s_gaba_i = seg(3), s_ampa_sig = seg(4),
                        s_ampa_cm1 = seg(5), s_nmda_cm1 = seg(6), s_ampa_cm2 = seg(7), s_nmda_cm2 = seg(8), s_ampa_cm3 = seg(9), s_nmda_cm3 = seg(10);
    UpdateAMPA(s_ampa_ext, input_ext, cfg.dt_ms, cfg.tau_ampa);
    UpdateAMPA(s_ampa_e, input_e, cfg.dt_ms, cfg.tau_ampa);
    UpdateNMDA(s_nmda_e, input_e, cfg.dt_ms, cfg.tau_nmda);
    UpdateGABA(s_gaba_i, input_i, cfg.dt_ms, cfg.tau_gaba);
    UpdateAMPA(s_ampa_sig, input_sig, cfg.dt_ms, cfg.tau_ampa);
    UpdateAMPA(s_ampa_cm1, input_cm1, cfg.dt_ms, cfg.tau_ampa);
    UpdateNMDA(s_nmda_cm1, input_cm1, cfg.dt_ms, cfg.tau_nmda);
    UpdateAMPA(s_ampa_cm2, input_cm2, cfg.dt_ms, cfg.tau_ampa);
    UpdateNMDA(s_nmda_cm2, input_cm2, cfg.dt_ms, cfg.tau_nmda);
    UpdateAMPA(s_ampa_cm3, input_cm3, cfg.dt_ms, cfg.tau_ampa);
    UpdateNMDA(s_nmda_cm3, input_cm3, cfg.dt_ms, cfg.tau_nmda);

    for (int i = 0; i < n; ++i) {
        const double vm = layer.vm[static_cast<std::size_t>(i)];
        const auto nmda = [&](double g, double sa, double sn) {
            return -g * sa * (vm - cfg.e_ampa) - g * sn * (vm - cfg.e_nmda) / (1.0 + std::exp(-0.062 * vm / 3.57));
        };
        const double i_ex =
            -cfg.g_ext_e * s_ampa_ext[static_cast<std::size_t>(i)] * (vm - cfg.e_ampa) +
            nmda(cfg.g_ee, s_ampa_e[static_cast<std::size_t>(i)], s_nmda_e[static_cast<std::size_t>(i)]) +
            -cfg.g_ie * s_gaba_i[static_cast<std::size_t>(i)] * (vm - cfg.e_gaba) +
            -cfg.g_sig_e * s_ampa_sig[static_cast<std::size_t>(i)] * (vm - cfg.e_ampa) +
            nmda(cfg.g_cm_e, s_ampa_cm1[static_cast<std::size_t>(i)], s_nmda_cm1[static_cast<std::size_t>(i)]) +
            nmda(cfg.g_cm_e, s_ampa_cm2[static_cast<std::size_t>(i)], s_nmda_cm2[static_cast<std::size_t>(i)]) +
            nmda(cfg.g_cm_e, s_ampa_cm3[static_cast<std::size_t>(i)], s_nmda_cm3[static_cast<std::size_t>(i)]);
        const double i_in = -cfg.g_l * (vm - cfg.e_leak);
        layer.vm[static_cast<std::size_t>(i)] += (i_in + i_ex) / cfg.cm * cfg.dt_ms;
    }

    ApplyIFAndRefractory(layer, out_spike, step, cfg.v_th, cfg.v_reset, cfg.v_spike, static_cast<int>(cfg.refractory_ms / cfg.dt_ms));
    auto put = [&](const std::vector<double>& src, int start) { std::copy(src.begin(), src.end(), s_state.begin() + start * n); };
    put(s_ampa_ext, 0); put(s_ampa_e, 1); put(s_nmda_e, 2); put(s_gaba_i, 3); put(s_ampa_sig, 4);
    put(s_ampa_cm1, 5); put(s_nmda_cm1, 6); put(s_ampa_cm2, 7); put(s_nmda_cm2, 8); put(s_ampa_cm3, 9); put(s_nmda_cm3, 10);
}

void UpdateI(
    const Stage3Config& cfg,
    const SpikeVector& spk_ext_i,
    const SpikeVector& spk_e,
    const SpikeVector& spk_i,
    int step,
    LayerState& layer,
    std::vector<double>& s_state,
    SpikeVector& out_spike) {
    const int n = cfg.n_i;
    int sum_e = 0, sum_i = 0;
    for (int i = 0; i < cfg.n_e; ++i) sum_e += spk_e[static_cast<std::size_t>(i)];
    for (int i = 0; i < n; ++i) sum_i += spk_i[static_cast<std::size_t>(i)];
    std::vector<double> input_ext(static_cast<std::size_t>(n), 0.0), input_e(static_cast<std::size_t>(n), static_cast<double>(sum_e)), input_i(static_cast<std::size_t>(n), static_cast<double>(sum_i));
    for (int i = 0; i < n; ++i) input_ext[static_cast<std::size_t>(i)] = static_cast<double>(spk_ext_i[static_cast<std::size_t>(i)]);

    auto seg = [&](int start) { return std::vector<double>(s_state.begin() + start * n, s_state.begin() + (start + 1) * n); };
    std::vector<double> s_ampa_ext = seg(0), s_ampa_e = seg(1), s_nmda_e = seg(2), s_gaba_i = seg(3);
    UpdateAMPA(s_ampa_ext, input_ext, cfg.dt_ms, cfg.tau_ampa);
    UpdateAMPA(s_ampa_e, input_e, cfg.dt_ms, cfg.tau_ampa);
    UpdateNMDA(s_nmda_e, input_e, cfg.dt_ms, cfg.tau_nmda);
    UpdateGABA(s_gaba_i, input_i, cfg.dt_ms, cfg.tau_gaba);

    for (int i = 0; i < n; ++i) {
        const double vm = layer.vm[static_cast<std::size_t>(i)];
        const double i_ex =
            -cfg.g_ext_i * s_ampa_ext[static_cast<std::size_t>(i)] * (vm - cfg.e_ampa) +
            (-cfg.g_ei * s_ampa_e[static_cast<std::size_t>(i)] * (vm - cfg.e_ampa) - cfg.g_ei * s_nmda_e[static_cast<std::size_t>(i)] * (vm - cfg.e_nmda) / (1.0 + std::exp(-0.062 * vm / 3.57))) +
            -cfg.g_ii * s_gaba_i[static_cast<std::size_t>(i)] * (vm - cfg.e_gaba);
        const double i_in = -cfg.g_l * (vm - cfg.e_leak);
        layer.vm[static_cast<std::size_t>(i)] += (i_in + i_ex) / cfg.cm * cfg.dt_ms;
    }

    ApplyIFAndRefractory(layer, out_spike, step, cfg.v_th, cfg.v_reset, cfg.v_spike, static_cast<int>(cfg.refractory_ms / cfg.dt_ms));
    std::copy(s_ampa_ext.begin(), s_ampa_ext.end(), s_state.begin());
    std::copy(s_ampa_e.begin(), s_ampa_e.end(), s_state.begin() + n);
    std::copy(s_nmda_e.begin(), s_nmda_e.end(), s_state.begin() + 2 * n);
    std::copy(s_gaba_i.begin(), s_gaba_i.end(), s_state.begin() + 3 * n);
}

void HebbUpdateStrictMatlab(
    MatrixF32& w,
    const MatrixF32& mask,
    const std::vector<int>& mm_window_counts,
    const std::vector<int>& bg_window_counts,
    double increment,
    std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < w.rows; ++i) {
        for (int j = 0; j < w.cols; ++j) {
            if (i < static_cast<int>(mm_window_counts.size()) && i < static_cast<int>(bg_window_counts.size())
                && mm_window_counts[static_cast<std::size_t>(i)] > 0
                && bg_window_counts[static_cast<std::size_t>(i)] > 0
                && MatrixAt(mask, i, j) == 1.0f) {
                MatrixAtRef(w, i, j) = static_cast<float>(MatrixAt(w, i, j) + increment * dist(rng));
            }
        }
    }
    for (float& v : w.values) {
        if (v > 1.0f) v = 1.0f;
        if (v < 0.0f) v = 0.0f;
    }
}

std::vector<int> WindowSum(const std::vector<SpikeVector>& spikes, int from_step, int to_step, int neuron_count) {
    std::vector<int> counts(static_cast<std::size_t>(neuron_count), 0);
    for (int step = from_step; step < to_step; ++step) {
        const SpikeVector& s = spikes[static_cast<std::size_t>(step)];
        for (int i = 0; i < neuron_count; ++i) counts[static_cast<std::size_t>(i)] += s[static_cast<std::size_t>(i)];
    }
    return counts;
}

std::vector<double> FirstJointPair(const std::vector<double>& flat) {
    if (flat.size() < 2 || (flat.size() % 2) != 0) {
        throw std::runtime_error("Joint angle text export must contain an even number of values");
    }
    const std::size_t stride = flat.size() / 2;
    return {flat[0], flat[stride]};
}

MatrixF32 SliceRows(const MatrixF32& matrix, int row_begin, int row_count) {
    MatrixF32 out;
    out.rows = row_count;
    out.cols = matrix.cols;
    out.values.assign(static_cast<std::size_t>(out.rows) * static_cast<std::size_t>(out.cols), 0.0f);
    for (int col = 0; col < out.cols; ++col) {
        for (int row = 0; row < out.rows; ++row) {
            MatrixAtRef(out, row, col) = MatrixAt(matrix, row_begin + row, col);
        }
    }
    return out;
}

MatrixF32 SliceCols(const MatrixF32& matrix, int col_begin, int col_count) {
    MatrixF32 out;
    out.rows = matrix.rows;
    out.cols = col_count;
    out.values.assign(static_cast<std::size_t>(out.rows) * static_cast<std::size_t>(out.cols), 0.0f);
    for (int col = 0; col < out.cols; ++col) {
        for (int row = 0; row < out.rows; ++row) {
            MatrixAtRef(out, row, col) = MatrixAt(matrix, row, col_begin + col);
        }
    }
    return out;
}

SimulatedTrajectory ReplayForwardDynamicsWithReset(
    const TorqueTrajectory& torques,
    const ArmParameters& params,
    double dt_seconds,
    double initial_theta1,
    double initial_theta2,
    int reset_after_step,
    double reset_theta1,
    double reset_theta2) {
    if (torques.q1.size() != torques.q2.size() || torques.q1.empty()) {
        throw std::runtime_error("Invalid torque vector size in ReplayForwardDynamicsWithReset");
    }

    const std::size_t step_count = torques.q1.size();
    SimulatedTrajectory sim;
    sim.theta1.resize(step_count + 1, initial_theta1);
    sim.theta2.resize(step_count + 1, initial_theta2);
    sim.dtheta1.resize(step_count + 1, 0.0);
    sim.dtheta2.resize(step_count + 1, 0.0);
    sim.ddtheta1.resize(step_count, 0.0);
    sim.ddtheta2.resize(step_count, 0.0);

    for (std::size_t i = 0; i < step_count; ++i) {
        const AccelerationSample sample = JointAcceleration(
            sim.theta1[i], sim.theta2[i], sim.dtheta1[i], sim.dtheta2[i], torques.q1[i], torques.q2[i], params);
        sim.ddtheta1[i] = sample.ddtheta1;
        sim.ddtheta2[i] = sample.ddtheta2;
        sim.dtheta1[i + 1] = sim.dtheta1[i] + sample.ddtheta1 * dt_seconds;
        sim.dtheta2[i + 1] = sim.dtheta2[i] + sample.ddtheta2 * dt_seconds;
        sim.theta1[i + 1] = sim.theta1[i] + sim.dtheta1[i] * dt_seconds;
        sim.theta2[i + 1] = sim.theta2[i] + sim.dtheta2[i] * dt_seconds;
        if (static_cast<int>(i + 1) == reset_after_step) {
            sim.theta1[i + 1] = reset_theta1;
            sim.theta2[i + 1] = reset_theta2;
        }
    }

    sim.hand_path = HandPosition(sim.theta1, sim.theta2, params);
    return sim;
}

void WriteMatrixTsv(const std::filesystem::path& path, const MatrixF32& matrix) {
    std::ofstream out(path);
    for (int row = 0; row < matrix.rows; ++row) {
        for (int col = 0; col < matrix.cols; ++col) {
            if (col != 0) {
                out << '\t';
            }
            out << MatrixAt(matrix, row, col);
        }
        out << '\n';
    }
}

}  // namespace

Stage3Mode ParseStage3Mode(const std::string& text) {
    if (text == "test") return Stage3Mode::Test;
    if (text == "test_d") return Stage3Mode::TestD;
    if (text == "train_g") return Stage3Mode::TrainG;
    if (text == "train_d") return Stage3Mode::TrainD;
    throw std::runtime_error("Unsupported stage-3 mode: " + text);
}

Stage3Data LoadStage3Data(const std::filesystem::path& input_directory, bool require_test_weights) {
    Stage3Data data;
    data.sspk_ext = LoadBinarySpikeSeries(RequirePath(input_directory / "sspk_ext.bin"), RequirePath(input_directory / "sspk_ext.shape.txt"));
    data.sspk_ext_e = LoadBinarySpikeSeries(RequirePath(input_directory / "sspk_extE.bin"), RequirePath(input_directory / "sspk_extE.shape.txt"));
    data.sspk_ext_i = LoadBinarySpikeSeries(RequirePath(input_directory / "sspk_extI.bin"), RequirePath(input_directory / "sspk_extI.shape.txt"));
    data.sspk_ext_mm = LoadBinarySpikeSeries(RequirePath(input_directory / "sspk_extMM.bin"), RequirePath(input_directory / "sspk_extMM.shape.txt"));
    data.sspk_sig = LoadBinarySpikeSeries(RequirePath(input_directory / "sspk_sig.bin"), RequirePath(input_directory / "sspk_sig.shape.txt"));
    data.sspk_sm = LoadBinarySpikeSeries(RequirePath(input_directory / "sspk_SM.bin"), RequirePath(input_directory / "sspk_SM.shape.txt"));
    data.ww_extcm1 = LoadBinaryWeightWindows(RequirePath(input_directory / "ww_extCM1.bin"), RequirePath(input_directory / "ww_extCM1.shape.txt"));
    data.ww_extcm2 = LoadBinaryWeightWindows(RequirePath(input_directory / "ww_extCM2.bin"), RequirePath(input_directory / "ww_extCM2.shape.txt"));
    data.ww_extcm3 = LoadBinaryWeightWindows(RequirePath(input_directory / "ww_extCM3.bin"), RequirePath(input_directory / "ww_extCM3.shape.txt"));

    if (require_test_weights) {
        data.c_emm1 = LoadMatrixF32(RequirePath(input_directory / "C_EMM1.bin"), RequirePath(input_directory / "C_EMM1.shape.txt"));
        data.c_emm2 = LoadMatrixF32(RequirePath(input_directory / "C_EMM2.bin"), RequirePath(input_directory / "C_EMM2.shape.txt"));
        data.c_mmbg1 = LoadMatrixF32(RequirePath(input_directory / "C_MMbg1.bin"), RequirePath(input_directory / "C_MMbg1.shape.txt"));
        data.c_mmbg2 = LoadMatrixF32(RequirePath(input_directory / "C_MMbg2.bin"), RequirePath(input_directory / "C_MMbg2.shape.txt"));
        data.w_mmbg1 = LoadMatrixF32(RequirePath(input_directory / "w_MMbg1_12.bin"), RequirePath(input_directory / "w_MMbg1_12.shape.txt"));
        data.w_mmbg2 = LoadMatrixF32(RequirePath(input_directory / "w_MMbg2_12.bin"), RequirePath(input_directory / "w_MMbg2_12.shape.txt"));

        if (Exists(input_directory / "C_CM1E.bin")) {
            data.c_cm1e = LoadMatrixF32(RequirePath(input_directory / "C_CM1E.bin"), RequirePath(input_directory / "C_CM1E.shape.txt"));
        }
        if (Exists(input_directory / "C_CM2E.bin")) {
            data.c_cm2e = LoadMatrixF32(RequirePath(input_directory / "C_CM2E.bin"), RequirePath(input_directory / "C_CM2E.shape.txt"));
        }
        if (Exists(input_directory / "C_CM3E.bin")) {
            data.c_cm3e = LoadMatrixF32(RequirePath(input_directory / "C_CM3E.bin"), RequirePath(input_directory / "C_CM3E.shape.txt"));
        }
        if (Exists(input_directory / "C_sigE.bin")) {
            data.c_sig_e = LoadMatrixF32(RequirePath(input_directory / "C_sigE.bin"), RequirePath(input_directory / "C_sigE.shape.txt"));
        }
        if (Exists(input_directory / "w_EE.bin")) {
            data.w_ee = LoadMatrixF32(RequirePath(input_directory / "w_EE.bin"), RequirePath(input_directory / "w_EE.shape.txt"));
        }
        if (Exists(input_directory / "C_EMM.bin")) {
            data.c_emm_d = LoadMatrixF32(RequirePath(input_directory / "C_EMM.bin"), RequirePath(input_directory / "C_EMM.shape.txt"));
        }
        if (Exists(input_directory / "C_MMbg.bin")) {
            data.c_mmbg_d = LoadMatrixF32(RequirePath(input_directory / "C_MMbg.bin"), RequirePath(input_directory / "C_MMbg.shape.txt"));
        }
        if (Exists(input_directory / "w_MMbg_13.bin")) {
            data.w_mmbg_d = LoadMatrixF32(RequirePath(input_directory / "w_MMbg_13.bin"), RequirePath(input_directory / "w_MMbg_13.shape.txt"));
        }
    }

    data.a1_flat = LoadMatrixTextAll(RequirePath(input_directory / "A1.tsv"));
    data.a2_flat = LoadMatrixTextAll(RequirePath(input_directory / "A2.tsv"));
    data.a3_flat = LoadMatrixTextAll(RequirePath(input_directory / "A3.tsv"));
    data.x1y1 = LoadStrokePathFromDirectory(input_directory, "1");
    data.x2y2 = LoadStrokePathFromDirectory(input_directory, "2");
    data.x3y3 = LoadStrokePathFromDirectory(input_directory, "3");
    return data;
}

Stage3Result RunStage3(const Stage3Config& cfg, const Stage3Data& data, Stage3Mode mode) {
    const bool is_test = (mode == Stage3Mode::Test || mode == Stage3Mode::TestD);
    Stage3Config cfg_local = cfg;
    if (mode == Stage3Mode::TestD) {
        cfg_local.g_sm_bg = 10.0;
        cfg_local.g_mm_bg = 10.0;
    }
    const int step_count = static_cast<int>(cfg.total_time_ms / cfg.dt_ms);
    const int window_steps = static_cast<int>(10.0 / cfg.dt_ms);
    const int window_count = step_count / window_steps;
    const std::vector<int> window_sequence = BuildWindowIndexSequence(window_count);
    std::mt19937 rng(0);
    const char* diag_dir = std::getenv("HANDWRITING_STAGE3_DIAG_DIR");
    const bool write_train_diag = (!is_test && diag_dir != NULL && diag_dir[0] != '\0');

    MatrixF32 c_bgcm1 = ConnectivityBgCM(cfg.n_cm, cfg.n_bg, 1);
    MatrixF32 c_bgcm2 = ConnectivityBgCM(cfg.n_cm, cfg.n_bg, 2);
    MatrixF32 c_bgcm3 = ConnectivityBgCM(cfg.n_cm, cfg.n_bg, 3);
    MatrixF32 c_cm1bg = ConnectivityCMbg(cfg.n_cm, cfg.n_bg, 1);
    MatrixF32 c_cm2bg = ConnectivityCMbg(cfg.n_cm, cfg.n_bg, 2);
    MatrixF32 c_cm3bg = ConnectivityCMbg(cfg.n_cm, cfg.n_bg, 3);
    MatrixF32 c_smbg = ConnectivitySMbg(cfg.n_sm_group, cfg.n_bg);
    MatrixF32 w_ee = (!data.w_ee.values.empty()) ? data.w_ee : StructuredConnectivityEE(cfg.n_e);
    MatrixF32 c_cm1e = (!data.c_cm1e.values.empty()) ? data.c_cm1e : ConnectivityCME(cfg.n_cm, cfg.n_e, cfg.n_rcv, 200);
    MatrixF32 c_cm2e = (!data.c_cm2e.values.empty()) ? data.c_cm2e : ConnectivityCME(cfg.n_cm, cfg.n_e, cfg.n_rcv, 400);
    MatrixF32 c_cm3e = (!data.c_cm3e.values.empty()) ? data.c_cm3e : ConnectivityCME(cfg.n_cm, cfg.n_e, cfg.n_rcv, 600);
    MatrixF32 c_sig_e = (!data.c_sig_e.values.empty()) ? data.c_sig_e : ConnectivitySignalE(cfg.n_sig, cfg.n_e, cfg.n_rcv);

    MatrixF32 c_emm1 = is_test ? data.c_emm1 : RandomBinaryConnectivity(cfg.n_mm, cfg.n_e, 0.1, rng);
    MatrixF32 c_emm2 = is_test ? data.c_emm2 : RandomBinaryConnectivity(cfg.n_mm, cfg.n_e, 0.1, rng);
    MatrixF32 c_mmbg1 = is_test ? data.c_mmbg1 : RandomBinaryConnectivity(cfg.n_bg_total / 2, cfg.n_mm, 0.1, rng);
    MatrixF32 c_mmbg2 = is_test ? data.c_mmbg2 : RandomBinaryConnectivity(cfg.n_bg_total / 2, cfg.n_mm, 0.1, rng);
    MatrixF32 w_mmbg1;
    MatrixF32 w_mmbg2;
    if (is_test) {
        w_mmbg1 = data.w_mmbg1;
        w_mmbg2 = data.w_mmbg2;
        if (mode == Stage3Mode::TestD && !data.c_emm_d.values.empty() && !data.c_mmbg_d.values.empty() && !data.w_mmbg_d.values.empty()) {
            c_emm1 = SliceRows(data.c_emm_d, 0, cfg.n_mm);
            c_emm2 = SliceRows(data.c_emm_d, cfg.n_mm, cfg.n_mm);
            c_mmbg1 = SliceCols(data.c_mmbg_d, 0, cfg.n_mm);
            c_mmbg2 = SliceCols(data.c_mmbg_d, cfg.n_mm, cfg.n_mm);
            w_mmbg1 = SliceCols(data.w_mmbg_d, 0, cfg.n_mm);
            w_mmbg2 = SliceCols(data.w_mmbg_d, cfg.n_mm, cfg.n_mm);
            for (float& v : w_mmbg1.values) if (v < 1.0f) v = 0.0f;
            for (float& v : w_mmbg2.values) if (v < 1.0f) v = 0.0f;
        }
    } else {
        w_mmbg1.rows = cfg.n_bg_total;
        w_mmbg1.cols = cfg.n_mm;
        w_mmbg1.values.assign(static_cast<std::size_t>(cfg.n_bg_total) * static_cast<std::size_t>(cfg.n_mm), 0.0f);
        w_mmbg2 = w_mmbg1;

        MatrixF32 mask1;
        mask1.rows = cfg.n_bg_total;
        mask1.cols = cfg.n_mm;
        mask1.values.assign(static_cast<std::size_t>(cfg.n_bg_total) * static_cast<std::size_t>(cfg.n_mm), 0.0f);
        MatrixF32 mask2 = mask1;
        for (int r = 0; r < cfg.n_bg_total / 2; ++r) {
            for (int c = 0; c < cfg.n_mm; ++c) {
                MatrixAtRef(mask1, r, c) = MatrixAt(c_mmbg1, r, c);
                MatrixAtRef(mask2, cfg.n_bg_total / 2 + r, c) = MatrixAt(c_mmbg2, r, c);
            }
        }
        c_mmbg1 = mask1;
        c_mmbg2 = mask2;
    }

    NetworkState state;
    auto init_layer = [&](int count) {
        LayerState layer;
        layer.vm = Filled(count, cfg.v_reset);
        layer.last_spike_step = Filled(count, -1000000);
        return layer;
    };
    state.cm1 = init_layer(cfg.n_cm); state.cm2 = init_layer(cfg.n_cm); state.cm3 = init_layer(cfg.n_cm);
    state.bg = init_layer(cfg.n_bg_total); state.mm1 = init_layer(cfg.n_mm); state.mm2 = init_layer(cfg.n_mm);
    state.e = init_layer(cfg.n_e); state.i = init_layer(cfg.n_i);
    state.s_cm1 = Filled(2 * cfg.n_cm, 0.0); state.s_cm2 = Filled(2 * cfg.n_cm, 0.0); state.s_cm3 = Filled(2 * cfg.n_cm, 0.0);
    state.s_bg = Filled(12 * cfg.n_bg_total, 0.0); state.s_mm1 = Filled(3 * cfg.n_mm, 0.0); state.s_mm2 = Filled(3 * cfg.n_mm, 0.0);
    state.s_e = Filled(11 * cfg.n_e, 0.0); state.s_i = Filled(4 * cfg.n_i, 0.0);

    std::vector<SpikeVector> hist_bg, hist_mm1, hist_mm2, hist_cm1, hist_cm2, hist_cm3;
    hist_bg.reserve(step_count); hist_mm1.reserve(step_count); hist_mm2.reserve(step_count); hist_cm1.reserve(step_count); hist_cm2.reserve(step_count); hist_cm3.reserve(step_count);
    std::vector<int> pop_counts(static_cast<std::size_t>(window_count) * static_cast<std::size_t>(cfg.n_pop), 0);
    std::vector<int> train_updates1(static_cast<std::size_t>(cfg.n_bg_total) * static_cast<std::size_t>(cfg.n_mm), 0);
    std::vector<int> train_updates2(train_updates1.size(), 0);

    SpikeVector spk_bg(static_cast<std::size_t>(cfg.n_bg_total), 0);
    SpikeVector spk_mm1(static_cast<std::size_t>(cfg.n_mm), 0), spk_mm2(static_cast<std::size_t>(cfg.n_mm), 0);
    SpikeVector spk_e(static_cast<std::size_t>(cfg.n_e), 0), spk_i(static_cast<std::size_t>(cfg.n_i), 0);

    for (int step = 0; step < step_count; ++step) {
        const int window = std::min(window_count - 1, step / window_steps);
        const int weight_window = window_sequence[static_cast<std::size_t>(window)];

        SpikeVector ext_cm1 = BuildCmExternalSpike(data.sspk_ext, mode, is_test, 0, step, cfg.n_ext);
        SpikeVector ext_cm2 = BuildCmExternalSpike(data.sspk_ext, mode, is_test, 1, step, cfg.n_ext);
        SpikeVector ext_cm3 = BuildCmExternalSpike(data.sspk_ext, mode, is_test, 2, step, cfg.n_ext);
        SpikeVector spk_cm1(static_cast<std::size_t>(cfg.n_cm), 0), spk_cm2(static_cast<std::size_t>(cfg.n_cm), 0), spk_cm3(static_cast<std::size_t>(cfg.n_cm), 0);
        SpikeVector spk_sm(static_cast<std::size_t>(cfg.n_sm_total), 0), spk_sig(static_cast<std::size_t>(cfg.n_sig), 0);
        SpikeVector ext_mm1(static_cast<std::size_t>(cfg.n_mm), 0), ext_mm2(static_cast<std::size_t>(cfg.n_mm), 0);
        SpikeVector ext_e(static_cast<std::size_t>(cfg.n_e), 0), ext_i(static_cast<std::size_t>(cfg.n_i), 0);

        for (int i = 0; i < cfg.n_sm_total; ++i) spk_sm[static_cast<std::size_t>(i)] = SpikeAt(data.sspk_sm, i, step);
        for (int i = 0; i < cfg.n_sig; ++i) spk_sig[static_cast<std::size_t>(i)] = SpikeAt(data.sspk_sig, i, step);
        for (int i = 0; i < cfg.n_mm; ++i) {
            ext_mm1[static_cast<std::size_t>(i)] = SpikeAt(data.sspk_ext_mm, i, step);
            ext_mm2[static_cast<std::size_t>(i)] = SpikeAt(data.sspk_ext_mm, cfg.n_mm + i, step);
        }
        for (int i = 0; i < cfg.n_e; ++i) ext_e[static_cast<std::size_t>(i)] = SpikeAt(data.sspk_ext_e, i, step);
        for (int i = 0; i < cfg.n_i; ++i) ext_i[static_cast<std::size_t>(i)] = SpikeAt(data.sspk_ext_i, i, step);

        UpdateCM(cfg_local, data.ww_extcm1, weight_window, c_bgcm1, ext_cm1, spk_bg, step, state.cm1, state.s_cm1, spk_cm1);
        UpdateCM(cfg_local, data.ww_extcm2, weight_window, c_bgcm2, ext_cm2, spk_bg, step, state.cm2, state.s_cm2, spk_cm2);
        UpdateCM(cfg_local, data.ww_extcm3, weight_window, c_bgcm3, ext_cm3, spk_bg, step, state.cm3, state.s_cm3, spk_cm3);
        UpdateBG(cfg_local, c_cm1bg, c_cm2bg, c_cm3bg, c_smbg, c_mmbg1, w_mmbg1, c_mmbg2, w_mmbg2, spk_cm1, spk_cm2, spk_cm3, spk_sm, spk_mm1, spk_mm2, step, state.bg, state.s_bg, spk_bg);
        UpdateMM(cfg_local, c_emm1, ext_mm1, spk_e, step, state.mm1, state.s_mm1, spk_mm1);
        UpdateMM(cfg_local, c_emm2, ext_mm2, spk_e, step, state.mm2, state.s_mm2, spk_mm2);
        UpdateE(cfg_local, w_ee, c_sig_e, c_cm1e, c_cm2e, c_cm3e, ext_e, spk_e, spk_i, spk_sig, spk_cm1, spk_cm2, spk_cm3, step, state.e, state.s_e, spk_e);
        UpdateI(cfg_local, ext_i, spk_e, spk_i, step, state.i, state.s_i, spk_i);

        hist_bg.push_back(spk_bg);
        hist_mm1.push_back(spk_mm1);
        hist_mm2.push_back(spk_mm2);
        hist_cm1.push_back(spk_cm1);
        hist_cm2.push_back(spk_cm2);
        hist_cm3.push_back(spk_cm3);

        if (!is_test && ((step + 1) % window_steps == 0)) {
            const int begin = step + 1 - window_steps;
            const std::vector<int> mm1_counts = WindowSum(hist_mm1, begin, step + 1, cfg.n_mm);
            const std::vector<int> mm2_counts = WindowSum(hist_mm2, begin, step + 1, cfg.n_mm);
            const std::vector<int> bg_counts = WindowSum(hist_bg, begin, step + 1, cfg.n_bg_total);
            if (write_train_diag) {
                for (int row = 0; row < cfg.n_bg_total; ++row) {
                    for (int col = 0; col < cfg.n_mm; ++col) {
                        if (row < static_cast<int>(mm1_counts.size()) &&
                            row < static_cast<int>(bg_counts.size()) &&
                            mm1_counts[static_cast<std::size_t>(row)] > 0 &&
                            bg_counts[static_cast<std::size_t>(row)] > 0 &&
                            MatrixAt(c_mmbg1, row, col) == 1.0f) {
                            train_updates1[static_cast<std::size_t>(row) * static_cast<std::size_t>(cfg.n_mm) + static_cast<std::size_t>(col)] += 1;
                        }
                        if (row < static_cast<int>(mm2_counts.size()) &&
                            row < static_cast<int>(bg_counts.size()) &&
                            mm2_counts[static_cast<std::size_t>(row)] > 0 &&
                            bg_counts[static_cast<std::size_t>(row)] > 0 &&
                            MatrixAt(c_mmbg2, row, col) == 1.0f) {
                            train_updates2[static_cast<std::size_t>(row) * static_cast<std::size_t>(cfg.n_mm) + static_cast<std::size_t>(col)] += 1;
                        }
                    }
                }
            }
            HebbUpdateStrictMatlab(w_mmbg1, c_mmbg1, mm1_counts, bg_counts, cfg_local.hebb_increment, rng);
            HebbUpdateStrictMatlab(w_mmbg2, c_mmbg2, mm2_counts, bg_counts, cfg_local.hebb_increment, rng);
        }
    }

    if (write_train_diag) {
        std::filesystem::create_directories(diag_dir);
        auto write_counts = [&](const char* filename, const std::vector<int>& values) {
            std::ofstream out(std::filesystem::path(diag_dir) / filename);
            for (int row = 0; row < cfg.n_bg_total; ++row) {
                for (int col = 0; col < cfg.n_mm; ++col) {
                    if (col != 0) {
                        out << '\t';
                    }
                    out << values[static_cast<std::size_t>(row) * static_cast<std::size_t>(cfg.n_mm) + static_cast<std::size_t>(col)];
                }
                out << '\n';
            }
        };
        write_counts("w_mmbg1_update_counts.tsv", train_updates1);
        write_counts("w_mmbg2_update_counts.tsv", train_updates2);
    }

    const int neurons_per_pop = cfg.n_cm / cfg.n_pop;
    for (int window = 0; window < window_count; ++window) {
        const int begin = window * window_steps;
        const int end = begin + window_steps;
        for (int pop = 0; pop < cfg.n_pop; ++pop) {
            const int n0 = pop * neurons_per_pop;
            const int n1 = n0 + neurons_per_pop;
            int total = 0;
            for (int step = begin; step < end; ++step) {
                for (int neuron = n0; neuron < n1; ++neuron) {
                    total += hist_cm1[static_cast<std::size_t>(step)][static_cast<std::size_t>(neuron)];
                    total += hist_cm2[static_cast<std::size_t>(step)][static_cast<std::size_t>(neuron)];
                    total += hist_cm3[static_cast<std::size_t>(step)][static_cast<std::size_t>(neuron)];
                }
            }
            pop_counts[static_cast<std::size_t>(window) * static_cast<std::size_t>(cfg.n_pop) + static_cast<std::size_t>(pop)] = total;
        }
    }

    Stage3Result result;
    result.pop_spk = pop_counts;
    result.qq = DecodePopulationSpikes(pop_counts, cfg.n_pop);
    result.w_mmbg1 = w_mmbg1;
    result.w_mmbg2 = w_mmbg2;

    const StrokePath& start_path_source = data.x1y1;
    StrokePath start_path;
    start_path.x.push_back(start_path_source.x.front());
    start_path.y.push_back(start_path_source.y.front());
    const JointTrajectory start_joint = BuildDesiredJointTrajectory(start_path, ArmParameters{}, cfg.verify_dt_s);

    if (mode == Stage3Mode::TestD) {
        const std::vector<double> a3_first = FirstJointPair(data.a3_flat);
        result.replay = ReplayForwardDynamicsWithReset(
            result.qq,
            ArmParameters{},
            cfg.verify_dt_s,
            start_joint.theta1.front(),
            start_joint.theta2.front(),
            451,
            a3_first[0],
            a3_first[1]);
    } else {
        JointTrajectory desired;
        desired.theta1.resize(result.qq.q1.size() + 1);
        desired.theta2.resize(result.qq.q2.size() + 1);
        desired.theta1[0] = start_joint.theta1.front();
        desired.theta2[0] = start_joint.theta2.front();
        result.replay = ReplayForwardDynamics(desired, result.qq, ArmParameters{}, cfg.verify_dt_s);
    }
    return result;
}

void WriteStage3Outputs(const std::filesystem::path& output_directory, const Stage3Result& result, const StrokePath& target_path) {
    std::filesystem::create_directories(output_directory);
    {
        std::ofstream out(output_directory / "QQ.tsv");
        out << "torque1\ttorque2\n";
        for (std::size_t i = 0; i < result.qq.q1.size(); ++i) {
            out << result.qq.q1[i] << '\t' << result.qq.q2[i] << '\n';
        }
    }
    {
        std::ofstream out(output_directory / "pop_spk.tsv");
        const int n_pop = 8;
        out << "window";
        for (int i = 0; i < n_pop; ++i) out << "\tpop_" << i;
        out << '\n';
        const std::size_t windows = result.pop_spk.size() / static_cast<std::size_t>(n_pop);
        for (std::size_t w = 0; w < windows; ++w) {
            out << w;
            for (int i = 0; i < n_pop; ++i) out << '\t' << result.pop_spk[w * static_cast<std::size_t>(n_pop) + static_cast<std::size_t>(i)];
            out << '\n';
        }
    }
    {
        std::ofstream out(output_directory / "replay_path.tsv");
        out << "x\ty\n";
        for (std::size_t i = 0; i < result.replay.hand_path.x.size(); ++i) {
            out << result.replay.hand_path.x[i] << '\t' << result.replay.hand_path.y[i] << '\n';
        }
    }
    {
        std::ofstream out(output_directory / "target_path.tsv");
        out << "x\ty\n";
        for (std::size_t i = 0; i < target_path.x.size() && i < target_path.y.size(); ++i) {
            out << target_path.x[i] << '\t' << target_path.y[i] << '\n';
        }
    }
    {
        std::ofstream out(output_directory / "summary.tsv");
        out << "metric\tvalue\n";
        const std::size_t count_x = std::min(target_path.x.size(), result.replay.hand_path.x.size());
        const std::size_t count_y = std::min(target_path.y.size(), result.replay.hand_path.y.size());
        if (count_x > 0 && count_y > 0) {
            const std::vector<double> target_x(target_path.x.begin(), target_path.x.begin() + static_cast<std::ptrdiff_t>(count_x));
            const std::vector<double> replay_x(result.replay.hand_path.x.begin(), result.replay.hand_path.x.begin() + static_cast<std::ptrdiff_t>(count_x));
            const std::vector<double> target_y(target_path.y.begin(), target_path.y.begin() + static_cast<std::ptrdiff_t>(count_y));
            const std::vector<double> replay_y(result.replay.hand_path.y.begin(), result.replay.hand_path.y.begin() + static_cast<std::ptrdiff_t>(count_y));
            out << "path_mse_x\t" << MeanSquaredError(target_x, replay_x) << '\n';
            out << "path_mse_y\t" << MeanSquaredError(target_y, replay_y) << '\n';
        } else {
            out << "path_mse_x\tNA\npath_mse_y\tNA\n";
        }
    }
    if (!result.w_mmbg1.values.empty()) {
        WriteMatrixTsv(output_directory / "w_mmbg1_final.tsv", result.w_mmbg1);
    }
    if (!result.w_mmbg2.values.empty()) {
        WriteMatrixTsv(output_directory / "w_mmbg2_final.tsv", result.w_mmbg2);
    }
}

}  // namespace handwriting
