#include "../source_file_realtime_v1_async/Handwriting/inc/HandwritingStage2.h"

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace handwriting {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<int> ParseShapeLine(const std::filesystem::path& path, int expected_count) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open shape file " + path.string());
    }

    std::vector<int> dims;
    int value = 0;
    while (in >> value) {
        dims.push_back(value);
    }
    if (static_cast<int>(dims.size()) != expected_count) {
        throw std::runtime_error("Unexpected rank in shape file " + path.string());
    }
    return dims;
}

template <typename T>
std::vector<T> ReadBinaryFile(const std::filesystem::path& path, std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open binary file " + path.string());
    }

    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        throw std::runtime_error("Failed to read expected bytes from " + path.string());
    }
    return values;
}

}  // namespace

Shape2D LoadShape2D(const std::filesystem::path& path) {
    const std::vector<int> dims = ParseShapeLine(path, 2);
    Shape2D shape;
    shape.rows = dims[0];
    shape.cols = dims[1];
    return shape;
}

Shape3D LoadShape3D(const std::filesystem::path& path) {
    const std::vector<int> dims = ParseShapeLine(path, 3);
    Shape3D shape;
    shape.dim0 = dims[0];
    shape.dim1 = dims[1];
    shape.dim2 = dims[2];
    return shape;
}

BinarySpikeSeries LoadBinarySpikeSeries(const std::filesystem::path& data_path, const std::filesystem::path& shape_path) {
    BinarySpikeSeries series;
    series.shape = LoadShape2D(shape_path);
    const std::size_t count = static_cast<std::size_t>(series.shape.rows) * static_cast<std::size_t>(series.shape.cols);
    series.values = ReadBinaryFile<std::uint8_t>(data_path, count);
    return series;
}

BinaryWeightWindows LoadBinaryWeightWindows(const std::filesystem::path& data_path, const std::filesystem::path& shape_path) {
    BinaryWeightWindows windows;
    windows.shape = LoadShape3D(shape_path);
    const std::size_t count =
        static_cast<std::size_t>(windows.shape.dim0) *
        static_cast<std::size_t>(windows.shape.dim1) *
        static_cast<std::size_t>(windows.shape.dim2);
    windows.values = ReadBinaryFile<float>(data_path, count);
    return windows;
}

std::vector<int> BuildExternalSpikeTimes(const BinarySpikeSeries& spikes) {
    std::vector<int> times;
    const int rows = spikes.shape.rows;
    const int cols = spikes.shape.cols;
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            if (spikes.values[static_cast<std::size_t>(col) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)] != 0) {
                times.push_back(col + 1);
            }
        }
    }
    return times;
}

std::vector<int> BuildExternalSpikeNeurons(const BinarySpikeSeries& spikes) {
    std::vector<int> neurons;
    const int rows = spikes.shape.rows;
    const int cols = spikes.shape.cols;
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            if (spikes.values[static_cast<std::size_t>(col) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)] != 0) {
                neurons.push_back(row);
            }
        }
    }
    return neurons;
}

std::vector<std::array<double, 2> > PopulationDirections(int n_pop) {
    std::vector<std::array<double, 2> > directions(static_cast<std::size_t>(n_pop));
    for (int i = 0; i < n_pop; ++i) {
        const double angle = (360.0 * static_cast<double>(i) / static_cast<double>(n_pop)) * kPi / 180.0;
        directions[static_cast<std::size_t>(i)] = {std::cos(angle), std::sin(angle)};
    }
    return directions;
}

TorqueTrajectory DecodePopulationSpikes(const std::vector<int>& counts_by_window_and_population, int n_pop) {
    if (counts_by_window_and_population.size() % static_cast<std::size_t>(n_pop) != 0) {
        throw std::runtime_error("Population spike count vector length is not divisible by n_pop");
    }

    const std::vector<std::array<double, 2> > directions = PopulationDirections(n_pop);
    const std::size_t window_count = counts_by_window_and_population.size() / static_cast<std::size_t>(n_pop);
    TorqueTrajectory torque;
    torque.q1.resize(window_count);
    torque.q2.resize(window_count);

    for (std::size_t window = 0; window < window_count; ++window) {
        double q1 = 0.0;
        double q2 = 0.0;
        for (int pop = 0; pop < n_pop; ++pop) {
            const int count = counts_by_window_and_population[window * static_cast<std::size_t>(n_pop) + static_cast<std::size_t>(pop)];
            q1 += directions[static_cast<std::size_t>(pop)][0] * static_cast<double>(count);
            q2 += directions[static_cast<std::size_t>(pop)][1] * static_cast<double>(count);
        }
        torque.q1[window] = q1 * 1.0e-3;
        torque.q2[window] = q2 * 1.0e-3;
    }

    return torque;
}

CmForwardResult SimulateCmForwardExact(
    const Stage2Config& cfg,
    const BinarySpikeSeries& ext_spikes,
    const BinaryWeightWindows& weights) {
    if (ext_spikes.shape.rows != cfg.n_ext) {
        throw std::runtime_error("CM forward received unexpected external spike row count");
    }
    if (weights.shape.dim0 != cfg.n_ext || weights.shape.dim2 != cfg.n_cm) {
        throw std::runtime_error("CM forward received unexpected weight tensor shape");
    }
    if (weights.shape.dim1 * static_cast<int>(10.0 / cfg.dt_ms) != ext_spikes.shape.cols) {
        throw std::runtime_error("Weight window count and spike timeline are inconsistent");
    }

    const int step_count = ext_spikes.shape.cols;
    const int window_count = weights.shape.dim1;
    const int window_steps = static_cast<int>(10.0 / cfg.dt_ms);
    const int refractory_steps = static_cast<int>(cfg.refractory_ms / cfg.dt_ms);
    const int neurons_per_pop = cfg.n_cm / cfg.n_pop;

    std::vector<float> vm(static_cast<std::size_t>(cfg.n_cm), cfg.v_reset);
    std::vector<float> s_ampa_ext(static_cast<std::size_t>(cfg.n_cm), 0.0f);
    std::vector<std::uint8_t> cm_spikes(static_cast<std::size_t>(cfg.n_cm) * static_cast<std::size_t>(step_count), 0);
    std::vector<float> input_ext(static_cast<std::size_t>(cfg.n_cm), 0.0f);
    std::vector<int> population_counts(static_cast<std::size_t>(window_count) * static_cast<std::size_t>(cfg.n_pop), 0);

    int current_window = 0;
    const std::size_t weight_block = static_cast<std::size_t>(cfg.n_ext) * static_cast<std::size_t>(cfg.n_cm);

    for (int step = 0; step < step_count; ++step) {
        std::fill(input_ext.begin(), input_ext.end(), 0.0f);
        const std::size_t window_offset = static_cast<std::size_t>(current_window) * weight_block;

        for (int source = 0; source < cfg.n_ext; ++source) {
            const std::uint8_t spike = ext_spikes.values[static_cast<std::size_t>(step) * static_cast<std::size_t>(cfg.n_ext) + static_cast<std::size_t>(source)];
            if (spike == 0) {
                continue;
            }
            for (int target = 0; target < cfg.n_cm; ++target) {
                const std::size_t local = static_cast<std::size_t>(target) * static_cast<std::size_t>(cfg.n_ext) + static_cast<std::size_t>(source);
                input_ext[static_cast<std::size_t>(target)] += weights.values[window_offset + local];
            }
        }

        for (int target = 0; target < cfg.n_cm; ++target) {
            const std::size_t idx = static_cast<std::size_t>(target);
            s_ampa_ext[idx] += static_cast<float>((-s_ampa_ext[idx] / cfg.tau_ampa + input_ext[idx]) * cfg.dt_ms);
            const float i_ex_ext = -cfg.g_ext_cm * s_ampa_ext[idx] * (vm[idx] - cfg.e_ampa);
            const float i_in = -cfg.g_l * (vm[idx] - cfg.e_leak);
            const float dvm = (i_in + i_ex_ext) / cfg.cm_membrane_capacitance;
            vm[idx] += dvm * static_cast<float>(cfg.dt_ms);

            float v = vm[idx];
            std::uint8_t spike = 0;
            if (v >= cfg.v_th) {
                v = cfg.v_spike;
                vm[idx] = cfg.v_reset;
            }
            if (v > cfg.v_th) {
                spike = 1;
            }

            const int refractory_start = std::max(0, step - refractory_steps);
            bool refractory = false;
            for (int past = refractory_start; past < step; ++past) {
                if (cm_spikes[static_cast<std::size_t>(past) * static_cast<std::size_t>(cfg.n_cm) + idx] != 0) {
                    refractory = true;
                    break;
                }
            }
            if (refractory) {
                spike = 0;
                v = cfg.v_reset;
                vm[idx] = cfg.v_reset;
            }

            cm_spikes[static_cast<std::size_t>(step) * static_cast<std::size_t>(cfg.n_cm) + idx] = spike;
        }

        const int next_step_1based = step + 1;
        const int current_window_end = (current_window + 1) * window_steps;
        if (next_step_1based == current_window_end && current_window + 1 < window_count) {
            current_window += 1;
        }
    }

    for (int window = 0; window < window_count; ++window) {
        const int end_exclusive = (window + 1) * window_steps;
        const int begin_inclusive = end_exclusive - window_steps;
        for (int pop = 0; pop < cfg.n_pop; ++pop) {
            const int neuron_begin = pop * neurons_per_pop;
            const int neuron_end = neuron_begin + neurons_per_pop;
            int total = 0;
            for (int step = begin_inclusive; step < end_exclusive; ++step) {
                for (int neuron = neuron_begin; neuron < neuron_end; ++neuron) {
                    total += static_cast<int>(cm_spikes[static_cast<std::size_t>(step) * static_cast<std::size_t>(cfg.n_cm) + static_cast<std::size_t>(neuron)]);
                }
            }
            population_counts[static_cast<std::size_t>(window) * static_cast<std::size_t>(cfg.n_pop) + static_cast<std::size_t>(pop)] = total;
        }
    }

    CmForwardResult result;
    result.cm_spikes = std::move(cm_spikes);
    result.population_counts = std::move(population_counts);
    return result;
}

std::filesystem::path FindExistingStage2Input(const std::filesystem::path& directory, const std::string& stem, const std::string& extension) {
    return directory / (stem + extension);
}

}  // namespace handwriting
