#include "../source_file_realtime_v1_async/Handwriting/inc/PlanarArm2DStage1.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace handwriting {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::filesystem::path FindExistingSeriesPath(const std::filesystem::path& directory, const std::string& stem) {
    static const char* kExtensions[] = {".tsv", ".txt", ".csv"};
    for (const char* extension : kExtensions) {
        const std::filesystem::path candidate = directory / (stem + extension);
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    const std::filesystem::path matlab_file = directory / (stem + ".mat");
    if (std::filesystem::exists(matlab_file)) {
        throw std::runtime_error(
            "Found MATLAB file " + matlab_file.string() +
            " but this stage-1 C++ program reads exported text arrays (.tsv/.txt/.csv). "
            "Run export_stage1_series.m in the same folder first.");
    }

    throw std::runtime_error("Could not find input series for stem '" + stem + "' in " + directory.string());
}

std::vector<double> Gradient(const std::vector<double>& values, double dt_seconds) {
    const std::size_t count = values.size();
    if (count == 0) {
        return {};
    }
    if (count == 1) {
        return std::vector<double>(1, 0.0);
    }

    std::vector<double> gradient(count, 0.0);
    gradient[0] = (values[1] - values[0]) / dt_seconds;
    for (std::size_t i = 1; i + 1 < count; ++i) {
        gradient[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt_seconds);
    }
    gradient[count - 1] = (values[count - 1] - values[count - 2]) / dt_seconds;
    return gradient;
}

void WriteTwoColumnFile(
    const std::filesystem::path& file_path,
    const std::vector<double>& first,
    const std::vector<double>& second,
    const char* header_a,
    const char* header_b) {
    if (first.size() != second.size()) {
        throw std::runtime_error("Column size mismatch when writing " + file_path.string());
    }

    std::ofstream out(file_path);
    if (!out) {
        throw std::runtime_error("Failed to open output file " + file_path.string());
    }

    out << header_a << '\t' << header_b << '\n';
    for (std::size_t i = 0; i < first.size(); ++i) {
        out << first[i] << '\t' << second[i] << '\n';
    }
}

}  // namespace

double Clamp(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(max_value, value));
}

double MeanSquaredError(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("MeanSquaredError requires equal-length vectors");
    }
    if (lhs.empty()) {
        return 0.0;
    }

    double total = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const double diff = lhs[i] - rhs[i];
        total += diff * diff;
    }
    return total / static_cast<double>(lhs.size());
}

std::vector<double> LoadNumericSeries(const std::filesystem::path& file_path) {
    std::ifstream in(file_path);
    if (!in) {
        throw std::runtime_error("Failed to open input file " + file_path.string());
    }

    std::vector<double> values;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::stringstream ss(line);
        double value = 0.0;
        while (ss >> value) {
            values.push_back(value);
        }
    }

    if (values.empty()) {
        throw std::runtime_error("No numeric data found in " + file_path.string());
    }
    return values;
}

StrokePath LoadStrokePathFromDirectory(const std::filesystem::path& directory, const std::string& stem) {
    StrokePath path;
    path.x = LoadNumericSeries(FindExistingSeriesPath(directory, "X" + stem));
    path.y = LoadNumericSeries(FindExistingSeriesPath(directory, "Y" + stem));
    if (path.x.size() != path.y.size()) {
        throw std::runtime_error("X/Y length mismatch for stroke " + stem);
    }
    return path;
}

JointSample InverseKinematics(double x, double y, const ArmParameters& params) {
    const double l1 = params.link1;
    const double l2 = params.link2;
    const double radius_squared = x * x + y * y;
    const double cos_theta2_raw = (radius_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
    const double cos_theta2 = Clamp(cos_theta2_raw, -1.0, 1.0);
    const double theta2 = std::acos(cos_theta2);

    const double denom = -2.0 * l1 * std::sqrt(std::max(radius_squared, 1.0e-12));
    const double cos_phi_raw = (l2 * l2 - l1 * l1 - radius_squared) / denom;
    const double phi = std::acos(Clamp(cos_phi_raw, -1.0, 1.0));
    const double theta1 = std::atan2(y, x) - phi;

    JointSample sample;
    sample.theta1 = theta1;
    sample.theta2 = theta2;
    return sample;
}

JointTrajectory BuildDesiredJointTrajectory(const StrokePath& path, const ArmParameters& params, double dt_seconds) {
    JointTrajectory joints;
    const std::size_t count = path.x.size();
    joints.theta1.resize(count);
    joints.theta2.resize(count);

    for (std::size_t i = 0; i < count; ++i) {
        const JointSample sample = InverseKinematics(path.x[i], path.y[i], params);
        joints.theta1[i] = sample.theta1;
        joints.theta2[i] = sample.theta2;
    }

    joints.dtheta1 = Gradient(joints.theta1, dt_seconds);
    joints.dtheta2 = Gradient(joints.theta2, dt_seconds);
    joints.ddtheta1 = Gradient(joints.dtheta1, dt_seconds);
    joints.ddtheta2 = Gradient(joints.dtheta2, dt_seconds);
    return joints;
}

TorqueTrajectory BuildDesiredTorqueTrajectory(const JointTrajectory& joints, const ArmParameters& params) {
    const double l1 = params.link1;
    const double l2 = params.link2;
    const double m1 = params.mass1;
    const double m2 = params.mass2;
    const double d1 = l1 / 2.0;
    const double d2 = l2 / 2.0;
    const double i1 = (m1 * l1 * l1) / 3.0;
    const double i2 = (m2 * l2 * l2) / 12.0;

    const std::size_t count = joints.theta1.size();
    TorqueTrajectory torques;
    torques.q1.resize(count);
    torques.q2.resize(count);

    for (std::size_t i = 0; i < count; ++i) {
        const double theta2 = joints.theta2[i];
        const double dtheta1 = joints.dtheta1[i];
        const double dtheta2 = joints.dtheta2[i];
        const double ddtheta1 = joints.ddtheta1[i];
        const double ddtheta2 = joints.ddtheta2[i];
        const double c2 = std::cos(theta2);
        const double s2 = std::sin(theta2);

        torques.q1[i] =
            (i1 + i2 + 2.0 * m2 * l1 * d2 * c2 + m1 * d1 * d1 + m2 * (d2 * d2 + l1 * l1)) * ddtheta1 +
            (i2 + m2 * l1 * d2 * c2 + m2 * d2 * d2) * ddtheta2 +
            (-2.0 * m2 * l1 * d2 * s2) * dtheta1 * dtheta2 +
            (-m2 * l1 * d2 * s2) * dtheta2 * dtheta2;

        torques.q2[i] =
            (i2 + m2 * l1 * d2 * c2 + m2 * d2 * d2) * ddtheta1 +
            (i2 + m2 * d2 * d2) * ddtheta2 +
            (m2 * l1 * d2 * s2) * dtheta1 * dtheta1;
    }

    return torques;
}

AccelerationSample JointAcceleration(
    double theta1,
    double theta2,
    double dtheta1,
    double dtheta2,
    double torque1,
    double torque2,
    const ArmParameters& params) {
    (void)theta1;
    const double l1 = params.link1;
    const double l2 = params.link2;
    const double m1 = params.mass1;
    const double m2 = params.mass2;
    const double d1 = l1 / 2.0;
    const double d2 = l2 / 2.0;
    const double i1 = (m1 * l1 * l1) / 3.0;
    const double i2 = (m2 * l2 * l2) / 12.0;

    const double c2 = std::cos(theta2);
    const double s2 = std::sin(theta2);
    const double m11 = i1 + i2 + 2.0 * m2 * l1 * d2 * c2 + m1 * d1 * d1 + m2 * (d2 * d2 + l1 * l1);
    const double m12 = i2 + m2 * l1 * d2 * c2 + m2 * d2 * d2;
    const double m22 = i2 + m2 * d2 * d2;
    const double b1 = -2.0 * m2 * l1 * d2 * s2;
    const double c12 = -m2 * l1 * d2 * s2;
    const double c21 = m2 * l1 * d2 * s2;

    const double rhs1 = torque1 - b1 * dtheta1 * dtheta2 - c12 * dtheta2 * dtheta2;
    const double rhs2 = torque2 - c21 * dtheta1 * dtheta1;
    const double determinant = m11 * m22 - m12 * m12;
    if (std::abs(determinant) < 1.0e-12) {
        throw std::runtime_error("Singular mass matrix encountered in JointAcceleration");
    }

    AccelerationSample sample;
    sample.ddtheta1 = (rhs1 * m22 - rhs2 * m12) / determinant;
    sample.ddtheta2 = (m11 * rhs2 - m12 * rhs1) / determinant;
    return sample;
}

StrokePath HandPosition(const std::vector<double>& theta1, const std::vector<double>& theta2, const ArmParameters& params) {
    if (theta1.size() != theta2.size()) {
        throw std::runtime_error("Theta arrays must have equal length in HandPosition");
    }

    StrokePath path;
    const std::size_t count = theta1.size();
    path.x.resize(count);
    path.y.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        path.x[i] = params.link1 * std::cos(theta1[i]) + params.link2 * std::cos(theta1[i] + theta2[i]);
        path.y[i] = params.link1 * std::sin(theta1[i]) + params.link2 * std::sin(theta1[i] + theta2[i]);
    }
    return path;
}

SimulatedTrajectory ReplayForwardDynamics(
    const JointTrajectory& desired_joints,
    const TorqueTrajectory& torques,
    const ArmParameters& params,
    double dt_seconds) {
    if (torques.q1.size() != torques.q2.size()) {
        throw std::runtime_error("Torque vector length mismatch in ReplayForwardDynamics");
    }
    if (desired_joints.theta1.empty() || desired_joints.theta2.empty()) {
        throw std::runtime_error("Desired joint trajectory is empty");
    }
    if (torques.q1.size() < 2) {
        throw std::runtime_error("Need at least two torque samples in ReplayForwardDynamics");
    }

    const std::size_t step_count = torques.q1.size() - 1;

    SimulatedTrajectory sim;
    sim.theta1.resize(step_count + 1, desired_joints.theta1.front());
    sim.theta2.resize(step_count + 1, desired_joints.theta2.front());
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
    }

    sim.hand_path = HandPosition(sim.theta1, sim.theta2, params);
    return sim;
}

StrokeComputation ComputeStroke(const StrokePath& path, const ArmParameters& params, double dt_seconds) {
    StrokeComputation result;
    result.desired_path = path;
    result.joints = BuildDesiredJointTrajectory(path, params, dt_seconds);
    result.torques = BuildDesiredTorqueTrajectory(result.joints, params);
    result.replay = ReplayForwardDynamics(result.joints, result.torques, params, dt_seconds);
    return result;
}

void WriteStrokeOutputs(const std::filesystem::path& output_directory, const std::string& stem, const StrokeComputation& result) {
    std::filesystem::create_directories(output_directory);

    WriteTwoColumnFile(output_directory / ("A" + stem + ".tsv"), result.joints.theta1, result.joints.theta2, "theta1", "theta2");
    WriteTwoColumnFile(output_directory / ("Q" + stem + ".tsv"), result.torques.q1, result.torques.q2, "torque1", "torque2");
    WriteTwoColumnFile(output_directory / ("desired_path_" + stem + ".tsv"), result.desired_path.x, result.desired_path.y, "x", "y");
    WriteTwoColumnFile(output_directory / ("replay_path_" + stem + ".tsv"), result.replay.hand_path.x, result.replay.hand_path.y, "x", "y");

    std::ofstream summary(output_directory / ("summary_" + stem + ".tsv"));
    if (!summary) {
        throw std::runtime_error("Failed to open summary file for stroke " + stem);
    }
    summary << "metric\tvalue\n";
    summary << "path_mse_x\t" << MeanSquaredError(result.desired_path.x, result.replay.hand_path.x) << '\n';
    summary << "path_mse_y\t" << MeanSquaredError(result.desired_path.y, result.replay.hand_path.y) << '\n';
}

}  // namespace handwriting
