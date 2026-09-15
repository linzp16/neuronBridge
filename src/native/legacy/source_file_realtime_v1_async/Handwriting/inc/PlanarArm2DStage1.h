#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace handwriting {

struct ArmParameters {
    double link1 = 0.34;
    double link2 = 0.32;
    double mass1 = 1.8;
    double mass2 = 1.6;
};

struct StrokePath {
    std::vector<double> x;
    std::vector<double> y;
};

struct JointTrajectory {
    std::vector<double> theta1;
    std::vector<double> theta2;
    std::vector<double> dtheta1;
    std::vector<double> dtheta2;
    std::vector<double> ddtheta1;
    std::vector<double> ddtheta2;
};

struct TorqueTrajectory {
    std::vector<double> q1;
    std::vector<double> q2;
};

struct SimulatedTrajectory {
    std::vector<double> theta1;
    std::vector<double> theta2;
    std::vector<double> dtheta1;
    std::vector<double> dtheta2;
    std::vector<double> ddtheta1;
    std::vector<double> ddtheta2;
    StrokePath hand_path;
};

struct StrokeComputation {
    StrokePath desired_path;
    JointTrajectory joints;
    TorqueTrajectory torques;
    SimulatedTrajectory replay;
};

struct JointSample {
    double theta1 = 0.0;
    double theta2 = 0.0;
};

struct AccelerationSample {
    double ddtheta1 = 0.0;
    double ddtheta2 = 0.0;
};

double Clamp(double value, double min_value, double max_value);
double MeanSquaredError(const std::vector<double>& lhs, const std::vector<double>& rhs);

std::vector<double> LoadNumericSeries(const std::filesystem::path& file_path);
StrokePath LoadStrokePathFromDirectory(const std::filesystem::path& directory, const std::string& stem);

JointSample InverseKinematics(double x, double y, const ArmParameters& params);
JointTrajectory BuildDesiredJointTrajectory(const StrokePath& path, const ArmParameters& params, double dt_seconds);
TorqueTrajectory BuildDesiredTorqueTrajectory(const JointTrajectory& joints, const ArmParameters& params);
AccelerationSample JointAcceleration(
    double theta1,
    double theta2,
    double dtheta1,
    double dtheta2,
    double torque1,
    double torque2,
    const ArmParameters& params);
StrokePath HandPosition(const std::vector<double>& theta1, const std::vector<double>& theta2, const ArmParameters& params);
SimulatedTrajectory ReplayForwardDynamics(
    const JointTrajectory& desired_joints,
    const TorqueTrajectory& torques,
    const ArmParameters& params,
    double dt_seconds);
StrokeComputation ComputeStroke(const StrokePath& path, const ArmParameters& params, double dt_seconds);

void WriteStrokeOutputs(const std::filesystem::path& output_directory, const std::string& stem, const StrokeComputation& result);

}  // namespace handwriting
