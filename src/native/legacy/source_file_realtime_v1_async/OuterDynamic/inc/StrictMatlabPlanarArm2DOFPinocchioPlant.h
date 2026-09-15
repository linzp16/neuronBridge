#ifndef STRICT_MATLAB_PLANAR_ARM_2DOF_PINOCCHIO_PLANT_H
#define STRICT_MATLAB_PLANAR_ARM_2DOF_PINOCCHIO_PLANT_H

#include "../source_file_realtime_v1_async/OuterDynamic/inc/StrictMatlabPlanarArm2DOF.h"

#include <array>
#include <memory>

struct StrictIkOptions {
    int max_iter = 100;
    double tolerance = 1.0e-6;
    double damping = 1.0e-4;
    double step_size = 0.5;
    bool clamp_joint_limits = true;
    bool verbose = false;
};

struct StrictIkResult {
    bool success = false;
    int iterations = 0;
    double final_error = 0.0;
    std::array<double, 2> q_solution{ {0.0, 0.0} };
};

class StrictMatlabPlanarArm2DOFPinocchioPlant {
public:
    explicit StrictMatlabPlanarArm2DOFPinocchioPlant(const StrictMatlabPlanarArm2DOFConfig& config);
    ~StrictMatlabPlanarArm2DOFPinocchioPlant();

    StrictMatlabPlanarArm2DOFPinocchioPlant(StrictMatlabPlanarArm2DOFPinocchioPlant&&) noexcept;
    StrictMatlabPlanarArm2DOFPinocchioPlant& operator=(StrictMatlabPlanarArm2DOFPinocchioPlant&&) noexcept;

    StrictMatlabPlanarArm2DOFPinocchioPlant(const StrictMatlabPlanarArm2DOFPinocchioPlant&) = delete;
    StrictMatlabPlanarArm2DOFPinocchioPlant& operator=(const StrictMatlabPlanarArm2DOFPinocchioPlant&) = delete;

    void ResetState(const std::array<double, 2>& q, const std::array<double, 2>& qd);
    void SetDesiredState(const std::array<double, 2>& q_des, const std::array<double, 2>& qd_des);
    void SetTorqueCommand(const std::array<double, 2>& tau);
    void Step(double interval_s);

    const StrictMatlabPlanarArm2DOFState& GetState() const;
    std::array<double, 2> HandPosition() const;
    std::array<double, 2> ClampToJointLimits(const std::array<double, 2>& q) const;
    std::array<std::array<double, 2>, 2> EndEffectorJacobianXY(const std::array<double, 2>& q) const;
    std::array<double, 2> CartesianVelocityXY(
        const std::array<double, 2>& q,
        const std::array<double, 2>& qd) const;
    std::array<double, 2> JointVelocityFromCartesianXY(
        const std::array<double, 2>& q,
        const std::array<double, 2>& xy_dot,
        const StrictIkOptions& options) const;
    std::array<double, 2> InverseKinematicsXY(
        const std::array<double, 2>& q_init,
        const std::array<double, 2>& target_xy,
        const StrictIkOptions& options,
        StrictIkResult* result = NULL) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
