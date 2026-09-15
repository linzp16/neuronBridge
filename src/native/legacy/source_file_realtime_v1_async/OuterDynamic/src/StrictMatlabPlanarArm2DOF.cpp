#include "../source_file_realtime_v1_async/OuterDynamic/inc/StrictMatlabPlanarArm2DOF.h"

#include <algorithm>
#include <cmath>

StrictMatlabPlanarArm2DOF::StrictMatlabPlanarArm2DOF(const StrictMatlabPlanarArm2DOFConfig& config)
    : config_(config), state_() {}

void StrictMatlabPlanarArm2DOF::ResetState(const std::array<double, 2>& q, const std::array<double, 2>& qd) {
    state_.q = q;
    state_.qd = qd;
    state_.qdd = { 0.0, 0.0 };
    state_.tau = { 0.0, 0.0 };
}

void StrictMatlabPlanarArm2DOF::SetDesiredState(const std::array<double, 2>& q_des, const std::array<double, 2>& qd_des) {
    state_.q_des = q_des;
    state_.qd_des = qd_des;
}

void StrictMatlabPlanarArm2DOF::SetTorqueCommand(const std::array<double, 2>& tau) {
    state_.tau = tau;
}

void StrictMatlabPlanarArm2DOF::Step(double interval_s) {
    state_.qdd = JointAcceleration(state_.tau);
    const std::array<double, 2> old_qd = state_.qd;
    state_.qd[0] += state_.qdd[0] * interval_s;
    state_.qd[1] += state_.qdd[1] * interval_s;
    state_.q[0] += old_qd[0] * interval_s;
    state_.q[1] += old_qd[1] * interval_s;
    state_.q[0] = Clamp(state_.q[0], config_.q1_min, config_.q1_max);
    state_.q[1] = Clamp(state_.q[1], config_.q2_min, config_.q2_max);
}

const StrictMatlabPlanarArm2DOFState& StrictMatlabPlanarArm2DOF::GetState() const {
    return state_;
}

std::array<double, 2> StrictMatlabPlanarArm2DOF::HandPosition() const {
    return {
        config_.link1 * std::cos(state_.q[0]) + config_.link2 * std::cos(state_.q[0] + state_.q[1]),
        config_.link1 * std::sin(state_.q[0]) + config_.link2 * std::sin(state_.q[0] + state_.q[1])
    };
}

std::array<double, 2> StrictMatlabPlanarArm2DOF::JointAcceleration(const std::array<double, 2>& tau) const {
    const double theta2 = state_.q[1];
    const double d_theta1 = state_.qd[0];
    const double d_theta2 = state_.qd[1];
    const double d1 = config_.link1 * 0.5;
    const double d2 = config_.link2 * 0.5;
    const double i1 = (config_.mass1 * config_.link1 * config_.link1) / 3.0;
    const double i2 = (config_.mass2 * config_.link2 * config_.link2) / 12.0;
    const double c2 = std::cos(theta2);
    const double s2 = std::sin(theta2);

    const double m11 =
        i1 + i2 + 2.0 * config_.mass2 * config_.link1 * d2 * c2 +
        config_.mass1 * d1 * d1 + config_.mass2 * (d2 * d2 + config_.link1 * config_.link1);
    const double m12 = i2 + config_.mass2 * config_.link1 * d2 * c2 + config_.mass2 * d2 * d2;
    const double m21 = m12;
    const double m22 = i2 + config_.mass2 * d2 * d2;

    const double rhs1 =
        tau[0] + 2.0 * config_.mass2 * config_.link1 * d2 * s2 * d_theta1 * d_theta2 +
        config_.mass2 * config_.link1 * d2 * s2 * d_theta2 * d_theta2;
    const double rhs2 =
        tau[1] - config_.mass2 * config_.link1 * d2 * s2 * d_theta1 * d_theta1;

    const double det = m11 * m22 - m12 * m21;
    if (std::fabs(det) < 1.0e-12) {
        return { 0.0, 0.0 };
    }

    return {
        (rhs1 * m22 - rhs2 * m12) / det,
        (m11 * rhs2 - m21 * rhs1) / det
    };
}

double StrictMatlabPlanarArm2DOF::Clamp(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(max_value, value));
}
