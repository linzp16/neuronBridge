#include "../source_file_realtime_v1_async/OuterDynamic/inc/StrictMatlabPlanarArm2DOFPinocchioPlant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if SNN_WITH_PINOCCHIO
#include <Eigen/Dense>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody.hpp>
#include <pinocchio/spatial.hpp>
#endif

namespace {

double ClampValue(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(max_value, value));
}

std::array<double, 2> EigenToArray(const Eigen::Vector2d& vector) {
    return { vector(0), vector(1) };
}

Eigen::Vector2d ArrayToEigen(const std::array<double, 2>& values) {
    return Eigen::Vector2d(values[0], values[1]);
}

}

struct StrictMatlabPlanarArm2DOFPinocchioPlant::Impl {
    StrictMatlabPlanarArm2DOFConfig config;
    StrictMatlabPlanarArm2DOFState state;

#if SNN_WITH_PINOCCHIO
    pinocchio::Model model;
    pinocchio::Data data;
    pinocchio::FrameIndex end_effector_frame_id;
    Eigen::Vector2d q;
    Eigen::Vector2d qd;
#endif

    explicit Impl(const StrictMatlabPlanarArm2DOFConfig& cfg)
        : config(cfg), state()
#if SNN_WITH_PINOCCHIO
        , model(), data(model), end_effector_frame_id(0), q(Eigen::Vector2d::Zero()), qd(Eigen::Vector2d::Zero())
#endif
    {
#if SNN_WITH_PINOCCHIO
        pinocchio::Model temp_model;
        temp_model.gravity.linear() = Eigen::Vector3d(0.0, 0.0, 0.0);

        const double inertia1 = this->config.mass1 * this->config.link1 * this->config.link1 / 3.0;
        const double inertia2 = this->config.mass2 * this->config.link2 * this->config.link2 / 12.0;

        const pinocchio::JointIndex joint1 =
            temp_model.addJoint(0, pinocchio::JointModelRZ(), pinocchio::SE3::Identity(), "joint1");

        Eigen::Matrix3d link1_inertia_matrix = Eigen::Matrix3d::Zero();
        link1_inertia_matrix(0, 0) = 1.0e-6;
        link1_inertia_matrix(1, 1) = inertia1;
        link1_inertia_matrix(2, 2) = inertia1;

        temp_model.appendBodyToJoint(
            joint1,
            pinocchio::Inertia(
                this->config.mass1,
                Eigen::Vector3d(this->config.link1 * 0.5, 0.0, 0.0),
                link1_inertia_matrix),
            pinocchio::SE3::Identity());

        const pinocchio::SE3 joint2_placement(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d(this->config.link1, 0.0, 0.0));

        const pinocchio::JointIndex joint2 =
            temp_model.addJoint(joint1, pinocchio::JointModelRZ(), joint2_placement, "joint2");

        Eigen::Matrix3d link2_inertia_matrix = Eigen::Matrix3d::Zero();
        link2_inertia_matrix(0, 0) = 1.0e-6;
        link2_inertia_matrix(1, 1) = inertia2;
        link2_inertia_matrix(2, 2) = inertia2;

        temp_model.appendBodyToJoint(
            joint2,
            pinocchio::Inertia(
                this->config.mass2,
                Eigen::Vector3d(this->config.link2 * 0.5, 0.0, 0.0),
                link2_inertia_matrix),
            pinocchio::SE3::Identity());

        temp_model.addFrame(
            pinocchio::Frame(
                "end_effector",
                joint2,
                joint2,
                pinocchio::SE3(
                    Eigen::Matrix3d::Identity(),
                    Eigen::Vector3d(this->config.link2, 0.0, 0.0)),
                pinocchio::OP_FRAME));

        this->model = std::move(temp_model);
        this->data = pinocchio::Data(this->model);
        this->end_effector_frame_id = this->model.getFrameId("end_effector");
#endif
    }
};

StrictMatlabPlanarArm2DOFPinocchioPlant::StrictMatlabPlanarArm2DOFPinocchioPlant(const StrictMatlabPlanarArm2DOFConfig& config)
    : impl_(new Impl(config)) {
#if !SNN_WITH_PINOCCHIO
    (void)config;
#endif
}

StrictMatlabPlanarArm2DOFPinocchioPlant::~StrictMatlabPlanarArm2DOFPinocchioPlant() {}

StrictMatlabPlanarArm2DOFPinocchioPlant::StrictMatlabPlanarArm2DOFPinocchioPlant(StrictMatlabPlanarArm2DOFPinocchioPlant&&) noexcept = default;
StrictMatlabPlanarArm2DOFPinocchioPlant& StrictMatlabPlanarArm2DOFPinocchioPlant::operator=(StrictMatlabPlanarArm2DOFPinocchioPlant&&) noexcept = default;

void StrictMatlabPlanarArm2DOFPinocchioPlant::ResetState(const std::array<double, 2>& q, const std::array<double, 2>& qd) {
    this->impl_->state.q = q;
    this->impl_->state.qd = qd;
    this->impl_->state.qdd = { 0.0, 0.0 };
    this->impl_->state.tau = { 0.0, 0.0 };
#if SNN_WITH_PINOCCHIO
    this->impl_->q << q[0], q[1];
    this->impl_->qd << qd[0], qd[1];
#endif
}

void StrictMatlabPlanarArm2DOFPinocchioPlant::SetDesiredState(const std::array<double, 2>& q_des, const std::array<double, 2>& qd_des) {
    this->impl_->state.q_des = q_des;
    this->impl_->state.qd_des = qd_des;
}

void StrictMatlabPlanarArm2DOFPinocchioPlant::SetTorqueCommand(const std::array<double, 2>& tau) {
    this->impl_->state.tau = tau;
}

void StrictMatlabPlanarArm2DOFPinocchioPlant::Step(double interval_s) {
#if SNN_WITH_PINOCCHIO
    const Eigen::Vector2d tau_input(this->impl_->state.tau[0], this->impl_->state.tau[1]);
    const Eigen::Vector2d qdd = pinocchio::aba(this->impl_->model, this->impl_->data, this->impl_->q, this->impl_->qd, tau_input);
    this->impl_->state.qdd = { qdd(0), qdd(1) };

    const std::array<double, 2> old_qd = this->impl_->state.qd;
    this->impl_->state.qd[0] += this->impl_->state.qdd[0] * interval_s;
    this->impl_->state.qd[1] += this->impl_->state.qdd[1] * interval_s;
    this->impl_->state.q[0] += old_qd[0] * interval_s;
    this->impl_->state.q[1] += old_qd[1] * interval_s;
    this->impl_->state.q[0] = ClampValue(this->impl_->state.q[0], this->impl_->config.q1_min, this->impl_->config.q1_max);
    this->impl_->state.q[1] = ClampValue(this->impl_->state.q[1], this->impl_->config.q2_min, this->impl_->config.q2_max);
    this->impl_->q << this->impl_->state.q[0], this->impl_->state.q[1];
    this->impl_->qd << this->impl_->state.qd[0], this->impl_->state.qd[1];
#else
    (void)interval_s;
    throw std::runtime_error("StrictMatlabPlanarArm2DOFPinocchioPlant requires SNN_WITH_PINOCCHIO=1.");
#endif
}

const StrictMatlabPlanarArm2DOFState& StrictMatlabPlanarArm2DOFPinocchioPlant::GetState() const {
    return this->impl_->state;
}

std::array<double, 2> StrictMatlabPlanarArm2DOFPinocchioPlant::HandPosition() const {
#if SNN_WITH_PINOCCHIO
    pinocchio::forwardKinematics(this->impl_->model, this->impl_->data, this->impl_->q);
    pinocchio::updateFramePlacements(this->impl_->model, this->impl_->data);
    const pinocchio::SE3& placement = this->impl_->data.oMf[this->impl_->end_effector_frame_id];
    return { placement.translation()[0], placement.translation()[1] };
#else
    throw std::runtime_error("StrictMatlabPlanarArm2DOFPinocchioPlant requires SNN_WITH_PINOCCHIO=1.");
#endif
}

std::array<double, 2> StrictMatlabPlanarArm2DOFPinocchioPlant::ClampToJointLimits(const std::array<double, 2>& q) const {
    return {
        ClampValue(q[0], this->impl_->config.q1_min, this->impl_->config.q1_max),
        ClampValue(q[1], this->impl_->config.q2_min, this->impl_->config.q2_max)
    };
}

std::array<std::array<double, 2>, 2> StrictMatlabPlanarArm2DOFPinocchioPlant::EndEffectorJacobianXY(
    const std::array<double, 2>& q) const {
#if SNN_WITH_PINOCCHIO
    const Eigen::Vector2d q_vector = ArrayToEigen(q);
    Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian(6, this->impl_->model.nv);
    jacobian.setZero();
    pinocchio::computeFrameJacobian(
        this->impl_->model,
        this->impl_->data,
        q_vector,
        this->impl_->end_effector_frame_id,
        pinocchio::LOCAL_WORLD_ALIGNED,
        jacobian);
    return {
        std::array<double, 2>{ jacobian(0, 0), jacobian(0, 1) },
        std::array<double, 2>{ jacobian(1, 0), jacobian(1, 1) }
    };
#else
    (void)q;
    throw std::runtime_error("StrictMatlabPlanarArm2DOFPinocchioPlant requires SNN_WITH_PINOCCHIO=1.");
#endif
}

std::array<double, 2> StrictMatlabPlanarArm2DOFPinocchioPlant::CartesianVelocityXY(
    const std::array<double, 2>& q,
    const std::array<double, 2>& qd) const {
#if SNN_WITH_PINOCCHIO
    const std::array<std::array<double, 2>, 2> jacobian = this->EndEffectorJacobianXY(q);
    const Eigen::Vector2d qd_vector = ArrayToEigen(qd);
    Eigen::Matrix2d j_matrix;
    j_matrix <<
        jacobian[0][0], jacobian[0][1],
        jacobian[1][0], jacobian[1][1];
    return EigenToArray(j_matrix * qd_vector);
#else
    (void)q;
    (void)qd;
    throw std::runtime_error("StrictMatlabPlanarArm2DOFPinocchioPlant requires SNN_WITH_PINOCCHIO=1.");
#endif
}

std::array<double, 2> StrictMatlabPlanarArm2DOFPinocchioPlant::JointVelocityFromCartesianXY(
    const std::array<double, 2>& q,
    const std::array<double, 2>& xy_dot,
    const StrictIkOptions& options) const {
#if SNN_WITH_PINOCCHIO
    const std::array<std::array<double, 2>, 2> jacobian = this->EndEffectorJacobianXY(q);
    Eigen::Matrix2d j_matrix;
    j_matrix <<
        jacobian[0][0], jacobian[0][1],
        jacobian[1][0], jacobian[1][1];

    const Eigen::Vector2d xy_dot_vector = ArrayToEigen(xy_dot);
    const double damping = std::max(1.0e-12, options.damping);
    const Eigen::Matrix2d jj_t = j_matrix * j_matrix.transpose();
    const Eigen::Vector2d qd_vector =
        j_matrix.transpose() * (jj_t + damping * Eigen::Matrix2d::Identity()).ldlt().solve(xy_dot_vector);
    return EigenToArray(qd_vector);
#else
    (void)q;
    (void)xy_dot;
    (void)options;
    throw std::runtime_error("StrictMatlabPlanarArm2DOFPinocchioPlant requires SNN_WITH_PINOCCHIO=1.");
#endif
}

std::array<double, 2> StrictMatlabPlanarArm2DOFPinocchioPlant::InverseKinematicsXY(
    const std::array<double, 2>& q_init,
    const std::array<double, 2>& target_xy,
    const StrictIkOptions& options,
    StrictIkResult* result) const {
#if SNN_WITH_PINOCCHIO
    Eigen::Vector2d q_current = ArrayToEigen(options.clamp_joint_limits ? this->ClampToJointLimits(q_init) : q_init);
    const Eigen::Vector2d target = ArrayToEigen(target_xy);
    StrictIkResult local_result;
    local_result.q_solution = EigenToArray(q_current);

    for (int iter = 0; iter < std::max(1, options.max_iter); ++iter) {
        pinocchio::forwardKinematics(this->impl_->model, this->impl_->data, q_current);
        pinocchio::updateFramePlacements(this->impl_->model, this->impl_->data);
        const pinocchio::SE3& placement = this->impl_->data.oMf[this->impl_->end_effector_frame_id];
        const Eigen::Vector2d current_xy(placement.translation()[0], placement.translation()[1]);
        const Eigen::Vector2d error = target - current_xy;
        const double error_norm = error.norm();

        local_result.iterations = iter + 1;
        local_result.final_error = error_norm;
        local_result.q_solution = EigenToArray(q_current);
        if (error_norm <= options.tolerance) {
            local_result.success = true;
            break;
        }

        Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian(6, this->impl_->model.nv);
        jacobian.setZero();
        pinocchio::computeFrameJacobian(
            this->impl_->model,
            this->impl_->data,
            q_current,
            this->impl_->end_effector_frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED,
            jacobian);
        Eigen::Matrix2d j_xy;
        j_xy <<
            jacobian(0, 0), jacobian(0, 1),
            jacobian(1, 0), jacobian(1, 1);

        const double damping = std::max(1.0e-12, options.damping);
        const Eigen::Matrix2d jj_t = j_xy * j_xy.transpose();
        const Eigen::Vector2d delta_q =
            options.step_size * j_xy.transpose() *
            (jj_t + damping * Eigen::Matrix2d::Identity()).ldlt().solve(error);
        q_current += delta_q;
        if (options.clamp_joint_limits) {
            q_current = ArrayToEigen(this->ClampToJointLimits(EigenToArray(q_current)));
        }
    }

    if (result != NULL) {
        *result = local_result;
    }
    return local_result.q_solution;
#else
    (void)q_init;
    (void)target_xy;
    (void)options;
    (void)result;
    throw std::runtime_error("StrictMatlabPlanarArm2DOFPinocchioPlant requires SNN_WITH_PINOCCHIO=1.");
#endif
}
