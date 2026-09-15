#include "../source_file_realtime_v1_async/OuterDynamic/inc/PlanarArm2DOFPinocchio.h"

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeBuffer.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"

#include <boost/any.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

template <typename T>
T GetAnyOrDefault(const std::map<std::string, boost::any>& params, const std::string& key, const T& default_value) {
    std::map<std::string, boost::any>::const_iterator it = params.find(key);
    if (it == params.end()) {
        return default_value;
    }
    try {
        return boost::any_cast<T>(it->second);
    } catch (const boost::bad_any_cast&) {
        return default_value;
    }
}

Eigen::Vector2d GetVector2OrDefault(const std::map<std::string, boost::any>& params, const std::string& key, const Eigen::Vector2d& default_value) {
    std::map<std::string, boost::any>::const_iterator it = params.find(key);
    if (it == params.end()) {
        return default_value;
    }
    try {
        const std::vector<double> values = boost::any_cast<std::vector<double>>(it->second);
        if (values.size() != 2) {
            return default_value;
        }
        return Eigen::Vector2d(values[0], values[1]);
    } catch (const boost::bad_any_cast&) {
        return default_value;
    }
}

std::array<double, 2> GetArray2OrDefault(
    const std::map<std::string, boost::any>& params,
    const std::string& key,
    const std::array<double, 2>& default_value) {
    std::map<std::string, boost::any>::const_iterator it = params.find(key);
    if (it == params.end()) {
        return default_value;
    }
    try {
        const std::vector<double> values = boost::any_cast<std::vector<double> >(it->second);
        if (values.size() != 2) {
            return default_value;
        }
        return { values[0], values[1] };
    } catch (const boost::bad_any_cast&) {
        return default_value;
    }
}

constexpr double kPi = 3.14159265358979323846;

} // namespace

PlanarArm2DOFPinocchio::PlanarArm2DOFPinocchio(int timestep_size)
    : OuterDynamicModel(timestep_size),
      model_(),
      data_aba_(model_),
      data_rnea_(model_),
      data_fk_(model_),
      model_ready_(false),
      end_effector_frame_id_(0),
      q_(Eigen::Vector2d::Zero()),
      v_(Eigen::Vector2d::Zero()),
      qdd_(Eigen::Vector2d::Zero()),
      q_des_(Eigen::Vector2d::Zero()),
      v_des_(Eigen::Vector2d::Zero()),
      qdd_des_(Eigen::Vector2d::Zero()),
      tau_pd_(Eigen::Vector2d::Zero()),
      tau_id_(Eigen::Vector2d::Zero()),
      tau_cereb_(Eigen::Vector2d::Zero()),
      tau_cereb_positive_(Eigen::Vector2d::Zero()),
      tau_cereb_negative_(Eigen::Vector2d::Zero()),
      accumulated_joint_torque_(Eigen::Vector2d::Zero()),
      tau_total_(Eigen::Vector2d::Zero()),
      link_lengths_(1.0, 1.0),
      link_masses_(1.0, 1.0),
      damping_(0.05, 0.05),
      kp_(12.0, 10.0),
      kd_(2.0, 1.8),
      gravity_magnitude_(9.81),
      trajectory_frequency_hz_(0.35),
      circle_center_x_(1.0),
      circle_center_y_(0.2),
      circle_radius_(0.3),
      dcn_torque_gain_(0.0025),
      spike_retention_steps_(256),
      state_feedback_delay_steps_(1),
      state_feedback_repeats_(1),
      state_feedback_repeat_period_steps_(1),
      error_feedback_delay_steps_(0),
      error_feedback_window_steps_(1),
      error_feedback_sample_count_(50),
      cf_mix_position_(0.8),
      spike_cf_max_(15.0),
      angle_min_deg_{ -30.0, 0.1 },
      angle_max_deg_{ 90.0, 150.0 },
      velocity_min_deg_{ -400.0, -400.0 },
      velocity_max_deg_{ 400.0, 400.0 },
      cf_angle_norm_deg_{ 60.0, 75.0 },
      cf_velocity_norm_deg_{ 400.0, 400.0 },
      cf_positive_neurons_(),
      cf_negative_neurons_(),
      vertical_plane_(true),
      elbow_down_(true),
      enable_inverse_dynamics_(true),
      external_desired_state_enabled_(false),
      started_(false),
      error_feedback_rng_(17u) {}

PlanarArm2DOFPinocchio::~PlanarArm2DOFPinocchio() {}

void PlanarArm2DOFPinocchio::Initialize(const OuterDynamicDescription& description, Simulation* simulation) {
    // set update timestep
    this->setTimestepSize(std::max(1, description.update_timestep));
    // set communication interval
    this->setCommunicationInterval(std::max(1, description.communication_interval));
    // set queue index
    this->setQueueIndex(description.queue_index);

    this->link_lengths_ = GetVector2OrDefault(description.ModelParameter, "link_lengths", this->link_lengths_);
    this->link_masses_ = GetVector2OrDefault(description.ModelParameter, "link_masses", this->link_masses_);
    this->damping_ = GetVector2OrDefault(description.ModelParameter, "joint_damping", this->damping_);
    this->kp_ = GetVector2OrDefault(description.ModelParameter, "pd_kp", this->kp_);
    this->kd_ = GetVector2OrDefault(description.ModelParameter, "pd_kd", this->kd_);

    this->gravity_magnitude_ = GetAnyOrDefault<double>(description.ModelParameter, "gravity", this->gravity_magnitude_);
    this->trajectory_frequency_hz_ = GetAnyOrDefault<double>(description.ModelParameter, "trajectory_frequency_hz", this->trajectory_frequency_hz_);
    this->circle_center_x_ = GetAnyOrDefault<double>(description.ModelParameter, "circle_center_x", this->circle_center_x_);
    this->circle_center_y_ = GetAnyOrDefault<double>(description.ModelParameter, "circle_center_y", this->circle_center_y_);
    this->circle_radius_ = GetAnyOrDefault<double>(description.ModelParameter, "circle_radius", this->circle_radius_);
    this->dcn_torque_gain_ = GetAnyOrDefault<double>(description.ModelParameter, "dcn_torque_gain", this->dcn_torque_gain_);
    this->spike_retention_steps_ = GetAnyOrDefault<int>(description.ModelParameter, "spike_retention_steps", this->spike_retention_steps_);
    this->state_feedback_delay_steps_ = std::max(
        0,
        GetAnyOrDefault<int>(
            description.ModelParameter,
            "state_feedback_delay_steps",
            this->state_feedback_delay_steps_));
    this->state_feedback_repeats_ = std::max(
        1,
        GetAnyOrDefault<int>(
            description.ModelParameter,
            "state_feedback_repeats_per_update",
            this->state_feedback_repeats_));
    this->state_feedback_repeat_period_steps_ = std::max(
        1,
        GetAnyOrDefault<int>(
            description.ModelParameter,
            "state_feedback_repeat_period_steps",
            this->state_feedback_repeat_period_steps_));
    this->error_feedback_delay_steps_ = std::max(
        0,
        GetAnyOrDefault<int>(
            description.ModelParameter,
            "error_feedback_delay_steps",
            this->error_feedback_delay_steps_));
    this->error_feedback_window_steps_ = std::max(
        1,
        GetAnyOrDefault<int>(
            description.ModelParameter,
            "error_feedback_window_steps",
            std::max(1, description.communication_interval)));
    this->error_feedback_sample_count_ = std::max(
        1,
        GetAnyOrDefault<int>(
            description.ModelParameter,
            "error_feedback_sample_count",
            this->error_feedback_sample_count_));
    this->cf_mix_position_ = GetAnyOrDefault<double>(
        description.ModelParameter, "cf_mix_position", this->cf_mix_position_);
    this->spike_cf_max_ = GetAnyOrDefault<double>(
        description.ModelParameter, "spike_cf_max", this->spike_cf_max_);
    this->angle_min_deg_ = GetArray2OrDefault(
        description.ModelParameter, "angle_min_deg", this->angle_min_deg_);
    this->angle_max_deg_ = GetArray2OrDefault(
        description.ModelParameter, "angle_max_deg", this->angle_max_deg_);
    this->velocity_min_deg_ = GetArray2OrDefault(
        description.ModelParameter, "velocity_min_deg_s", this->velocity_min_deg_);
    this->velocity_max_deg_ = GetArray2OrDefault(
        description.ModelParameter, "velocity_max_deg_s", this->velocity_max_deg_);
    const std::array<double, 2> velocity_range_deg = GetArray2OrDefault(
        description.ModelParameter, "velocity_range_deg_s", { -1.0, -1.0 });
    for (int joint = 0; joint < 2; ++joint) {
        if (velocity_range_deg[static_cast<std::size_t>(joint)] > 0.0) {
            this->velocity_max_deg_[static_cast<std::size_t>(joint)] =
                this->velocity_min_deg_[static_cast<std::size_t>(joint)] +
                velocity_range_deg[static_cast<std::size_t>(joint)];
        }
    }
    this->cf_angle_norm_deg_ = GetArray2OrDefault(
        description.ModelParameter, "cf_angle_norm_deg", this->cf_angle_norm_deg_);
    this->cf_velocity_norm_deg_ = GetArray2OrDefault(
        description.ModelParameter, "cf_velocity_norm_deg_s", this->cf_velocity_norm_deg_);
    this->vertical_plane_ = GetAnyOrDefault<bool>(description.ModelParameter, "vertical_plane", this->vertical_plane_);
    this->elbow_down_ = GetAnyOrDefault<bool>(description.ModelParameter, "elbow_down", this->elbow_down_);
    this->enable_inverse_dynamics_ = GetAnyOrDefault<bool>(description.ModelParameter, "enable_inverse_dynamics", this->enable_inverse_dynamics_);
    this->error_feedback_rng_.seed(static_cast<std::mt19937::result_type>(
        GetAnyOrDefault<int>(description.ModelParameter, "error_feedback_seed", 17)));

    this->state_feedback_product_ = description.state_feedback_product;
    this->state_feedback_single_ = description.state_feedback_single;
    if (!description.cf_positive_neuron_indices_by_joint.empty() ||
        !description.cf_negative_neuron_indices_by_joint.empty()) {
        if (description.cf_positive_neuron_indices_by_joint.size() != 2 ||
            description.cf_negative_neuron_indices_by_joint.size() != 2) {
            throw std::runtime_error("PlanarArm2DOFPinocchio error feedback expects positive and negative CF targets for 2 joints.");
        }
        for (int joint = 0; joint < 2; ++joint) {
            this->cf_positive_neurons_[static_cast<std::size_t>(joint)] =
                description.cf_positive_neuron_indices_by_joint[static_cast<std::size_t>(joint)];
            this->cf_negative_neurons_[static_cast<std::size_t>(joint)] =
                description.cf_negative_neuron_indices_by_joint[static_cast<std::size_t>(joint)];
        }
    } else {
        for (int joint = 0; joint < 2; ++joint) {
            this->cf_positive_neurons_[static_cast<std::size_t>(joint)].clear();
            this->cf_negative_neurons_[static_cast<std::size_t>(joint)].clear();
        }
    }

    const bool has_dcn_mapping =
        !description.dcn_positive_neuron_indices_by_joint.empty() ||
        !description.dcn_negative_neuron_indices_by_joint.empty();
    if (has_dcn_mapping &&
        (description.dcn_positive_neuron_indices_by_joint.size() != 2 ||
         description.dcn_negative_neuron_indices_by_joint.size() != 2)) {
        throw std::runtime_error("PlanarArm2DOFPinocchio expects 2-joint DCN positive/negative mappings when DCN decoding is used.");
    }

    this->joint_mappings_.resize(2);
    std::vector<int> watched_dcn_neurons;
    for (int joint = 0; joint < 2; ++joint) {
        JointMapping& mapping = this->joint_mappings_[static_cast<std::size_t>(joint)];
        if (has_dcn_mapping) {
            mapping.dcn_positive_neurons = description.dcn_positive_neuron_indices_by_joint[static_cast<std::size_t>(joint)];
            mapping.dcn_negative_neurons = description.dcn_negative_neuron_indices_by_joint[static_cast<std::size_t>(joint)];
        } else {
            mapping.dcn_positive_neurons.clear();
            mapping.dcn_negative_neurons.clear();
        }

        watched_dcn_neurons.insert(watched_dcn_neurons.end(), mapping.dcn_positive_neurons.begin(), mapping.dcn_positive_neurons.end());
        watched_dcn_neurons.insert(watched_dcn_neurons.end(), mapping.dcn_negative_neurons.begin(), mapping.dcn_negative_neurons.end());
    }

    if (simulation->outer_dynamic_spike_buffer != NULL) {
        simulation->outer_dynamic_spike_buffer->RegisterWatchedNeurons(watched_dcn_neurons);
    }

    this->BuildPlanarModel();
    if (!this->external_desired_state_enabled_) {
        this->UpdateDesiredState(0.0);
    }
    this->q_ = this->q_des_;
    this->v_ = this->v_des_;
    this->qdd_ = this->qdd_des_;
    this->tau_total_ = this->tau_cereb_;
    this->started_ = false;
}

void PlanarArm2DOFPinocchio::Update(int time, Simulation* simulation) {
    if (!this->model_ready_) {
        throw std::runtime_error("PlanarArm2DOFPinocchio model not initialized.");
    }

    const double step_seconds = static_cast<double>(simulation->basetimesteps) * 0.001 * static_cast<double>(this->getTimestepSize());
    const double time_seconds = static_cast<double>(time) * static_cast<double>(simulation->basetimesteps) * 0.001;

    if (!this->external_desired_state_enabled_) {
        this->UpdateDesiredState(time_seconds);
    }
    if (!this->started_) {
        OuterDynamicJointState state;
        state.q = { this->q_(0), this->q_(1) };
        state.qv = { this->v_(0), this->v_(1) };
        state.qdd = { this->qdd_(0), this->qdd_(1) };
        state.q_des = { this->q_des_(0), this->q_des_(1) };
        state.qv_des = { this->v_des_(0), this->v_des_(1) };
        state.tau_total = { 0.0, 0.0 };
        simulation->WriteOuterDynamicState(time, state, this);
        this->EncodeAndEmitStateFeedback(time, simulation);
        this->started_ = true;
        return;
    }
    this->DecodeDcnTorques(time, step_seconds, simulation);

    this->tau_total_ = this->tau_cereb_; //this->tau_pd_ + this->tau_cereb_;
    //if (this->enable_inverse_dynamics_) {
        //this->tau_total_ += this->tau_id_;
    //}

    this->IntegrateDynamics(step_seconds);
    OuterDynamicJointState state;
    state.q = { this->q_(0), this->q_(1) };
    state.qv = { this->v_(0), this->v_(1) };
    state.qdd = { this->qdd_(0), this->qdd_(1) };
    state.q_des = { this->q_des_(0), this->q_des_(1) };
    state.qv_des = { this->v_des_(0), this->v_des_(1) };
    state.tau_total = { this->tau_total_(0), this->tau_total_(1) };
    simulation->WriteOuterDynamicState(time, state, this);
    this->EncodeAndEmitStateFeedback(time, simulation);
    this->EncodeAndEmitErrorFeedback(time, simulation);

    if (simulation->outer_dynamic_spike_buffer != NULL) {
        const int discard_before = std::max(0, time - this->spike_retention_steps_);
        simulation->outer_dynamic_spike_buffer->DiscardOlderThan(discard_before);
    }
}

void PlanarArm2DOFPinocchio::BuildPlanarModel() {
    pinocchio::Model temp_model;
    if (this->vertical_plane_) {
        temp_model.gravity.linear() = Eigen::Vector3d(0.0, -this->gravity_magnitude_, 0.0);
    } else {
        temp_model.gravity.linear() = Eigen::Vector3d::Zero();
    }

    const double inertia1 = this->link_masses_(0) * this->link_lengths_(0) * this->link_lengths_(0) / 12.0;
    const double inertia2 = this->link_masses_(1) * this->link_lengths_(1) * this->link_lengths_(1) / 12.0;

    const pinocchio::JointIndex joint1 =
        temp_model.addJoint(
            0,
            pinocchio::JointModelRZ(),
            pinocchio::SE3::Identity(),
            "joint1");

    Eigen::Matrix3d link1_inertia_matrix = Eigen::Matrix3d::Zero();
    link1_inertia_matrix(0, 0) = 1.0e-6;
    link1_inertia_matrix(1, 1) = inertia1;
    link1_inertia_matrix(2, 2) = inertia1;

    temp_model.appendBodyToJoint(
        joint1,
        pinocchio::Inertia(
            this->link_masses_(0),
            Eigen::Vector3d(this->link_lengths_(0) * 0.5, 0.0, 0.0),
            link1_inertia_matrix),
        pinocchio::SE3::Identity());

    const pinocchio::SE3 joint2_placement(
        Eigen::Matrix3d::Identity(),
        Eigen::Vector3d(this->link_lengths_(0), 0.0, 0.0));

    const pinocchio::JointIndex joint2 =
        temp_model.addJoint(
            joint1,
            pinocchio::JointModelRZ(),
            joint2_placement,
            "joint2");

    Eigen::Matrix3d link2_inertia_matrix = Eigen::Matrix3d::Zero();
    link2_inertia_matrix(0, 0) = 1.0e-6;
    link2_inertia_matrix(1, 1) = inertia2;
    link2_inertia_matrix(2, 2) = inertia2;

    temp_model.appendBodyToJoint(
        joint2,
        pinocchio::Inertia(
            this->link_masses_(1),
            Eigen::Vector3d(this->link_lengths_(1) * 0.5, 0.0, 0.0),
            link2_inertia_matrix),
        pinocchio::SE3::Identity());

    temp_model.addFrame(
        pinocchio::Frame(
            "end_effector",
            joint2,
            joint2,
            pinocchio::SE3(
                Eigen::Matrix3d::Identity(),
                Eigen::Vector3d(this->link_lengths_(1), 0.0, 0.0)),
            pinocchio::OP_FRAME));

    this->model_ = std::move(temp_model);
    this->data_aba_ = pinocchio::Data(this->model_);
    this->data_rnea_ = pinocchio::Data(this->model_);
    this->data_fk_ = pinocchio::Data(this->model_);
    this->end_effector_frame_id_ = this->model_.getFrameId("end_effector");
    this->model_ready_ = true;
}

void PlanarArm2DOFPinocchio::UpdateDesiredState(double time_seconds) {
    const double omega = 2.0 * kPi * this->trajectory_frequency_hz_;

    const double xd = this->circle_center_x_ + this->circle_radius_ * std::cos(omega * time_seconds);
    const double yd = this->circle_center_y_ + this->circle_radius_ * std::sin(omega * time_seconds);
    const double xd_dot = -this->circle_radius_ * omega * std::sin(omega * time_seconds);
    const double yd_dot = this->circle_radius_ * omega * std::cos(omega * time_seconds);
    const double xd_ddot = -this->circle_radius_ * omega * omega * std::cos(omega * time_seconds);
    const double yd_ddot = -this->circle_radius_ * omega * omega * std::sin(omega * time_seconds);

    const IKResult ik_result = this->InverseKinematics2R(xd, yd);
    if (!ik_result.reachable) {
        throw std::runtime_error("Desired circular trajectory is unreachable for PlanarArm2DOFPinocchio.");
    }

    this->q_des_ = ik_result.q;

    const Eigen::Vector2d xdot_des(xd_dot, yd_dot);
    const Eigen::Vector2d xddot_des(xd_ddot, yd_ddot);
    const Eigen::Matrix2d jacobian = this->Jacobian2R(this->q_des_);

    this->v_des_ = this->DampedLeastSquaresSolve(jacobian, xdot_des);
    this->qdd_des_ = this->DampedLeastSquaresSolve(
        jacobian,
        xddot_des - this->JacobianDot2R(this->q_des_, this->v_des_) * this->v_des_);
    /*
    if (this->enable_inverse_dynamics_) {
        this->tau_id_ = pinocchio::rnea(
            this->model_,
            this->data_rnea_,
            this->q_des_,
            this->v_des_,
            this->qdd_des_);
    } else {
        this->tau_id_.setZero();
    }
    */
}

void PlanarArm2DOFPinocchio::DecodeDcnTorques(int time, double step_seconds, Simulation* simulation) {
    this->tau_cereb_.setZero();
    this->tau_cereb_positive_.setZero();
    this->tau_cereb_negative_.setZero();
    if (simulation->outer_dynamic_spike_buffer == NULL) {
        this->tau_cereb_ += this->accumulated_joint_torque_;
        return;
    }

    const int window_steps = std::max(1, this->getCommunicationInterval());
    const int window_start = std::max(0, time - window_steps);
    const double window_seconds = std::max(step_seconds, static_cast<double>(window_steps) * static_cast<double>(simulation->basetimesteps) * 0.001);

    for (int joint = 0; joint < 2; ++joint) {
        const JointMapping& mapping = this->joint_mappings_[joint];
        const int pos_count = simulation->outer_dynamic_spike_buffer->CountSpikesInWindow(window_start, time, mapping.dcn_positive_neurons);
        const int neg_count = simulation->outer_dynamic_spike_buffer->CountSpikesInWindow(window_start, time, mapping.dcn_negative_neurons);
        const double positive_rate = static_cast<double>(pos_count) / window_seconds;
        const double negative_rate = static_cast<double>(neg_count) / window_seconds;
        this->tau_cereb_positive_(joint) = this->dcn_torque_gain_ * positive_rate;
        this->tau_cereb_negative_(joint) = this->dcn_torque_gain_ * negative_rate;
        this->tau_cereb_(joint) = this->tau_cereb_positive_(joint) - this->tau_cereb_negative_(joint);
    }
    this->tau_cereb_ += this->accumulated_joint_torque_;
}

void PlanarArm2DOFPinocchio::AccumulateInputSpike(int joint_id, int type, float weight, int time) {
    (void)time;
    if (joint_id < 0 || joint_id >= 2) {
        return;
    }
    if (type == 0) {
        this->accumulated_joint_torque_(joint_id) += weight;
    } else if (type == 1) {
        this->accumulated_joint_torque_(joint_id) -= weight;
    }
}

void PlanarArm2DOFPinocchio::ClearAccumulatedInputs() {
    this->accumulated_joint_torque_.setZero();
}

void PlanarArm2DOFPinocchio::ResetState(
    const std::array<double, 2>& q,
    const std::array<double, 2>& qd) {
    this->q_ = Eigen::Vector2d(q[0], q[1]);
    this->v_ = Eigen::Vector2d(qd[0], qd[1]);
    this->qdd_.setZero();
    this->tau_total_.setZero();
    this->accumulated_joint_torque_.setZero();
    // Match StrictMatlab semantics: the next update publishes the reset state
    // and emits state feedback before integrating the plant.
    this->started_ = false;
}

void PlanarArm2DOFPinocchio::SetDesiredState(
    const std::array<double, 2>& q_des,
    const std::array<double, 2>& qd_des) {
    this->q_des_ = Eigen::Vector2d(q_des[0], q_des[1]);
    this->v_des_ = Eigen::Vector2d(qd_des[0], qd_des[1]);
    this->qdd_des_.setZero();
    // Once the user supplies a desired state explicitly, keep using that
    // externally driven trajectory instead of regenerating the built-in circle.
    this->external_desired_state_enabled_ = true;
}

void PlanarArm2DOFPinocchio::IntegrateDynamics(double step_seconds) {
    const Eigen::Vector2d tau_input = this->tau_total_ - this->damping_.cwiseProduct(this->v_);
    this->qdd_ = pinocchio::aba(this->model_, this->data_aba_, this->q_, this->v_, tau_input);
    this->v_ += this->qdd_ * step_seconds;
    this->q_ += this->v_ * step_seconds;
}

void PlanarArm2DOFPinocchio::EncodeAndEmitStateFeedback(int time, Simulation* simulation) {
    if (simulation == NULL) {
        return;
    }
    const int emit_time = time + this->state_feedback_delay_steps_;

    if (!this->state_feedback_product_.neuron_indices_by_joint.empty()) {
        if (this->state_feedback_product_.neuron_indices_by_joint.size() != 2 ||
            this->state_feedback_product_.bins.size() != 4) {
            throw std::runtime_error("PlanarArm2DOFPinocchio state_feedback_product expects 2 joints and 4 state bins.");
        }
        for (int joint = 0; joint < 2; ++joint) {
            const int product_size =
                this->state_feedback_product_.bins[0] *
                this->state_feedback_product_.bins[1] *
                this->state_feedback_product_.bins[2] *
                this->state_feedback_product_.bins[3];
            const std::vector<int>& targets =
                this->state_feedback_product_.neuron_indices_by_joint[static_cast<std::size_t>(joint)];
            if (static_cast<int>(targets.size()) != product_size) {
                throw std::runtime_error("PlanarArm2DOFPinocchio state_feedback_product target size does not match product bins.");
            }
            const int index = this->EncodeStateProductIndex(joint);
            for (int repeat = 0; repeat < this->state_feedback_repeats_; ++repeat) {
                this->EmitSpike(
                    simulation,
                    targets[static_cast<std::size_t>(index)],
                    emit_time + repeat * this->state_feedback_repeat_period_steps_);
            }
        }
    }

    if (!this->state_feedback_single_.neuron_indices_by_joint_variable.empty()) {
        if (this->state_feedback_single_.neuron_indices_by_joint_variable.size() != 2 ||
            this->state_feedback_single_.bins.size() != 4) {
            throw std::runtime_error("PlanarArm2DOFPinocchio state_feedback_single expects 2 joints and 4 state variables.");
        }
        for (int joint = 0; joint < 2; ++joint) {
            const std::vector<double> values = {
                RadToDeg(this->q_des_(joint)),
                RadToDeg(this->q_(joint)),
                RadToDeg(this->v_des_(joint)),
                RadToDeg(this->v_(joint))
            };
            const std::vector<double> mins = {
                this->angle_min_deg_[static_cast<std::size_t>(joint)],
                this->angle_min_deg_[static_cast<std::size_t>(joint)],
                this->velocity_min_deg_[static_cast<std::size_t>(joint)],
                this->velocity_min_deg_[static_cast<std::size_t>(joint)]
            };
            const std::vector<double> maxs = {
                this->angle_max_deg_[static_cast<std::size_t>(joint)],
                this->angle_max_deg_[static_cast<std::size_t>(joint)],
                this->velocity_max_deg_[static_cast<std::size_t>(joint)],
                this->velocity_max_deg_[static_cast<std::size_t>(joint)]
            };
            const std::vector<std::vector<int> >& joint_targets =
                this->state_feedback_single_.neuron_indices_by_joint_variable[static_cast<std::size_t>(joint)];
            if (joint_targets.size() != 4) {
                throw std::runtime_error("PlanarArm2DOFPinocchio state_feedback_single target variable count must be 4.");
            }
            for (int variable = 0; variable < 4; ++variable) {
                const int bin_count = this->state_feedback_single_.bins[static_cast<std::size_t>(variable)];
                if (bin_count <= 0 ||
                    static_cast<int>(joint_targets[static_cast<std::size_t>(variable)].size()) != bin_count) {
                    throw std::runtime_error("PlanarArm2DOFPinocchio state_feedback_single target bin count is invalid.");
                }
                const int bin = this->EncodeUniformBin(
                    values[static_cast<std::size_t>(variable)],
                    mins[static_cast<std::size_t>(variable)],
                    maxs[static_cast<std::size_t>(variable)],
                    bin_count);
                for (int repeat = 0; repeat < this->state_feedback_repeats_; ++repeat) {
                    this->EmitSpike(
                        simulation,
                        joint_targets[static_cast<std::size_t>(variable)][static_cast<std::size_t>(bin)],
                        emit_time + repeat * this->state_feedback_repeat_period_steps_);
                }
            }
        }
    }
}

void PlanarArm2DOFPinocchio::EncodeAndEmitErrorFeedback(int time, Simulation* simulation) {
    if (simulation == NULL) {
        return;
    }
    if (this->cf_positive_neurons_[0].empty() &&
        this->cf_negative_neurons_[0].empty() &&
        this->cf_positive_neurons_[1].empty() &&
        this->cf_negative_neurons_[1].empty()) {
        return;
    }
    for (int joint = 0; joint < 2; ++joint) {
        if (this->cf_positive_neurons_[static_cast<std::size_t>(joint)].empty() ||
            this->cf_negative_neurons_[static_cast<std::size_t>(joint)].empty() ||
            this->cf_positive_neurons_[static_cast<std::size_t>(joint)].size() !=
                this->cf_negative_neurons_[static_cast<std::size_t>(joint)].size()) {
            throw std::runtime_error("PlanarArm2DOFPinocchio error feedback requires matched positive/negative CF targets per joint.");
        }
    }

    for (int joint = 0; joint < 2; ++joint) {
        const double position_error = RadToDeg(this->q_des_(joint)) - RadToDeg(this->q_(joint));
        const double velocity_error = RadToDeg(this->v_des_(joint)) - RadToDeg(this->v_(joint));
        double normalized_angle =
            LinearSaturating(this->cf_angle_norm_deg_[static_cast<std::size_t>(joint)], std::fabs(position_error));
        double normalized_velocity =
            LinearSaturating(this->cf_velocity_norm_deg_[static_cast<std::size_t>(joint)], std::fabs(velocity_error));
        if (position_error < 0.0) {
            normalized_angle = -normalized_angle;
        }
        if (velocity_error < 0.0) {
            normalized_velocity = -normalized_velocity;
        }
        const double mixed_error =
            this->cf_mix_position_ * normalized_angle +
            (1.0 - this->cf_mix_position_) * normalized_velocity;
        const int cf_count = static_cast<int>(
            std::llround(
                std::fabs(
                    this->spike_cf_max_ *
                    static_cast<double>(this->cf_positive_neurons_[static_cast<std::size_t>(joint)].size()) /
                    static_cast<double>(this->error_feedback_sample_count_) *
                    mixed_error)));
        const std::vector<int>& targets =
            mixed_error > 0.0 ?
                this->cf_positive_neurons_[static_cast<std::size_t>(joint)] :
                this->cf_negative_neurons_[static_cast<std::size_t>(joint)];
        this->EmitBinaryCfSpikes(
            cf_count,
            targets,
            time + this->error_feedback_delay_steps_,
            this->error_feedback_window_steps_,
            simulation);
    }
}

void PlanarArm2DOFPinocchio::EmitBinaryCfSpikes(
    int seed,
    const std::vector<int>& target_neurons,
    int base_time,
    int window_steps,
    Simulation* simulation) {
    if (simulation == NULL || seed <= 0 || target_neurons.empty() || window_steps <= 0) {
        return;
    }
    std::vector<unsigned char> bitmap(
        target_neurons.size() * static_cast<std::size_t>(window_steps),
        0u);
    std::uniform_int_distribution<int> neuron_dist(0, static_cast<int>(target_neurons.size()) - 1);
    std::uniform_int_distribution<int> time_dist(0, window_steps - 1);
    for (int event = 0; event < seed; ++event) {
        const int neuron = neuron_dist(this->error_feedback_rng_);
        const int local_time = time_dist(this->error_feedback_rng_);
        bitmap[static_cast<std::size_t>(neuron) * static_cast<std::size_t>(window_steps) +
               static_cast<std::size_t>(local_time)] = 1u;
    }
    for (std::size_t neuron = 0; neuron < target_neurons.size(); ++neuron) {
        for (int local_time = 0; local_time < window_steps; ++local_time) {
            if (bitmap[neuron * static_cast<std::size_t>(window_steps) +
                       static_cast<std::size_t>(local_time)] == 0u) {
                continue;
            }
            // Emit through the regular InputSpikeNeuronModel path so feedback
            // participates in the same event semantics as StrictMatlab.
            this->EmitSpike(simulation, target_neurons[neuron], base_time + 1 + local_time);
        }
    }
}

void PlanarArm2DOFPinocchio::EmitSpike(Simulation* simulation, int neuron_id, int time) {
    if (simulation == NULL || simulation->network == NULL) {
        return;
    }
    if (neuron_id < 0 || neuron_id >= simulation->network->neuronsNum) {
        return;
    }
    simulation->QueueOuterDynamicInputSpike(time, neuron_id);
}

PlanarArm2DOFPinocchio::IKResult PlanarArm2DOFPinocchio::InverseKinematics2R(double x, double y) const {
    IKResult result;
    result.q.setZero();
    result.reachable = true;

    const double l1 = this->link_lengths_(0);
    const double l2 = this->link_lengths_(1);
    const double radius_squared = x * x + y * y;
    const double c2_raw = (radius_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
    if (c2_raw < -1.0 || c2_raw > 1.0) {
        result.reachable = false;
        return result;
    }

    const double c2 = std::clamp(c2_raw, -1.0, 1.0);
    double s2 = std::sqrt(std::max(0.0, 1.0 - c2 * c2));
    if (!this->elbow_down_) {
        s2 = -s2;
    }

    const double q2 = std::atan2(s2, c2);
    const double k1 = l1 + l2 * c2;
    const double k2 = l2 * s2;
    const double q1 = std::atan2(y, x) - std::atan2(k2, k1);

    result.q << q1, q2;
    return result;
}

Eigen::Matrix2d PlanarArm2DOFPinocchio::Jacobian2R(const Eigen::Vector2d& q) const {
    const double q1 = q(0);
    const double q2 = q(1);
    const double s1 = std::sin(q1);
    const double c1 = std::cos(q1);
    const double s12 = std::sin(q1 + q2);
    const double c12 = std::cos(q1 + q2);

    Eigen::Matrix2d jacobian;
    jacobian(0, 0) = -this->link_lengths_(0) * s1 - this->link_lengths_(1) * s12;
    jacobian(0, 1) = -this->link_lengths_(1) * s12;
    jacobian(1, 0) = this->link_lengths_(0) * c1 + this->link_lengths_(1) * c12;
    jacobian(1, 1) = this->link_lengths_(1) * c12;
    return jacobian;
}

Eigen::Matrix2d PlanarArm2DOFPinocchio::JacobianDot2R(const Eigen::Vector2d& q, const Eigen::Vector2d& qdot) const {
    const double q1 = q(0);
    const double q2 = q(1);
    const double q1dot = qdot(0);
    const double q2dot = qdot(1);
    const double q12 = q1 + q2;
    const double q12dot = q1dot + q2dot;
    const double s1 = std::sin(q1);
    const double c1 = std::cos(q1);
    const double s12 = std::sin(q12);
    const double c12 = std::cos(q12);

    Eigen::Matrix2d jacobian_dot;
    jacobian_dot(0, 0) = -this->link_lengths_(0) * c1 * q1dot - this->link_lengths_(1) * c12 * q12dot;
    jacobian_dot(0, 1) = -this->link_lengths_(1) * c12 * q12dot;
    jacobian_dot(1, 0) = -this->link_lengths_(0) * s1 * q1dot - this->link_lengths_(1) * s12 * q12dot;
    jacobian_dot(1, 1) = -this->link_lengths_(1) * s12 * q12dot;
    return jacobian_dot;
}

Eigen::Vector2d PlanarArm2DOFPinocchio::DampedLeastSquaresSolve(const Eigen::Matrix2d& jacobian, const Eigen::Vector2d& rhs) const {
    const double lambda = 1.0e-8;
    const Eigen::Matrix2d matrix =
        jacobian * jacobian.transpose() +
        lambda * lambda * Eigen::Matrix2d::Identity();
    return jacobian.transpose() * matrix.ldlt().solve(rhs);
}

int PlanarArm2DOFPinocchio::EncodeStateProductIndex(int joint) const {
    const std::vector<int>& bins = this->state_feedback_product_.bins;
    const std::vector<int> indices = {
        this->EncodeUniformBin(
            RadToDeg(this->q_des_(joint)),
            this->angle_min_deg_[static_cast<std::size_t>(joint)],
            this->angle_max_deg_[static_cast<std::size_t>(joint)],
            bins[0]),
        this->EncodeUniformBin(
            RadToDeg(this->q_(joint)),
            this->angle_min_deg_[static_cast<std::size_t>(joint)],
            this->angle_max_deg_[static_cast<std::size_t>(joint)],
            bins[1]),
        this->EncodeUniformBin(
            RadToDeg(this->v_des_(joint)),
            this->velocity_min_deg_[static_cast<std::size_t>(joint)],
            this->velocity_max_deg_[static_cast<std::size_t>(joint)],
            bins[2]),
        this->EncodeUniformBin(
            RadToDeg(this->v_(joint)),
            this->velocity_min_deg_[static_cast<std::size_t>(joint)],
            this->velocity_max_deg_[static_cast<std::size_t>(joint)],
            bins[3])
    };
    return this->EncodeProductIndex(indices, bins);
}

int PlanarArm2DOFPinocchio::EncodeUniformBin(
    double value,
    double min_value,
    double max_value,
    int bin_count) const {
    if (bin_count <= 1 || max_value <= min_value) {
        return 0;
    }
    const double normalized = (value - min_value) / (max_value - min_value);
    const double clipped = std::max(0.0, std::min(1.0, normalized));
    int index = static_cast<int>(std::floor(clipped * static_cast<double>(bin_count)));
    if (index >= bin_count) {
        index = bin_count - 1;
    }
    return std::max(0, index);
}

int PlanarArm2DOFPinocchio::EncodeProductIndex(
    const std::vector<int>& indices,
    const std::vector<int>& bins) const {
    int stride = 1;
    int index = 0;
    for (std::size_t variable = 0; variable < indices.size(); ++variable) {
        index += indices[variable] * stride;
        stride *= bins[variable];
    }
    return index;
}

double PlanarArm2DOFPinocchio::RadToDeg(double radians) {
    return radians * 180.0 / kPi;
}

double PlanarArm2DOFPinocchio::LinearSaturating(double max_abs, double value) {
    if (max_abs <= 1.0e-9) {
        return 0.0;
    }
    return std::max(0.0, std::min(1.0, value / max_abs));
}
