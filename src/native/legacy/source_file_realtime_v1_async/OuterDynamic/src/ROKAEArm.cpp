#include "../source_file_realtime_v1_async/OuterDynamic/inc/ROKAEArm.h"

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeBuffer.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"

#include <boost/any.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace {

template <typename T>
T GetOrDefault(const std::map<std::string, boost::any>& params,
               const std::string& key,
               const T& default_value) {
    const auto found = params.find(key);
    if (found == params.end()) {
        return default_value;
    }
    try {
        return boost::any_cast<T>(found->second);
    } catch (const boost::bad_any_cast&) {
        throw std::runtime_error("ROKAE_Arm parameter has an invalid native type: " + key);
    }
}

Eigen::VectorXd GetVector(const std::map<std::string, boost::any>& params,
                          const std::string& key,
                          const Eigen::VectorXd& default_value) {
    const auto found = params.find(key);
    if (found == params.end()) {
        return default_value;
    }
    try {
        const std::vector<double> values = boost::any_cast<std::vector<double>>(found->second);
        if (values.size() != static_cast<std::size_t>(default_value.size())) {
            throw std::runtime_error(
                "ROKAE_Arm parameter '" + key + "' must contain " +
                std::to_string(default_value.size()) + " values.");
        }
        return Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
    } catch (const boost::bad_any_cast&) {
        throw std::runtime_error("ROKAE_Arm parameter must be float64_list: " + key);
    }
}

std::vector<double> ToVector(const Eigen::VectorXd& values) {
    return std::vector<double>(values.data(), values.data() + values.size());
}

constexpr double kPi = 3.14159265358979323846;

}  // namespace

ROKAEArm::ROKAEArm(int timestep_size)
    : OuterDynamicModel(timestep_size),
      model_(),
      data_(model_),
      model_ready_(false),
      joint_count_(0),
      end_effector_frame_("xMateSR3C_link6"),
      trajectory_mode_("smooth_step"),
      trajectory_start_s_(0.0),
      motion_duration_s_(5.0),
      dcn_torque_gain_(0.0025),
      spike_retention_steps_(256),
      enable_pd_control_(true),
      external_desired_state_enabled_(false),
      started_(false) {}

ROKAEArm::~ROKAEArm() {}

void ROKAEArm::Initialize(const OuterDynamicDescription& description, Simulation* simulation) {
    if (simulation == nullptr) {
        throw std::runtime_error("ROKAE_Arm requires a valid Simulation.");
    }
    setTimestepSize(std::max(1, description.update_timestep));
    setCommunicationInterval(std::max(1, description.communication_interval));
    setQueueIndex(description.queue_index);

    const std::string urdf_path = GetOrDefault<std::string>(description.ModelParameter, "urdf_path", "");
    if (urdf_path.empty()) {
        throw std::runtime_error("ROKAE_Arm requires the 'urdf_path' parameter.");
    }
    if (!std::filesystem::is_regular_file(std::filesystem::path(urdf_path))) {
        throw std::runtime_error("ROKAE_Arm URDF file does not exist: " + urdf_path);
    }
    end_effector_frame_ = GetOrDefault<std::string>(
        description.ModelParameter, "end_effector_frame", end_effector_frame_);

    pinocchio::Model loaded_model;
    pinocchio::urdf::buildModel(urdf_path, loaded_model);
    if (loaded_model.nq != loaded_model.nv) {
        throw std::runtime_error("ROKAE_Arm currently requires nq == nv.");
    }
    if (loaded_model.nv != 6) {
        throw std::runtime_error(
            "ROKAE_Arm expects the six-axis xMate model, but the URDF has " +
            std::to_string(loaded_model.nv) + " velocity DOFs.");
    }
    if (!loaded_model.existFrame(end_effector_frame_)) {
        throw std::runtime_error("ROKAE_Arm end-effector frame was not found: " + end_effector_frame_);
    }

    model_ = std::move(loaded_model);
    data_ = pinocchio::Data(model_);
    joint_count_ = static_cast<int>(model_.nv);
    model_ready_ = true;

    q_ = pinocchio::neutral(model_);
    v_ = Eigen::VectorXd::Zero(model_.nv);
    qdd_ = Eigen::VectorXd::Zero(model_.nv);
    q_start_ = q_;
    q_goal_ = q_;
    const double default_goal_deg[] = {0.0, 45.0, -135.0, 0.0, 90.0, 180.0};
    for (int joint = 0; joint < joint_count_; ++joint) {
        q_goal_(joint) = default_goal_deg[joint] * kPi / 180.0;
    }

    q_start_ = GetVector(description.ModelParameter, "initial_q", q_start_);
    q_goal_ = GetVector(description.ModelParameter, "goal_q", q_goal_);
    q_ = q_start_;
    q_des_ = q_start_;
    v_des_ = Eigen::VectorXd::Zero(model_.nv);
    kp_ = GetVector(description.ModelParameter, "pd_kp", Eigen::VectorXd::Constant(model_.nv, 60.0));
    kd_ = GetVector(description.ModelParameter, "pd_kd", Eigen::VectorXd::Constant(model_.nv, 12.0));
    damping_ = GetVector(description.ModelParameter, "joint_damping", Eigen::VectorXd::Constant(model_.nv, 0.1));
    accumulated_torque_ = Eigen::VectorXd::Zero(model_.nv);
    tau_total_ = Eigen::VectorXd::Zero(model_.nv);
    torque_limit_ = GetVector(description.ModelParameter, "torque_limit", model_.effortLimit);

    trajectory_mode_ = GetOrDefault<std::string>(description.ModelParameter, "trajectory_mode", trajectory_mode_);
    if (trajectory_mode_ != "hold" && trajectory_mode_ != "smooth_step") {
        throw std::runtime_error("ROKAE_Arm trajectory_mode must be 'hold' or 'smooth_step'.");
    }
    trajectory_start_s_ = GetOrDefault<double>(description.ModelParameter, "trajectory_start_s", trajectory_start_s_);
    motion_duration_s_ = GetOrDefault<double>(description.ModelParameter, "motion_duration_s", motion_duration_s_);
    if (motion_duration_s_ <= 0.0) {
        throw std::runtime_error("ROKAE_Arm motion_duration_s must be positive.");
    }
    dcn_torque_gain_ = GetOrDefault<double>(description.ModelParameter, "dcn_torque_gain", dcn_torque_gain_);
    spike_retention_steps_ = std::max(
        1, GetOrDefault<int>(description.ModelParameter, "spike_retention_steps", spike_retention_steps_));
    enable_pd_control_ = GetOrDefault<bool>(description.ModelParameter, "enable_pd_control", enable_pd_control_);

    ClampState();
    q_start_ = q_;
    q_des_ = q_;
    for (int joint = 0; joint < joint_count_; ++joint) {
        q_goal_(joint) = std::clamp(q_goal_(joint), model_.lowerPositionLimit(joint), model_.upperPositionLimit(joint));
    }

    const bool has_dcn_mapping =
        !description.dcn_positive_neuron_indices_by_joint.empty() ||
        !description.dcn_negative_neuron_indices_by_joint.empty();
    if (has_dcn_mapping &&
        (description.dcn_positive_neuron_indices_by_joint.size() != static_cast<std::size_t>(joint_count_) ||
         description.dcn_negative_neuron_indices_by_joint.size() != static_cast<std::size_t>(joint_count_))) {
        throw std::runtime_error("ROKAE_Arm DCN positive/negative mappings must each contain 6 joints.");
    }
    joint_mappings_.assign(static_cast<std::size_t>(joint_count_), JointMapping());
    std::vector<int> watched_neurons;
    if (has_dcn_mapping) {
        for (int joint = 0; joint < joint_count_; ++joint) {
            JointMapping& mapping = joint_mappings_[static_cast<std::size_t>(joint)];
            mapping.positive_neurons = description.dcn_positive_neuron_indices_by_joint[static_cast<std::size_t>(joint)];
            mapping.negative_neurons = description.dcn_negative_neuron_indices_by_joint[static_cast<std::size_t>(joint)];
            watched_neurons.insert(watched_neurons.end(), mapping.positive_neurons.begin(), mapping.positive_neurons.end());
            watched_neurons.insert(watched_neurons.end(), mapping.negative_neurons.begin(), mapping.negative_neurons.end());
        }
    }
    if (simulation->outer_dynamic_spike_buffer != nullptr) {
        simulation->outer_dynamic_spike_buffer->RegisterWatchedNeurons(watched_neurons);
    }
    if (!description.state_feedback_product.neuron_indices_by_joint.empty() ||
        !description.state_feedback_single.neuron_indices_by_joint_variable.empty() ||
        !description.error_feedback_product.neuron_indices_by_joint.empty() ||
        !description.error_feedback_single.neuron_indices_by_joint_variable.empty()) {
        throw std::runtime_error("ROKAE_Arm neural feedback encoding is not configured; read state through outer_dynamic_state(s). ");
    }
    external_desired_state_enabled_ = false;
    started_ = false;
}

void ROKAEArm::Update(int time, Simulation* simulation) {
    if (!model_ready_) {
        throw std::runtime_error("ROKAE_Arm model is not initialized.");
    }
    const double base_step_s = static_cast<double>(simulation->basetimesteps) * 0.001;
    const double step_seconds = base_step_s * static_cast<double>(getTimestepSize());
    const double time_seconds = static_cast<double>(time) * base_step_s;
    if (!external_desired_state_enabled_) {
        UpdateDesiredState(time_seconds);
    }
    if (!started_) {
        WriteState(time, simulation);
        started_ = true;
        return;
    }

    Eigen::VectorXd neural_torque = Eigen::VectorXd::Zero(joint_count_);
    DecodeDcnTorque(time, step_seconds, simulation, neural_torque);
    tau_total_ = neural_torque;
    if (enable_pd_control_) {
        tau_total_ += kp_.cwiseProduct(q_des_ - q_) + kd_.cwiseProduct(v_des_ - v_);
    }
    for (int joint = 0; joint < joint_count_; ++joint) {
        const double limit = torque_limit_(joint);
        if (std::isfinite(limit) && limit > 0.0) {
            tau_total_(joint) = std::clamp(tau_total_(joint), -limit, limit);
        }
    }
    Integrate(step_seconds);
    WriteState(time, simulation);
    if (simulation->outer_dynamic_spike_buffer != nullptr) {
        simulation->outer_dynamic_spike_buffer->DiscardOlderThan(std::max(0, time - spike_retention_steps_));
    }
}

void ROKAEArm::UpdateDesiredState(double time_seconds) {
    if (trajectory_mode_ == "hold") {
        q_des_ = q_start_;
        v_des_.setZero();
        return;
    }
    const double raw = (time_seconds - trajectory_start_s_) / motion_duration_s_;
    const double t = std::clamp(raw, 0.0, 1.0);
    const double blend = t * t * (3.0 - 2.0 * t);
    const double blend_rate = (raw > 0.0 && raw < 1.0)
        ? 6.0 * t * (1.0 - t) / motion_duration_s_
        : 0.0;
    q_des_ = q_start_ + blend * (q_goal_ - q_start_);
    v_des_ = blend_rate * (q_goal_ - q_start_);
}

void ROKAEArm::DecodeDcnTorque(int time,
                              double step_seconds,
                              Simulation* simulation,
                              Eigen::VectorXd& torque) {
    torque = accumulated_torque_;
    if (simulation->outer_dynamic_spike_buffer == nullptr) {
        return;
    }
    const int window_steps = std::max(1, getCommunicationInterval());
    const int window_start = std::max(0, time - window_steps);
    const double window_s = std::max(
        step_seconds,
        static_cast<double>(window_steps) * static_cast<double>(simulation->basetimesteps) * 0.001);
    for (int joint = 0; joint < joint_count_; ++joint) {
        const JointMapping& mapping = joint_mappings_[static_cast<std::size_t>(joint)];
        const int positive = simulation->outer_dynamic_spike_buffer->CountSpikesInWindow(
            window_start, time, mapping.positive_neurons);
        const int negative = simulation->outer_dynamic_spike_buffer->CountSpikesInWindow(
            window_start, time, mapping.negative_neurons);
        torque(joint) += dcn_torque_gain_ * static_cast<double>(positive - negative) / window_s;
    }
}

void ROKAEArm::Integrate(double step_seconds) {
    const Eigen::VectorXd applied = tau_total_ - damping_.cwiseProduct(v_);
    qdd_ = pinocchio::aba(model_, data_, q_, v_, applied);
    v_ += qdd_ * step_seconds;
    q_ += v_ * step_seconds;
    ClampState();
}

void ROKAEArm::ClampState() {
    for (int joint = 0; joint < joint_count_; ++joint) {
        const double lower = model_.lowerPositionLimit(joint);
        const double upper = model_.upperPositionLimit(joint);
        if (std::isfinite(lower) && q_(joint) < lower) {
            q_(joint) = lower;
            v_(joint) = 0.0;
        } else if (std::isfinite(upper) && q_(joint) > upper) {
            q_(joint) = upper;
            v_(joint) = 0.0;
        }
        const double velocity_limit = model_.velocityLimit(joint);
        if (std::isfinite(velocity_limit) && velocity_limit > 0.0) {
            v_(joint) = std::clamp(v_(joint), -velocity_limit, velocity_limit);
        }
    }
}

void ROKAEArm::WriteState(int time, Simulation* simulation) const {
    OuterDynamicJointState state;
    state.q = ToVector(q_);
    state.qv = ToVector(v_);
    state.qdd = ToVector(qdd_);
    state.q_des = ToVector(q_des_);
    state.qv_des = ToVector(v_des_);
    state.tau_total = ToVector(tau_total_);
    simulation->WriteOuterDynamicState(time, state, const_cast<ROKAEArm*>(this));
}

void ROKAEArm::ValidateSize(const std::vector<double>& values, const char* name) const {
    if (values.size() != static_cast<std::size_t>(joint_count_)) {
        throw std::runtime_error(
            std::string("ROKAE_Arm ") + name + " must contain " +
            std::to_string(joint_count_) + " values.");
    }
}

void ROKAEArm::ResetState(const std::vector<double>& q, const std::vector<double>& qd) {
    ValidateSize(q, "q");
    ValidateSize(qd, "qd");
    q_ = Eigen::Map<const Eigen::VectorXd>(q.data(), static_cast<Eigen::Index>(q.size()));
    v_ = Eigen::Map<const Eigen::VectorXd>(qd.data(), static_cast<Eigen::Index>(qd.size()));
    qdd_.setZero();
    tau_total_.setZero();
    accumulated_torque_.setZero();
    ClampState();
    started_ = false;
}

void ROKAEArm::SetDesiredState(const std::vector<double>& q_des, const std::vector<double>& qd_des) {
    ValidateSize(q_des, "q_des");
    ValidateSize(qd_des, "qd_des");
    q_des_ = Eigen::Map<const Eigen::VectorXd>(q_des.data(), static_cast<Eigen::Index>(q_des.size()));
    v_des_ = Eigen::Map<const Eigen::VectorXd>(qd_des.data(), static_cast<Eigen::Index>(qd_des.size()));
    external_desired_state_enabled_ = true;
}

void ROKAEArm::AccumulateInputSpike(int joint_id, int type, float weight, int time) {
    (void)time;
    if (joint_id < 0 || joint_id >= joint_count_) {
        return;
    }
    if (type == 0) {
        accumulated_torque_(joint_id) += static_cast<double>(weight);
    } else if (type == 1) {
        accumulated_torque_(joint_id) -= static_cast<double>(weight);
    }
}

void ROKAEArm::ClearAccumulatedInputs() {
    accumulated_torque_.setZero();
}
