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

double RadToDeg(double radians) {
    return radians * 180.0 / kPi;
}

double LinearSaturating(double max_abs, double value) {
    if (max_abs <= 0.0) {
        return 0.0;
    }
    return std::max(0.0, std::min(1.0, value / max_abs));
}

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
      state_feedback_delay_steps_(1),
      state_feedback_repeats_(1),
      state_feedback_repeat_period_steps_(1),
      error_feedback_delay_steps_(0),
      error_feedback_window_steps_(1),
      error_feedback_sample_count_(50),
      cf_mix_position_(0.8),
      spike_cf_max_(15.0),
      error_feedback_rng_(17u),
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

    Eigen::VectorXd angle_min_default(model_.nv);
    Eigen::VectorXd angle_max_default(model_.nv);
    Eigen::VectorXd velocity_min_default(model_.nv);
    Eigen::VectorXd velocity_range_default(model_.nv);
    Eigen::VectorXd cf_angle_default(model_.nv);
    Eigen::VectorXd cf_velocity_default(model_.nv);
    for (int joint = 0; joint < joint_count_; ++joint) {
        angle_min_default(joint) = RadToDeg(model_.lowerPositionLimit(joint));
        angle_max_default(joint) = RadToDeg(model_.upperPositionLimit(joint));
        const double velocity_limit_deg = RadToDeg(model_.velocityLimit(joint));
        velocity_min_default(joint) = -velocity_limit_deg;
        velocity_range_default(joint) = 2.0 * velocity_limit_deg;
        cf_angle_default(joint) = std::max(1.0, 0.5 * (angle_max_default(joint) - angle_min_default(joint)));
        cf_velocity_default(joint) = std::max(1.0, velocity_limit_deg);
    }
    const Eigen::VectorXd angle_min = GetVector(description.ModelParameter, "angle_min_deg", angle_min_default);
    const Eigen::VectorXd angle_max = GetVector(description.ModelParameter, "angle_max_deg", angle_max_default);
    const Eigen::VectorXd velocity_min = GetVector(description.ModelParameter, "velocity_min_deg_s", velocity_min_default);
    const Eigen::VectorXd velocity_range = GetVector(description.ModelParameter, "velocity_range_deg_s", velocity_range_default);
    const Eigen::VectorXd cf_angle_norm = GetVector(description.ModelParameter, "cf_angle_norm_deg", cf_angle_default);
    const Eigen::VectorXd cf_velocity_norm = GetVector(description.ModelParameter, "cf_velocity_norm_deg_s", cf_velocity_default);
    angle_min_deg_ = ToVector(angle_min);
    angle_max_deg_ = ToVector(angle_max);
    velocity_min_deg_ = ToVector(velocity_min);
    velocity_max_deg_ = ToVector(velocity_min + velocity_range);
    cf_angle_norm_deg_ = ToVector(cf_angle_norm);
    cf_velocity_norm_deg_ = ToVector(cf_velocity_norm);

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
    state_feedback_delay_steps_ = std::max(
        0, GetOrDefault<int>(description.ModelParameter, "state_feedback_delay_steps", state_feedback_delay_steps_));
    state_feedback_repeats_ = std::max(
        1, GetOrDefault<int>(description.ModelParameter, "state_feedback_repeats_per_update", state_feedback_repeats_));
    state_feedback_repeat_period_steps_ = std::max(
        1, GetOrDefault<int>(description.ModelParameter, "state_feedback_repeat_period_steps", state_feedback_repeat_period_steps_));
    error_feedback_delay_steps_ = std::max(
        0, GetOrDefault<int>(description.ModelParameter, "error_feedback_delay_steps", error_feedback_delay_steps_));
    error_feedback_window_steps_ = std::max(
        1, GetOrDefault<int>(description.ModelParameter, "error_feedback_window_steps", getCommunicationInterval()));
    error_feedback_sample_count_ = std::max(
        1, GetOrDefault<int>(description.ModelParameter, "error_feedback_sample_count", error_feedback_sample_count_));
    cf_mix_position_ = std::clamp(
        GetOrDefault<double>(description.ModelParameter, "cf_mix_position", cf_mix_position_), 0.0, 1.0);
    spike_cf_max_ = std::max(
        0.0, GetOrDefault<double>(description.ModelParameter, "spike_cf_max", spike_cf_max_));
    error_feedback_rng_.seed(static_cast<std::mt19937::result_type>(
        GetOrDefault<int>(description.ModelParameter, "error_feedback_seed", 17)));
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
    state_feedback_product_ = description.state_feedback_product;
    state_feedback_single_ = description.state_feedback_single;
    if (!description.error_feedback_product.neuron_indices_by_joint.empty() ||
        !description.error_feedback_single.neuron_indices_by_joint_variable.empty()) {
        throw std::runtime_error("ROKAE_Arm supports signed CF mappings instead of generic error feedback encodings.");
    }
    const bool has_cf_mapping =
        !description.cf_positive_neuron_indices_by_joint.empty() ||
        !description.cf_negative_neuron_indices_by_joint.empty();
    if (has_cf_mapping &&
        (description.cf_positive_neuron_indices_by_joint.size() != static_cast<std::size_t>(joint_count_) ||
         description.cf_negative_neuron_indices_by_joint.size() != static_cast<std::size_t>(joint_count_))) {
        throw std::runtime_error("ROKAE_Arm CF positive/negative mappings must each contain 6 joints.");
    }
    cf_positive_neurons_.assign(static_cast<std::size_t>(joint_count_), std::vector<int>());
    cf_negative_neurons_.assign(static_cast<std::size_t>(joint_count_), std::vector<int>());
    if (has_cf_mapping) {
        cf_positive_neurons_ = description.cf_positive_neuron_indices_by_joint;
        cf_negative_neurons_ = description.cf_negative_neuron_indices_by_joint;
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
        EncodeAndEmitStateFeedback(time, simulation);
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
    EncodeAndEmitStateFeedback(time, simulation);
    EncodeAndEmitErrorFeedback(time, simulation);
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

void ROKAEArm::EncodeAndEmitStateFeedback(int time, Simulation* simulation) {
    if (simulation == nullptr) {
        return;
    }
    const int emit_time = time + state_feedback_delay_steps_;
    if (!state_feedback_product_.neuron_indices_by_joint.empty()) {
        if (state_feedback_product_.neuron_indices_by_joint.size() != static_cast<std::size_t>(joint_count_) ||
            state_feedback_product_.bins.size() != 4) {
            throw std::runtime_error("ROKAE_Arm state_feedback_product expects 6 joints and 4 state bins.");
        }
        const int product_size = state_feedback_product_.bins[0] * state_feedback_product_.bins[1] *
                                 state_feedback_product_.bins[2] * state_feedback_product_.bins[3];
        for (int joint = 0; joint < joint_count_; ++joint) {
            const std::vector<int>& targets =
                state_feedback_product_.neuron_indices_by_joint[static_cast<std::size_t>(joint)];
            if (static_cast<int>(targets.size()) != product_size) {
                throw std::runtime_error("ROKAE_Arm state_feedback_product target size does not match product bins.");
            }
            const int index = EncodeStateProductIndex(joint);
            for (int repeat = 0; repeat < state_feedback_repeats_; ++repeat) {
                EmitSpike(simulation, targets[static_cast<std::size_t>(index)],
                          emit_time + repeat * state_feedback_repeat_period_steps_);
            }
        }
    }

    if (!state_feedback_single_.neuron_indices_by_joint_variable.empty()) {
        if (state_feedback_single_.neuron_indices_by_joint_variable.size() != static_cast<std::size_t>(joint_count_) ||
            state_feedback_single_.bins.size() != 4) {
            throw std::runtime_error("ROKAE_Arm state_feedback_single expects 6 joints and 4 state variables.");
        }
        for (int joint = 0; joint < joint_count_; ++joint) {
            const std::vector<double> values = {
                RadToDeg(q_des_(joint)), RadToDeg(q_(joint)), RadToDeg(v_des_(joint)), RadToDeg(v_(joint))};
            const std::vector<double> mins = {
                angle_min_deg_[joint], angle_min_deg_[joint], velocity_min_deg_[joint], velocity_min_deg_[joint]};
            const std::vector<double> maxs = {
                angle_max_deg_[joint], angle_max_deg_[joint], velocity_max_deg_[joint], velocity_max_deg_[joint]};
            const std::vector<std::vector<int> >& targets =
                state_feedback_single_.neuron_indices_by_joint_variable[static_cast<std::size_t>(joint)];
            if (targets.size() != 4) {
                throw std::runtime_error("ROKAE_Arm state_feedback_single target variable count must be 4.");
            }
            for (int variable = 0; variable < 4; ++variable) {
                const int bin_count = state_feedback_single_.bins[static_cast<std::size_t>(variable)];
                if (bin_count <= 0 ||
                    static_cast<int>(targets[static_cast<std::size_t>(variable)].size()) != bin_count) {
                    throw std::runtime_error("ROKAE_Arm state_feedback_single target bin count is invalid.");
                }
                const int bin = EncodeUniformBin(values[variable], mins[variable], maxs[variable], bin_count);
                for (int repeat = 0; repeat < state_feedback_repeats_; ++repeat) {
                    EmitSpike(simulation,
                              targets[static_cast<std::size_t>(variable)][static_cast<std::size_t>(bin)],
                              emit_time + repeat * state_feedback_repeat_period_steps_);
                }
            }
        }
    }
}

void ROKAEArm::EncodeAndEmitErrorFeedback(int time, Simulation* simulation) {
    if (simulation == nullptr || cf_positive_neurons_.empty()) {
        return;
    }
    for (int joint = 0; joint < joint_count_; ++joint) {
        const std::vector<int>& positive = cf_positive_neurons_[static_cast<std::size_t>(joint)];
        const std::vector<int>& negative = cf_negative_neurons_[static_cast<std::size_t>(joint)];
        if (positive.empty() && negative.empty()) {
            continue;
        }
        if (positive.empty() || positive.size() != negative.size()) {
            throw std::runtime_error("ROKAE_Arm error feedback requires matched positive/negative CF targets per joint.");
        }
        const double position_error = RadToDeg(q_des_(joint) - q_(joint));
        const double velocity_error = RadToDeg(v_des_(joint) - v_(joint));
        double normalized_angle = LinearSaturating(cf_angle_norm_deg_[joint], std::fabs(position_error));
        double normalized_velocity = LinearSaturating(cf_velocity_norm_deg_[joint], std::fabs(velocity_error));
        normalized_angle = position_error < 0.0 ? -normalized_angle : normalized_angle;
        normalized_velocity = velocity_error < 0.0 ? -normalized_velocity : normalized_velocity;
        const double mixed_error =
            cf_mix_position_ * normalized_angle + (1.0 - cf_mix_position_) * normalized_velocity;
        const int count = static_cast<int>(std::llround(
            std::fabs(spike_cf_max_ * static_cast<double>(positive.size()) /
                      static_cast<double>(error_feedback_sample_count_) * mixed_error)));
        EmitBinaryCfSpikes(count,
                           mixed_error > 0.0 ? positive : negative,
                           time + error_feedback_delay_steps_,
                           error_feedback_window_steps_,
                           simulation);
    }
}

void ROKAEArm::EmitBinaryCfSpikes(int count,
                                  const std::vector<int>& target_neurons,
                                  int base_time,
                                  int window_steps,
                                  Simulation* simulation) {
    if (count <= 0 || target_neurons.empty() || window_steps <= 0 || simulation == nullptr) {
        return;
    }
    std::vector<unsigned char> bitmap(
        target_neurons.size() * static_cast<std::size_t>(window_steps), 0u);
    std::uniform_int_distribution<int> neuron_dist(0, static_cast<int>(target_neurons.size()) - 1);
    std::uniform_int_distribution<int> time_dist(0, window_steps - 1);
    for (int event = 0; event < count; ++event) {
        const int neuron = neuron_dist(error_feedback_rng_);
        const int local_time = time_dist(error_feedback_rng_);
        bitmap[static_cast<std::size_t>(neuron) * static_cast<std::size_t>(window_steps) +
               static_cast<std::size_t>(local_time)] = 1u;
    }
    for (std::size_t neuron = 0; neuron < target_neurons.size(); ++neuron) {
        for (int local_time = 0; local_time < window_steps; ++local_time) {
            if (bitmap[neuron * static_cast<std::size_t>(window_steps) +
                       static_cast<std::size_t>(local_time)] != 0u) {
                EmitSpike(simulation, target_neurons[neuron], base_time + 1 + local_time);
            }
        }
    }
}

int ROKAEArm::EncodeStateProductIndex(int joint) const {
    const std::vector<double> values = {
        RadToDeg(q_des_(joint)), RadToDeg(q_(joint)), RadToDeg(v_des_(joint)), RadToDeg(v_(joint))};
    const std::vector<double> mins = {
        angle_min_deg_[joint], angle_min_deg_[joint], velocity_min_deg_[joint], velocity_min_deg_[joint]};
    const std::vector<double> maxs = {
        angle_max_deg_[joint], angle_max_deg_[joint], velocity_max_deg_[joint], velocity_max_deg_[joint]};
    std::vector<int> indices;
    indices.reserve(4);
    for (int variable = 0; variable < 4; ++variable) {
        indices.push_back(EncodeUniformBin(
            values[variable], mins[variable], maxs[variable],
            state_feedback_product_.bins[static_cast<std::size_t>(variable)]));
    }
    return EncodeProductIndex(indices, state_feedback_product_.bins);
}

int ROKAEArm::EncodeUniformBin(double value, double min_value, double max_value, int bin_count) const {
    if (bin_count <= 1 || max_value <= min_value) {
        return 0;
    }
    const double ratio = (std::clamp(value, min_value, max_value) - min_value) / (max_value - min_value);
    return std::clamp(static_cast<int>(std::floor(ratio * static_cast<double>(bin_count))), 0, bin_count - 1);
}

int ROKAEArm::EncodeProductIndex(const std::vector<int>& indices, const std::vector<int>& bins) const {
    if (indices.size() != bins.size()) {
        throw std::runtime_error("ROKAE_Arm product index dimensions do not match bins.");
    }
    int index = 0;
    for (std::size_t dimension = 0; dimension < indices.size(); ++dimension) {
        if (bins[dimension] <= 0 || indices[dimension] < 0 || indices[dimension] >= bins[dimension]) {
            throw std::runtime_error("ROKAE_Arm product index contains an invalid bin.");
        }
        index = index * bins[dimension] + indices[dimension];
    }
    return index;
}

void ROKAEArm::EmitSpike(Simulation* simulation, int neuron_id, int time) {
    if (simulation == nullptr || simulation->network == nullptr ||
        neuron_id < 0 || neuron_id >= simulation->network->neuronsNum) {
        return;
    }
    simulation->QueueOuterDynamicInputSpike(time, neuron_id);
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
