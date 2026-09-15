#include "../source_file_realtime_v1_async/OuterDynamic/inc/StrictMatlabPlanarArm2DOFOuterDynamic.h"

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeBuffer.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"

#include <boost/any.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>

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

std::array<double, 2> GetVector2OrDefault(
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

double RadToDeg(double radians) {
    return radians * 180.0 / 3.14159265358979323846;
}

double LinearSaturating(double max_abs, double value) {
    if (max_abs <= 0.0) {
        return 0.0;
    }
    return std::max(0.0, std::min(1.0, value / max_abs));
}

} // namespace

StrictMatlabPlanarArm2DOFOuterDynamic::StrictMatlabPlanarArm2DOFOuterDynamic(int timestep_size)
    : OuterDynamicModel(timestep_size),
      joint_mappings_(),
      arm_(),
      torque_scale_{ 0.2, 0.05 },
      angle_min_deg_{ -30.0, 0.1 },
      angle_max_deg_{ 90.0, 150.0 },
      velocity_min_deg_{ -400.0, -400.0 },
      velocity_max_deg_{ 400.0, 400.0 },
      desired_q_{ 0.0, 0.0 },
      desired_qd_{ 0.0, 0.0 },
      desired_xy_{ 0.0, 0.0 },
      desired_xy_dot_{ 0.0, 0.0 },
      accumulated_joint_torque_{ 0.0, 0.0 },
      ik_options_(),
      desired_mode_(JointSpaceDesired),
      spike_retention_steps_(512),
      state_feedback_delay_steps_(1),
      state_feedback_repeats_(1),
      state_feedback_repeat_period_steps_(1),
      error_feedback_delay_steps_(0),
      error_feedback_window_steps_(1),
      error_feedback_sample_count_(50),
      cf_mix_position_(0.8),
      spike_cf_max_(15.0),
      cf_angle_norm_deg_{ 60.0, 75.0 },
      cf_velocity_norm_deg_{ 400.0, 400.0 },
      cf_positive_neurons_(),
      cf_negative_neurons_(),
      error_feedback_rng_(17u),
      started_(false) {}

StrictMatlabPlanarArm2DOFOuterDynamic::~StrictMatlabPlanarArm2DOFOuterDynamic() {}

StrictMatlabPlanarArm2DOFPinocchioPlant& StrictMatlabPlanarArm2DOFOuterDynamic::Arm() {
    if (!this->arm_) {
        throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic arm plant is not initialized.");
    }
    return *this->arm_;
}

const StrictMatlabPlanarArm2DOFPinocchioPlant& StrictMatlabPlanarArm2DOFOuterDynamic::Arm() const {
    if (!this->arm_) {
        throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic arm plant is not initialized.");
    }
    return *this->arm_;
}

void StrictMatlabPlanarArm2DOFOuterDynamic::Initialize(const OuterDynamicDescription& description, Simulation* simulation) {
    this->setTimestepSize(std::max(1, description.update_timestep));
    this->setCommunicationInterval(std::max(1, description.communication_interval));
    this->setQueueIndex(description.queue_index);

    StrictMatlabPlanarArm2DOFConfig config;
    const std::array<double, 2> link_lengths = GetVector2OrDefault(
        description.ModelParameter, "link_lengths", { config.link1, config.link2 });
    const std::array<double, 2> link_masses = GetVector2OrDefault(
        description.ModelParameter, "link_masses", { config.mass1, config.mass2 });
    const std::array<double, 2> angle_min_deg = GetVector2OrDefault(
        description.ModelParameter, "angle_min_deg", { -30.0, 0.1 });
    const std::array<double, 2> angle_max_deg = GetVector2OrDefault(
        description.ModelParameter, "angle_max_deg", { 90.0, 150.0 });
    this->velocity_min_deg_ = GetVector2OrDefault(
        description.ModelParameter, "velocity_min_deg_s", this->velocity_min_deg_);
    const std::array<double, 2> velocity_range_deg = GetVector2OrDefault(
        description.ModelParameter, "velocity_range_deg_s", { 800.0, 800.0 });
    this->velocity_max_deg_ = {
        this->velocity_min_deg_[0] + velocity_range_deg[0],
        this->velocity_min_deg_[1] + velocity_range_deg[1]
    };
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
    this->cf_angle_norm_deg_ = GetVector2OrDefault(
        description.ModelParameter, "cf_angle_norm_deg", this->cf_angle_norm_deg_);
    this->cf_velocity_norm_deg_ = GetVector2OrDefault(
        description.ModelParameter, "cf_velocity_norm_deg_s", this->cf_velocity_norm_deg_);
    this->error_feedback_rng_.seed(static_cast<std::mt19937::result_type>(
        GetAnyOrDefault<int>(description.ModelParameter, "error_feedback_seed", 17)));

    config.link1 = link_lengths[0];
    config.link2 = link_lengths[1];
    config.mass1 = link_masses[0];
    config.mass2 = link_masses[1];
    config.q1_min = angle_min_deg[0] * 3.14159265358979323846 / 180.0;
    config.q1_max = angle_max_deg[0] * 3.14159265358979323846 / 180.0;
    config.q2_min = angle_min_deg[1] * 3.14159265358979323846 / 180.0;
    config.q2_max = angle_max_deg[1] * 3.14159265358979323846 / 180.0;
    this->angle_min_deg_ = angle_min_deg;
    this->angle_max_deg_ = angle_max_deg;

    this->arm_.reset(new StrictMatlabPlanarArm2DOFPinocchioPlant(config));
    this->state_feedback_product_ = description.state_feedback_product;
    this->state_feedback_single_ = description.state_feedback_single;
    if (!description.cf_positive_neuron_indices_by_joint.empty() ||
        !description.cf_negative_neuron_indices_by_joint.empty()) {
        if (description.cf_positive_neuron_indices_by_joint.size() != 2 ||
            description.cf_negative_neuron_indices_by_joint.size() != 2) {
            throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic error feedback expects positive and negative CF targets for 2 joints.");
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
    this->torque_scale_ = GetVector2OrDefault(description.ModelParameter, "torque_scale", this->torque_scale_);
    this->spike_retention_steps_ = GetAnyOrDefault<int>(
        description.ModelParameter, "spike_retention_steps", this->spike_retention_steps_);
    this->ik_options_.max_iter = GetAnyOrDefault<int>(description.ModelParameter, "ik_max_iter", this->ik_options_.max_iter);
    this->ik_options_.tolerance = GetAnyOrDefault<double>(description.ModelParameter, "ik_tolerance", this->ik_options_.tolerance);
    this->ik_options_.damping = GetAnyOrDefault<double>(description.ModelParameter, "ik_damping", this->ik_options_.damping);
    this->ik_options_.step_size = GetAnyOrDefault<double>(description.ModelParameter, "ik_step_size", this->ik_options_.step_size);
    this->ik_options_.clamp_joint_limits = GetAnyOrDefault<bool>(
        description.ModelParameter, "ik_clamp_joint_limits", this->ik_options_.clamp_joint_limits);
    this->ik_options_.verbose = GetAnyOrDefault<bool>(description.ModelParameter, "ik_verbose", this->ik_options_.verbose);

    const bool has_legacy_dcn_mapping =
        !description.dcn_positive_neuron_indices_by_joint.empty() ||
        !description.dcn_negative_neuron_indices_by_joint.empty();
    if (has_legacy_dcn_mapping &&
        (description.dcn_positive_neuron_indices_by_joint.size() != 2 ||
         description.dcn_negative_neuron_indices_by_joint.size() != 2)) {
        throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic expects 2-joint DCN positive/negative mappings when legacy DCN mapping is used.");
    }

    std::vector<int> watched_neurons;
    for (int joint = 0; joint < 2; ++joint) {
        if (has_legacy_dcn_mapping) {
            this->joint_mappings_[joint].dcn_positive_neurons = description.dcn_positive_neuron_indices_by_joint[joint];
            this->joint_mappings_[joint].dcn_negative_neurons = description.dcn_negative_neuron_indices_by_joint[joint];
        } else {
            this->joint_mappings_[joint].dcn_positive_neurons.clear();
            this->joint_mappings_[joint].dcn_negative_neurons.clear();
        }
        watched_neurons.insert(
            watched_neurons.end(),
            this->joint_mappings_[joint].dcn_positive_neurons.begin(),
            this->joint_mappings_[joint].dcn_positive_neurons.end());
        watched_neurons.insert(
            watched_neurons.end(),
            this->joint_mappings_[joint].dcn_negative_neurons.begin(),
            this->joint_mappings_[joint].dcn_negative_neurons.end());
    }

    if (simulation->outer_dynamic_spike_buffer != NULL) {
        simulation->outer_dynamic_spike_buffer->RegisterWatchedNeurons(watched_neurons);
    }

    this->Arm().ResetState(this->desired_q_, { 0.0, 0.0 });
    this->Arm().SetDesiredState(this->desired_q_, this->desired_qd_);
    this->Arm().SetTorqueCommand({ 0.0, 0.0 });
    this->accumulated_joint_torque_ = { 0.0, 0.0 };
    this->started_ = false;
}

void StrictMatlabPlanarArm2DOFOuterDynamic::Update(int time, Simulation* simulation) {
    if (this->desired_mode_ == CartesianDesired) {
        StrictIkResult ik_result;
        const std::array<double, 2> q_seed = this->Arm().GetState().q;
        this->desired_q_ = this->Arm().InverseKinematicsXY(q_seed, this->desired_xy_, this->ik_options_, &ik_result);
        this->desired_qd_ = this->Arm().JointVelocityFromCartesianXY(this->desired_q_, this->desired_xy_dot_, this->ik_options_);
    }
    this->Arm().SetDesiredState(this->desired_q_, this->desired_qd_);

    if (!this->started_) {
        const StrictMatlabPlanarArm2DOFState& arm_state = this->Arm().GetState();
        OuterDynamicJointState state;
        state.q = arm_state.q;
        state.qv = arm_state.qd;
        state.qdd = arm_state.qdd;
        state.q_des = arm_state.q_des;
        state.qv_des = arm_state.qd_des;
        state.tau_total = { 0.0, 0.0 };
        simulation->WriteOuterDynamicState(time, state, this);
        this->EncodeAndEmitStateFeedback(time, simulation, arm_state);
        this->started_ = true;
        return;
    }

    std::array<double, 2> tau = { 0.0, 0.0 };
    this->DecodeDcnTorqueWindow(time, this->getCommunicationInterval(), simulation, tau);
    tau[0] += this->accumulated_joint_torque_[0];
    tau[1] += this->accumulated_joint_torque_[1];
    this->Arm().SetTorqueCommand(tau);
    this->Arm().Step(static_cast<double>(this->getTimestepSize()) * static_cast<double>(simulation->basetimesteps) * 0.001);

    const StrictMatlabPlanarArm2DOFState& arm_state = this->Arm().GetState();
    OuterDynamicJointState state;
    state.q = arm_state.q;
    state.qv = arm_state.qd;
    state.qdd = arm_state.qdd;
    state.q_des = arm_state.q_des;
    state.qv_des = arm_state.qd_des;
    state.tau_total = tau;
    simulation->WriteOuterDynamicState(time, state, this);
    this->EncodeAndEmitStateFeedback(time, simulation, arm_state);
    this->EncodeAndEmitErrorFeedback(time, simulation, arm_state);

    if (simulation->outer_dynamic_spike_buffer != NULL) {
        simulation->outer_dynamic_spike_buffer->DiscardOlderThan(std::max(0, time - this->spike_retention_steps_));
    }
}

void StrictMatlabPlanarArm2DOFOuterDynamic::EncodeAndEmitStateFeedback(
    int time,
    Simulation* simulation,
    const StrictMatlabPlanarArm2DOFState& arm_state) {
    if (simulation == NULL) {
        return;
    }
    const int emit_time = time + this->state_feedback_delay_steps_;

    if (!this->state_feedback_product_.neuron_indices_by_joint.empty()) {
        if (this->state_feedback_product_.neuron_indices_by_joint.size() != 2 ||
            this->state_feedback_product_.bins.size() != 4) {
            throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic state_feedback_product expects 2 joints and 4 state bins.");
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
                throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic state_feedback_product target size does not match product bins.");
            }
            const int index = this->EncodeStateProductIndex(joint, arm_state);
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
            throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic state_feedback_single expects 2 joints and 4 state variables.");
        }
        for (int joint = 0; joint < 2; ++joint) {
            const std::vector<double> values = {
                arm_state.q_des[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846,
                arm_state.q[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846,
                arm_state.qd_des[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846,
                arm_state.qd[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846
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
                throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic state_feedback_single target variable count must be 4.");
            }
            for (int variable = 0; variable < 4; ++variable) {
                const int bin_count = this->state_feedback_single_.bins[static_cast<std::size_t>(variable)];
                if (bin_count <= 0 ||
                    static_cast<int>(joint_targets[static_cast<std::size_t>(variable)].size()) != bin_count) {
                    throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic state_feedback_single target bin count is invalid.");
                }
                const int bin = this->EncodeUniformBin(values[static_cast<std::size_t>(variable)],
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

void StrictMatlabPlanarArm2DOFOuterDynamic::EncodeAndEmitErrorFeedback(
    int time,
    Simulation* simulation,
    const StrictMatlabPlanarArm2DOFState& arm_state) {
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
            throw std::runtime_error("StrictMatlabPlanarArm2DOFOuterDynamic error feedback requires matched positive/negative CF targets per joint.");
        }
    }

    const std::array<double, 2> position_error = {
        RadToDeg(arm_state.q_des[0]) - RadToDeg(arm_state.q[0]),
        RadToDeg(arm_state.q_des[1]) - RadToDeg(arm_state.q[1])
    };
    const std::array<double, 2> velocity_error = {
        RadToDeg(arm_state.qd_des[0]) - RadToDeg(arm_state.qd[0]),
        RadToDeg(arm_state.qd_des[1]) - RadToDeg(arm_state.qd[1])
    };

    for (int joint = 0; joint < 2; ++joint) {
        double normalized_angle =
            LinearSaturating(this->cf_angle_norm_deg_[static_cast<std::size_t>(joint)],
                             std::fabs(position_error[static_cast<std::size_t>(joint)]));
        double normalized_velocity =
            LinearSaturating(this->cf_velocity_norm_deg_[static_cast<std::size_t>(joint)],
                             std::fabs(velocity_error[static_cast<std::size_t>(joint)]));
        if (position_error[static_cast<std::size_t>(joint)] < 0.0) {
            normalized_angle = -normalized_angle;
        }
        if (velocity_error[static_cast<std::size_t>(joint)] < 0.0) {
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

void StrictMatlabPlanarArm2DOFOuterDynamic::EmitBinaryCfSpikes(
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
            // Emit into the next available step range so CF spikes enter the
            // regular InputSpikeNeuronModel and TriggerRule path.
            this->EmitSpike(simulation, target_neurons[neuron], base_time + 1 + local_time);
        }
    }
}

int StrictMatlabPlanarArm2DOFOuterDynamic::EncodeStateProductIndex(
    int joint,
    const StrictMatlabPlanarArm2DOFState& arm_state) const {
    const std::vector<double> values = {
        arm_state.q_des[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846,
        arm_state.q[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846,
        arm_state.qd_des[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846,
        arm_state.qd[static_cast<std::size_t>(joint)] * 180.0 / 3.14159265358979323846
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
    std::vector<int> indices;
    indices.reserve(4);
    for (int variable = 0; variable < 4; ++variable) {
        indices.push_back(this->EncodeUniformBin(values[static_cast<std::size_t>(variable)],
                                                 mins[static_cast<std::size_t>(variable)],
                                                 maxs[static_cast<std::size_t>(variable)],
                                                 this->state_feedback_product_.bins[static_cast<std::size_t>(variable)]));
    }
    return this->EncodeProductIndex(indices, this->state_feedback_product_.bins);
}

int StrictMatlabPlanarArm2DOFOuterDynamic::EncodeUniformBin(
    double value,
    double min_value,
    double max_value,
    int bin_count) const {
    if (bin_count <= 1 || max_value <= min_value) {
        return 0;
    }
    const double clamped = std::max(min_value, std::min(max_value, value));
    const double ratio = (clamped - min_value) / (max_value - min_value);
    int index = static_cast<int>(std::floor(ratio * static_cast<double>(bin_count)));
    if (index >= bin_count) {
        index = bin_count - 1;
    }
    return std::max(0, index);
}

int StrictMatlabPlanarArm2DOFOuterDynamic::EncodeProductIndex(
    const std::vector<int>& indices,
    const std::vector<int>& bins) const {
    if (indices.size() != bins.size()) {
        throw std::runtime_error("OuterDynamic product index dimensions do not match bins.");
    }
    int index = 0;
    for (std::size_t dimension = 0; dimension < indices.size(); ++dimension) {
        if (bins[dimension] <= 0 ||
            indices[dimension] < 0 ||
            indices[dimension] >= bins[dimension]) {
            throw std::runtime_error("OuterDynamic product index contains invalid bin.");
        }
        index = index * bins[dimension] + indices[dimension];
    }
    return index;
}

void StrictMatlabPlanarArm2DOFOuterDynamic::EmitSpike(
    Simulation* simulation,
    int neuron_id,
    int time) {
    if (simulation == NULL || simulation->network == NULL) {
        return;
    }
    if (neuron_id < 0 || neuron_id >= simulation->network->neuronsNum) {
        return;
    }
    simulation->QueueOuterDynamicInputSpike(time, neuron_id);
}

void StrictMatlabPlanarArm2DOFOuterDynamic::AccumulateInputSpike(
    int joint_id,
    int type,
    float weight,
    int time) {
    (void)time;
    if (joint_id < 0 || joint_id >= 2) {
        return;
    }
    if (type == 0) {
        this->accumulated_joint_torque_[static_cast<std::size_t>(joint_id)] += weight;
    } else if (type == 1) {
        this->accumulated_joint_torque_[static_cast<std::size_t>(joint_id)] -= weight;
    }
}

void StrictMatlabPlanarArm2DOFOuterDynamic::ClearAccumulatedInputs() {
    this->accumulated_joint_torque_ = { 0.0, 0.0 };
}

void StrictMatlabPlanarArm2DOFOuterDynamic::ResetState(const std::array<double, 2>& q, const std::array<double, 2>& qd) {
    this->Arm().ResetState(q, qd);
    this->started_ = false;
}

void StrictMatlabPlanarArm2DOFOuterDynamic::SetDesiredState(const std::array<double, 2>& q_des, const std::array<double, 2>& qd_des) {
    this->desired_mode_ = JointSpaceDesired;
    this->desired_q_ = q_des;
    this->desired_qd_ = qd_des;
    if (this->arm_) {
        this->arm_->SetDesiredState(q_des, qd_des);
    }
}

void StrictMatlabPlanarArm2DOFOuterDynamic::SetDesiredHandTarget(
    const std::array<double, 2>& xy_des,
    const std::array<double, 2>& xy_dot_des) {
    this->desired_mode_ = CartesianDesired;
    this->desired_xy_ = xy_des;
    this->desired_xy_dot_ = xy_dot_des;
}

void StrictMatlabPlanarArm2DOFOuterDynamic::DecodeDcnTorqueWindow(
    int time,
    int window_steps,
    Simulation* simulation,
    std::array<double, 2>& tau) const {
    if (simulation->outer_dynamic_spike_buffer == NULL) {
        tau = { 0.0, 0.0 };
        return;
    }

    const int window_start = std::max(0, time - std::max(1, window_steps));
    for (int joint = 0; joint < 2; ++joint) {
        const int positive_count = simulation->outer_dynamic_spike_buffer->CountSpikesInWindow(
            window_start, time, this->joint_mappings_[joint].dcn_positive_neurons);
        const int negative_count = simulation->outer_dynamic_spike_buffer->CountSpikesInWindow(
            window_start, time, this->joint_mappings_[joint].dcn_negative_neurons);
        tau[joint] = this->torque_scale_[joint] * static_cast<double>(positive_count - negative_count);
    }
}
