#ifndef ROKAE_ARM_H
#define ROKAE_ARM_H

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"

#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

class ROKAEArm : public OuterDynamicModel {
public:
    explicit ROKAEArm(int timestep_size);
    virtual ~ROKAEArm();

    virtual void Initialize(const OuterDynamicDescription& description, Simulation* simulation);
    virtual void Update(int time, Simulation* simulation);
    virtual int GetJointCount() const { return joint_count_; }
    virtual void AccumulateInputSpike(int joint_id, int type, float weight, int time);
    virtual void ClearAccumulatedInputs();

    void ResetState(const std::vector<double>& q, const std::vector<double>& qd);
    void SetDesiredState(const std::vector<double>& q_des, const std::vector<double>& qd_des);

private:
    struct JointMapping {
        std::vector<int> positive_neurons;
        std::vector<int> negative_neurons;
    };

    pinocchio::Model model_;
    pinocchio::Data data_;
    bool model_ready_;
    int joint_count_;
    std::string end_effector_frame_;

    Eigen::VectorXd q_;
    Eigen::VectorXd v_;
    Eigen::VectorXd qdd_;
    Eigen::VectorXd q_start_;
    Eigen::VectorXd q_goal_;
    Eigen::VectorXd q_des_;
    Eigen::VectorXd v_des_;
    Eigen::VectorXd kp_;
    Eigen::VectorXd kd_;
    Eigen::VectorXd damping_;
    Eigen::VectorXd accumulated_torque_;
    Eigen::VectorXd tau_total_;
    Eigen::VectorXd torque_limit_;
    std::vector<JointMapping> joint_mappings_;
    std::vector<double> angle_min_deg_;
    std::vector<double> angle_max_deg_;
    std::vector<double> velocity_min_deg_;
    std::vector<double> velocity_max_deg_;
    std::vector<double> cf_angle_norm_deg_;
    std::vector<double> cf_velocity_norm_deg_;
    std::vector<std::vector<int> > cf_positive_neurons_;
    std::vector<std::vector<int> > cf_negative_neurons_;
    OuterDynamicFeedbackProductEncoding state_feedback_product_;
    OuterDynamicFeedbackSingleEncoding state_feedback_single_;

    std::string trajectory_mode_;
    double trajectory_start_s_;
    double motion_duration_s_;
    double dcn_torque_gain_;
    int spike_retention_steps_;
    int state_feedback_delay_steps_;
    int state_feedback_repeats_;
    int state_feedback_repeat_period_steps_;
    int error_feedback_delay_steps_;
    int error_feedback_window_steps_;
    int error_feedback_sample_count_;
    double cf_mix_position_;
    double spike_cf_max_;
    std::mt19937 error_feedback_rng_;
    bool enable_pd_control_;
    bool external_desired_state_enabled_;
    bool started_;

    void UpdateDesiredState(double time_seconds);
    void DecodeDcnTorque(int time, double step_seconds, Simulation* simulation, Eigen::VectorXd& torque);
    void Integrate(double step_seconds);
    void ClampState();
    void WriteState(int time, Simulation* simulation) const;
    void ValidateSize(const std::vector<double>& values, const char* name) const;
    void EncodeAndEmitStateFeedback(int time, Simulation* simulation);
    void EncodeAndEmitErrorFeedback(int time, Simulation* simulation);
    void EmitBinaryCfSpikes(int count,
                            const std::vector<int>& target_neurons,
                            int base_time,
                            int window_steps,
                            Simulation* simulation);
    int EncodeStateProductIndex(int joint) const;
    int EncodeUniformBin(double value, double min_value, double max_value, int bin_count) const;
    int EncodeProductIndex(const std::vector<int>& indices, const std::vector<int>& bins) const;
    void EmitSpike(Simulation* simulation, int neuron_id, int time);
};

#endif
