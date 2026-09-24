#ifndef ROKAE_ARM_H
#define ROKAE_ARM_H

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"

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

    std::string trajectory_mode_;
    double trajectory_start_s_;
    double motion_duration_s_;
    double dcn_torque_gain_;
    int spike_retention_steps_;
    bool enable_pd_control_;
    bool external_desired_state_enabled_;
    bool started_;

    void UpdateDesiredState(double time_seconds);
    void DecodeDcnTorque(int time, double step_seconds, Simulation* simulation, Eigen::VectorXd& torque);
    void Integrate(double step_seconds);
    void ClampState();
    void WriteState(int time, Simulation* simulation) const;
    void ValidateSize(const std::vector<double>& values, const char* name) const;
};

#endif
