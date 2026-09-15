#ifndef PLANAR_ARM_2DOF_PINOCCHIO_H
#define PLANAR_ARM_2DOF_PINOCCHIO_H

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"

#include <array>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <pinocchio/spatial/se3.hpp>

class PlanarArm2DOFPinocchio : public OuterDynamicModel {
public:
    explicit PlanarArm2DOFPinocchio(int timestep_size);
    virtual ~PlanarArm2DOFPinocchio();

    virtual void Initialize(const OuterDynamicDescription& description, Simulation* simulation);
    virtual void Update(int time, Simulation* simulation);
    virtual int GetJointCount() const { return 2; }
    virtual void AccumulateInputSpike(int joint_id, int type, float weight, int time);
    virtual void ClearAccumulatedInputs();

    void ResetState(const std::array<double, 2>& q, const std::array<double, 2>& qd);
    void SetDesiredState(const std::array<double, 2>& q_des, const std::array<double, 2>& qd_des);

private:
    struct JointMapping {
        std::vector<int> dcn_positive_neurons;
        std::vector<int> dcn_negative_neurons;
    };

    pinocchio::Model model_;
    pinocchio::Data data_aba_;
    pinocchio::Data data_rnea_;
    pinocchio::Data data_fk_;
    bool model_ready_;
    pinocchio::FrameIndex end_effector_frame_id_;
    std::vector<JointMapping> joint_mappings_;

    Eigen::Vector2d q_;
    Eigen::Vector2d v_;
    Eigen::Vector2d qdd_;
    Eigen::Vector2d q_des_;
    Eigen::Vector2d v_des_;
    Eigen::Vector2d qdd_des_;
    Eigen::Vector2d tau_pd_;
    Eigen::Vector2d tau_id_;
    Eigen::Vector2d tau_cereb_;
    Eigen::Vector2d tau_cereb_positive_;
    Eigen::Vector2d tau_cereb_negative_;
    Eigen::Vector2d accumulated_joint_torque_;
    Eigen::Vector2d tau_total_;

    Eigen::Vector2d link_lengths_;
    Eigen::Vector2d link_masses_;
    Eigen::Vector2d damping_;
    Eigen::Vector2d kp_;
    Eigen::Vector2d kd_;

    double gravity_magnitude_;
    double trajectory_frequency_hz_;
    double circle_center_x_;
    double circle_center_y_;
    double circle_radius_;
    double dcn_torque_gain_;
    int spike_retention_steps_;
    // State feedback is emitted as ordinary input spikes after the plant state is integrated.
    int state_feedback_delay_steps_;
    int state_feedback_repeats_;
    int state_feedback_repeat_period_steps_;
    // Error feedback is encoded as binary CF spikes using the same path as StrictMatlabPlanarArm2DOFOuterDynamic.
    int error_feedback_delay_steps_;
    int error_feedback_window_steps_;
    int error_feedback_sample_count_;
    double cf_mix_position_;
    double spike_cf_max_;
    std::array<double, 2> angle_min_deg_;
    std::array<double, 2> angle_max_deg_;
    std::array<double, 2> velocity_min_deg_;
    std::array<double, 2> velocity_max_deg_;
    std::array<double, 2> cf_angle_norm_deg_;
    std::array<double, 2> cf_velocity_norm_deg_;
    std::array<std::vector<int>, 2> cf_positive_neurons_;
    std::array<std::vector<int>, 2> cf_negative_neurons_;
    bool vertical_plane_;
    bool elbow_down_;
    bool enable_inverse_dynamics_;
    bool external_desired_state_enabled_;
    bool started_;

    // Separate RNG keeps stochastic CF feedback deterministic without touching other plant state.
    std::mt19937 error_feedback_rng_;
    OuterDynamicFeedbackProductEncoding state_feedback_product_;
    OuterDynamicFeedbackSingleEncoding state_feedback_single_;

    struct IKResult {
        Eigen::Vector2d q;
        bool reachable;
    };

    void BuildPlanarModel();
    void UpdateDesiredState(double time_seconds);
    void DecodeDcnTorques(int time, double step_seconds, Simulation* simulation);
    void IntegrateDynamics(double step_seconds);
    void EncodeAndEmitStateFeedback(int time, Simulation* simulation);
    void EncodeAndEmitErrorFeedback(int time, Simulation* simulation);
    void EmitBinaryCfSpikes(
        int seed,
        const std::vector<int>& target_neurons,
        int base_time,
        int window_steps,
        Simulation* simulation);
    int EncodeStateProductIndex(int joint) const;
    int EncodeUniformBin(double value, double min_value, double max_value, int bin_count) const;
    int EncodeProductIndex(const std::vector<int>& indices, const std::vector<int>& bins) const;
    void EmitSpike(Simulation* simulation, int neuron_id, int time);
    IKResult InverseKinematics2R(double x, double y) const;
    Eigen::Matrix2d Jacobian2R(const Eigen::Vector2d& q) const;
    Eigen::Matrix2d JacobianDot2R(const Eigen::Vector2d& q, const Eigen::Vector2d& qdot) const;
    Eigen::Vector2d DampedLeastSquaresSolve(const Eigen::Matrix2d& jacobian, const Eigen::Vector2d& rhs) const;

    static double RadToDeg(double radians);
    static double LinearSaturating(double max_abs, double value);
};

#endif
