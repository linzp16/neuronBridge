#ifndef STRICT_MATLAB_PLANAR_ARM_2DOF_OUTER_DYNAMIC_H
#define STRICT_MATLAB_PLANAR_ARM_2DOF_OUTER_DYNAMIC_H

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"
#include "../source_file_realtime_v1_async/OuterDynamic/inc/StrictMatlabPlanarArm2DOFPinocchioPlant.h"

#include <array>
#include <memory>
#include <random>
#include <vector>

class Simulation;

class StrictMatlabPlanarArm2DOFOuterDynamic : public OuterDynamicModel {
public:
    explicit StrictMatlabPlanarArm2DOFOuterDynamic(int timestep_size);
    virtual ~StrictMatlabPlanarArm2DOFOuterDynamic();

    virtual void Initialize(const OuterDynamicDescription& description, Simulation* simulation);
    virtual void Update(int time, Simulation* simulation);
    virtual int GetJointCount() const { return 2; }
    virtual void AccumulateInputSpike(int joint_id, int type, float weight, int time);
    virtual void ClearAccumulatedInputs();

    void ResetState(const std::array<double, 2>& q, const std::array<double, 2>& qd);
    void SetDesiredState(const std::array<double, 2>& q_des, const std::array<double, 2>& qd_des);
    void SetDesiredHandTarget(const std::array<double, 2>& xy_des, const std::array<double, 2>& xy_dot_des);

private:
    enum DesiredMode {
        JointSpaceDesired = 0,
        CartesianDesired = 1
    };

    struct JointMapping {
        std::vector<int> dcn_positive_neurons;
        std::vector<int> dcn_negative_neurons;
    };

    std::array<JointMapping, 2> joint_mappings_;
    std::unique_ptr<StrictMatlabPlanarArm2DOFPinocchioPlant> arm_;
    std::array<double, 2> torque_scale_;
    std::array<double, 2> angle_min_deg_;
    std::array<double, 2> angle_max_deg_;
    std::array<double, 2> velocity_min_deg_;
    std::array<double, 2> velocity_max_deg_;
    std::array<double, 2> desired_q_;
    std::array<double, 2> desired_qd_;
    std::array<double, 2> desired_xy_;
    std::array<double, 2> desired_xy_dot_;
    std::array<double, 2> accumulated_joint_torque_;
    StrictIkOptions ik_options_;
    DesiredMode desired_mode_;
    int spike_retention_steps_;
    int state_feedback_delay_steps_;
    int state_feedback_repeats_;
    int state_feedback_repeat_period_steps_;
    int error_feedback_delay_steps_;
    int error_feedback_window_steps_;
    int error_feedback_sample_count_;
    double cf_mix_position_;
    double spike_cf_max_;
    std::array<double, 2> cf_angle_norm_deg_;
    std::array<double, 2> cf_velocity_norm_deg_;
    std::array<std::vector<int>, 2> cf_positive_neurons_;
    std::array<std::vector<int>, 2> cf_negative_neurons_;
    std::mt19937 error_feedback_rng_;
    bool started_;
    OuterDynamicFeedbackProductEncoding state_feedback_product_;
    OuterDynamicFeedbackSingleEncoding state_feedback_single_;

    StrictMatlabPlanarArm2DOFPinocchioPlant& Arm();
    const StrictMatlabPlanarArm2DOFPinocchioPlant& Arm() const;
    void DecodeDcnTorqueWindow(int time, int window_steps, Simulation* simulation, std::array<double, 2>& tau) const;
    void EncodeAndEmitStateFeedback(int time, Simulation* simulation, const StrictMatlabPlanarArm2DOFState& arm_state);
    void EncodeAndEmitErrorFeedback(int time, Simulation* simulation, const StrictMatlabPlanarArm2DOFState& arm_state);
    void EmitBinaryCfSpikes(int seed,
                            const std::vector<int>& target_neurons,
                            int base_time,
                            int window_steps,
                            Simulation* simulation);
    int EncodeStateProductIndex(int joint, const StrictMatlabPlanarArm2DOFState& arm_state) const;
    int EncodeUniformBin(double value, double min_value, double max_value, int bin_count) const;
    int EncodeProductIndex(const std::vector<int>& indices, const std::vector<int>& bins) const;
    void EmitSpike(Simulation* simulation, int neuron_id, int time);
};

#endif
