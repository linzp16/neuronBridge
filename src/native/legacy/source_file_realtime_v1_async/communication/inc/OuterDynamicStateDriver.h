/*
 * OuterDynamicStateDriver.h
 *
 * Abstract output sink for state samples produced by an outer-dynamics model,
 * such as a two-joint plant coupled to the spiking network.
 */
#ifndef OUTERDYNAMICSTATEDRIVER_H
#define OUTERDYNAMICSTATEDRIVER_H

#include <vector>

struct OuterDynamicJointState {
    // Joint positions.
    std::vector<double> q{0.0, 0.0};
    // Joint velocities.
    std::vector<double> qv{0.0, 0.0};
    // Joint accelerations.
    std::vector<double> qdd{0.0, 0.0};
    // Desired joint positions.
    std::vector<double> q_des{0.0, 0.0};
    // Desired joint velocities.
    std::vector<double> qv_des{0.0, 0.0};
    // Total torque command applied to each joint.
    std::vector<double> tau_total{0.0, 0.0};
};

class OuterDynamicStateDriver {
public:
    virtual ~OuterDynamicStateDriver();

    // Writes one outer-dynamics joint-state sample.
    virtual void WriteJointState(
        int time_step,
        float base_timestep_ms,
        const OuterDynamicJointState& state) = 0;

    // Flushes buffered state output to the backing sink.
    virtual void FlushBuffers() = 0;
};

#endif
