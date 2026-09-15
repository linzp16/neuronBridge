/*
 * OuterDynamicStateDriver.h
 *
 * Abstract output sink for state samples produced by an outer-dynamics model,
 * such as a two-joint plant coupled to the spiking network.
 */
#ifndef OUTERDYNAMICSTATEDRIVER_H
#define OUTERDYNAMICSTATEDRIVER_H

#include <array>

struct OuterDynamicJointState {
    // Joint positions.
    std::array<double, 2> q{ {0.0, 0.0} };
    // Joint velocities.
    std::array<double, 2> qv{ {0.0, 0.0} };
    // Joint accelerations.
    std::array<double, 2> qdd{ {0.0, 0.0} };
    // Desired joint positions.
    std::array<double, 2> q_des{ {0.0, 0.0} };
    // Desired joint velocities.
    std::array<double, 2> qv_des{ {0.0, 0.0} };
    // Total torque command applied to each joint.
    std::array<double, 2> tau_total{ {0.0, 0.0} };
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
