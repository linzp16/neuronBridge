#ifndef STRICT_MATLAB_PLANAR_ARM_2DOF_H
#define STRICT_MATLAB_PLANAR_ARM_2DOF_H

#include <array>

struct StrictMatlabPlanarArm2DOFConfig {
    double link1 = 0.34;
    double link2 = 0.32;
    double mass1 = 1.8;
    double mass2 = 1.6;
    double q1_min = -0.5235987755982988;
    double q1_max = 1.5707963267948966;
    double q2_min = 0.0017453292519943296;
    double q2_max = 2.6179938779914944;
};

struct StrictMatlabPlanarArm2DOFState {
    std::array<double, 2> q{ {0.0, 0.0} };
    std::array<double, 2> qd{ {0.0, 0.0} };
    std::array<double, 2> qdd{ {0.0, 0.0} };
    std::array<double, 2> q_des{ {0.0, 0.0} };
    std::array<double, 2> qd_des{ {0.0, 0.0} };
    std::array<double, 2> tau{ {0.0, 0.0} };
};

class StrictMatlabPlanarArm2DOF {
public:
    explicit StrictMatlabPlanarArm2DOF(const StrictMatlabPlanarArm2DOFConfig& config);

    void ResetState(const std::array<double, 2>& q, const std::array<double, 2>& qd);
    void SetDesiredState(const std::array<double, 2>& q_des, const std::array<double, 2>& qd_des);
    void SetTorqueCommand(const std::array<double, 2>& tau);
    void Step(double interval_s);

    const StrictMatlabPlanarArm2DOFState& GetState() const;
    std::array<double, 2> HandPosition() const;

private:
    StrictMatlabPlanarArm2DOFConfig config_;
    StrictMatlabPlanarArm2DOFState state_;

    std::array<double, 2> JointAcceleration(const std::array<double, 2>& tau) const;
    static double Clamp(double value, double min_value, double max_value);
};

#endif
