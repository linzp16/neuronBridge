#ifndef TIME_DRIVEN_LIF_EXPONENTIAL_TRIPLE_H
#define TIME_DRIVEN_LIF_EXPONENTIAL_TRIPLE_H

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_double.h"

class TimeDrivenLIF_Exponential_triple : public TimeDrivenLIF_Exponential_double {
public:
    // Thin CPU implementation for the public catalog path. It reuses the
    // mature double-conductance CPU integrator while preserving triple-model
    // parameter names for user declarations and save/load metadata.
    float E_ampa = 0.0f;
    float ampa_tau = 10.0f;
    float E_gaba = -80.0f;
    float gaba_tau = 10.0f;
    float nmda_tau = 10.0f;

    TimeDrivenLIF_Exponential_triple();
    explicit TimeDrivenLIF_Exponential_triple(int timesteps);

    virtual void CheckType(Interconnections* inter);
    void SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep);
    virtual std::map<std::string, boost::any> getParameters();
};

#endif
