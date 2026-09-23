#ifndef TIME_DRIVEN_LIF_EXPONENTIAL_TRIPLE_H
#define TIME_DRIVEN_LIF_EXPONENTIAL_TRIPLE_H

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_double.h"

#include <array>
#include <vector>

class TimeDrivenLIF_Exponential_triple : public TimeDrivenLIF_Exponential_double {
public:
    bool NMDA = false;

    float E_ampa = 0.0f;
    float ampa_tau = 10.0f;
    float E_gaba = -80.0f;
    float gaba_tau = 10.0f;
    float nmda_tau = 10.0f;

    const int N_NeuronStateVariables = 5;
    const int N_DifferentialStates = 1;
    const int index_V = 0;
    const int index_ampa = 1;
    const int index_gaba = 2;
    const int index_nmda = 3;
    const int I_EXT_index = 4;
    const int TimeDependentInputSize = 4;

    std::array<float, 5> init = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 5> sigma = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    TimeDrivenLIF_Exponential_triple();
    explicit TimeDrivenLIF_Exponential_triple(int timesteps);
    virtual ~TimeDrivenLIF_Exponential_triple();

    virtual void InitStateVector(int NumberOfNeurons, int GPUIndex);
    void CaculateDifferentialEquation(float* NeuronState, float* AuxNeuronState, int index);
    void CaculateTimeDependentEquation(float* NeuronState, int index, float dt);
    void CaculateSpike(float previous_V, float* NeuronState, int index);
    virtual InternalSpike* ProcessSpike(Interconnections* inter, int time);
    virtual void ProcessCurrent(Interconnections* inter, Neuron* Target, float current);
    virtual void CheckType(Interconnections* inter);
    virtual int get_NumberOfState();
    virtual bool compare(NeuronModel* neuralmodel);
    void SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep);
    virtual std::map<std::string, boost::any> getParameters();

private:
    std::vector<float> ampa_decay_lookup_;
    std::vector<float> gaba_decay_lookup_;
    std::vector<float> nmda_decay_lookup_;

    void ResetConductanceDecayLookup(float dt);
    void EnsureConductanceDecayLookupSize(float dt);
};

#endif
