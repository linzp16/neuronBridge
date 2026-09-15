#ifndef POISSON_RATE_H
#define POISSON_RATE_H

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenModel.h"

class CurrentSynapse;

class PoissonRate : public TimeDrivenModel {
public:
    // The CPU Poisson model keeps a single per-neuron rate state. Spike input
    // and current input are interpreted as transient rate contributions.
    float rate_bias_hz = 0.0f;
    float rate_gain_hz_per_current = 1.0f;
    const int N_NeuronStateVariables = 2;
    const int index_rate_hz = 0;
    const int I_EXT_index = 1;

    CurrentSynapse* CurrentSynapeModel;

    PoissonRate();
    explicit PoissonRate(int timesteps);
    ~PoissonRate();

    virtual void InitStateVector(int NumberOfNeurons, int GPUIndex);
    virtual Neuron_State_Vector* InitState();
    virtual void UpdateState(int index, int time, Simulation* simulation);
    virtual InternalSpike* ProcessSpike(Interconnections* inter, int time);
    virtual void ProcessCurrent(Interconnections* inter, Neuron* Target, float current);
    virtual void InitializeInputCurrentSynapseStructure();
    virtual void CheckType(Interconnections* inter);
    virtual int getV_index();
    virtual int get_NumberOfState();
    virtual enum NeuronModelType getNeuronModelType();
    void SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep);
    virtual std::map<std::string, boost::any> getParameters();
};

#endif
