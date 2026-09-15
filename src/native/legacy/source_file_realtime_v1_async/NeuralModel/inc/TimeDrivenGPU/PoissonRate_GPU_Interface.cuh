#ifndef POISSON_RATE_GPU_INTERFACE_CUH
#define POISSON_RATE_GPU_INTERFACE_CUH

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenNeuronModelGPU_Interface.cuh"

class CurrentSynapse;
class PoissonRate_GPU;

class PoissonRate_GPU_Interface : public TimeDrivenNeuronModelGPU_Interface {
public:
    bool Excited = false;
    bool Inhibitory = false;
    bool I_EXT = true;

    float rate_bias_hz = 0.0f;
    float rate_gain_hz_per_current = 1.0f;

    const int N_NeuronStateVariables = 2;
    const int N_TimedependentInput = 3;
    const int index_rate_hz = 0;
    const int I_EXT_index = 1;
    const int TimeDependentInputSize = 3;

    CurrentSynapse* CurrentSynapeModel;
    PoissonRate_GPU** NeuronModelOnGPU;
    float timestepdouble;

    explicit PoissonRate_GPU_Interface(int timestep);
    ~PoissonRate_GPU_Interface();

    virtual void DestroyGPUModel();
    virtual Neuron_State_Vector* InitState();
    virtual InternalSpike* ProcessSpike(Interconnections* inter, int time);
    virtual void ProcessCurrent(Interconnections* inter, Neuron* Target, float current);
    virtual void UpdateState(int index, int time, Simulation* simulation);
    virtual void InitStateVector(int NumberOfNeurons, int GPUindex);
    virtual void InitializeClassGPU2(int N_neurons);
    virtual void InitializeVectorNeuronState_GPU2();
    virtual void InitializeInputCurrentSynapseStructure();
    virtual void CheckType(Interconnections* inter);
    void SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep);
    virtual std::map<std::string, boost::any> getParameters();

    virtual int getV_index() { return this->index_rate_hz; }
    virtual int get_NumberOfState() { return this->N_NeuronStateVariables; }
    virtual enum NeuronModelType getNeuronModelType() { return NEURAL_LAYER; }
};

#endif
