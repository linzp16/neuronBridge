#ifndef TIME_DRIVEN_LIF_VOLTAGE_JUMP_GPU_INTERFACE_CUH
#define TIME_DRIVEN_LIF_VOLTAGE_JUMP_GPU_INTERFACE_CUH

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenNeuronModelGPU_Interface.cuh"
#include <array>

class CurrentSynapse;
class TimeDrivenLIF_Voltage_jump_GPU;

class TimeDrivenLIF_Voltage_jump_GPU_Interface : public TimeDrivenNeuronModelGPU_Interface {
public:
    bool Excited = false;
    bool Inhibitory = false;
    bool I_EXT = true;

    float V_rest = -65.0f;
    float tau = 20.0f;
    float V_th = -50.0f;
    float R = 1.0f;
    float V_reset = -65.0f;
    int t_ref = 50;

    const int N_NeuronStateVariables = 2;
    const int N_TimedependentInput = 3;
    const int index_V = 0;
    const int I_EXT_index = 1;
    const int TimeDependentInputSize = 3;

    CurrentSynapse* CurrentSynapeModel;
    TimeDrivenLIF_Voltage_jump_GPU** NeuronModelOnGPU;
    float timestepdouble;
    std::array<float, 2> init = {0.0f, 0.0f};
    std::array<float, 2> sigma = {0.0f, 0.0f};

    explicit TimeDrivenLIF_Voltage_jump_GPU_Interface(int timestep);
    ~TimeDrivenLIF_Voltage_jump_GPU_Interface();

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

    virtual int getV_index() { return this->index_V; }
    virtual int get_NumberOfState() { return this->N_NeuronStateVariables; }
    virtual enum NeuronModelType getNeuronModelType() { return NEURAL_LAYER; }
};

#endif
