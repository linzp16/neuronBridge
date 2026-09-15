#ifndef TIME_DRIVEN_LIF_EXPONENTIAL_DECAY_GPU_CUH
#define TIME_DRIVEN_LIF_EXPONENTIAL_DECAY_GPU_CUH

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenNeuralModelGPU.cuh"
#include "../source_file_realtime_v1_async/ModelFactory/Integration_method_GPU_Factory.cuh"

class TimeDrivenLIF_Exponential_Decay_GPU : public TimeDrivenNeuralModelGPU {
public:
    float V_rest = -65.0f;
    float tau = 20.0f;
    float V_th = -20.0f;
    float R = 1.0f;
    float V_reset = -65.0f;
    int t_ref = 50;
    float g_tau = 12.0f;
    float E = 0.0f;

    const int N_NeuronStateVariables = 3;
    const int N_TimedependentInput = 3;
    const int N_DifferentialStates = 1;
    const int TimeDependentInputSize = 3;
    const int index_V = 0;
    const int index_g = 1;
    const int I_EXT_index = 2;

    bool Excited = false;
    bool Inhibitory = false;
    bool I_EXT = true;

    __device__ TimeDrivenLIF_Exponential_Decay_GPU(char* int_Name,
                                                   float newV_rest,
                                                   float newtau,
                                                   float newV_th,
                                                   float newR,
                                                   float newV_reset,
                                                   int newt_ref,
                                                   float newg_tau,
                                                   float newE,
                                                   int Timestep,
                                                   float timesteps,
                                                   void** parm)
        : TimeDrivenNeuralModelGPU(Timestep, timesteps),
          V_rest(newV_rest),
          tau(newtau),
          V_th(newV_th),
          R(newR),
          V_reset(newV_reset),
          t_ref(newt_ref),
          g_tau(newg_tau),
          E(newE) {
        // Use the same legacy GPU integration template path as the older LIF
        // GPU models. The model supplies equations; the integration object
        // owns stepping policy and temporary buffers.
        this->integration_method_on_GPU =
            Integration_method_GPU_Factory<TimeDrivenLIF_Exponential_Decay_GPU>::create_Integration_method_GPU(
                int_Name,
                parm,
                this);
        this->integration_method_on_GPU->Calculate_conductance_exp_values();
    }

    __device__ void SetEnabledSynapsis(bool new_excited, bool new_inhibitory, bool new_i_ext) {
        this->Excited = new_excited;
        this->Inhibitory = new_inhibitory;
        this->I_EXT = new_i_ext;
    }

    __device__ ~TimeDrivenLIF_Exponential_Decay_GPU() {
        delete this->integration_method_on_GPU;
    }

    __device__ void InitializeNeuronStateVector(int NumberofState,
                                                float* initState,
                                                float* AuxStateGPU,
                                                float* VectorNeuronStateGPU,
                                                int* LastUpdate,
                                                int* LastSpike,
                                                bool* InternalSpikeGPU,
                                                int NumberofNeuron) {
        this->Neuron_State_Vector = new Neuron_State_Vector_on_GPU(
            NumberofState,
            initState,
            AuxStateGPU,
            VectorNeuronStateGPU,
            LastUpdate,
            LastSpike,
            InternalSpikeGPU,
            NumberofNeuron);
    }

    __device__ virtual void UpdateState(int time, float timestep) {
        (void)timestep;
        const int neuron_count = this->Neuron_State_Vector->NumberofNeurons;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < neuron_count) {
            float* state = this->Neuron_State_Vector->Vector_of_State_VariableGPU;
            const float arrival_exc = this->Excited ? this->Neuron_State_Vector->AuxStateGPU[index] : 0.0f;
            const float arrival_inh = this->Inhibitory ? this->Neuron_State_Vector->AuxStateGPU[index + neuron_count] : 0.0f;
            const float input_current = this->I_EXT ? this->Neuron_State_Vector->AuxStateGPU[index + neuron_count * 2] : 0.0f;
            state[index + neuron_count] += arrival_exc - arrival_inh;
            state[index + neuron_count * 2] = input_current;
            this->Neuron_State_Vector->LastUpdateGPU[index] = time;
            index += blockDim.x * gridDim.x;
        }
        // The legacy integration template owns the numerical stepping; this
        // model only stages arrivals and provides the differential equations.
        this->integration_method_on_GPU->CaculateIncreament(neuron_count, this->Neuron_State_Vector->Vector_of_State_VariableGPU);
    }

    __device__ void Caculate_Conductance(int index, float elapsed_time) {
        this->Set_conductance_exp_values(index, 0, expf(-elapsed_time / this->g_tau));
        this->Set_conductance_exp_values(index, 1, 1.0f);
        this->Set_conductance_exp_values(index, 2, 1.0f);
    }

    __device__ void CaculateDifferentialEquation(int index,
                                                 int NumberofNeuron,
                                                 float* NeuronState,
                                                 float* AuxNeuronState,
                                                 float elapsetime) {
        (void)elapsetime;
        const float v = NeuronState[index];
        const float g = NeuronState[index + NumberofNeuron];
        const float input_current = NeuronState[index + NumberofNeuron * 2];
        const int gpu_offset = blockDim.x * blockIdx.x + threadIdx.x;
        if (this->Neuron_State_Vector->LastSpikeGPU[index] >= this->t_ref) {
            AuxNeuronState[gpu_offset] =
                (this->R * input_current + g * (this->E - v) + (this->V_rest - v)) / this->tau;
        } else {
            AuxNeuronState[gpu_offset] = 0.0f;
        }
    }

    __device__ void CaculateTimeDependentEquation(int index,
                                                  int NumberofNeuron,
                                                  float* NeuronState,
                                                  float elapsetime,
                                                  int elapsetimeindex) {
        (void)elapsetime;
        const float decay = this->Get_conductance_exponential_values(elapsetimeindex, 0);
        NeuronState[NumberofNeuron + index] *= decay;
    }

    __device__ void CaculateSpike(float previous_V, float* NeuronState, int index, float timestep) {
        (void)previous_V;
        (void)timestep;
        if (NeuronState[index] >= this->V_th) {
            NeuronState[index] = this->V_reset;
            this->Neuron_State_Vector->InternalSpikeGPU[index] = true;
            this->Neuron_State_Vector->LastSpikeGPU[index] = 0;
            this->integration_method_on_GPU->ResetState(index);
        }
    }
};

#endif
