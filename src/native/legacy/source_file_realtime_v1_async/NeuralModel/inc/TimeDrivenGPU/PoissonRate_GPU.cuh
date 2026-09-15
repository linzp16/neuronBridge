#ifndef POISSON_RATE_GPU_CUH
#define POISSON_RATE_GPU_CUH

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenNeuralModelGPU.cuh"

class PoissonRate_GPU : public TimeDrivenNeuralModelGPU {
public:
    float rate_bias_hz = 0.0f;
    float rate_gain_hz_per_current = 1.0f;

    const int N_NeuronStateVariables = 2;
    const int N_TimedependentInput = 3;
    const int index_rate_hz = 0;
    const int I_EXT_index = 1;

    bool Excited = false;
    bool Inhibitory = false;
    bool I_EXT = true;

    __device__ PoissonRate_GPU(float new_rate_bias_hz,
                               float new_rate_gain_hz_per_current,
                               int Timestep,
                               float timesteps)
        : TimeDrivenNeuralModelGPU(Timestep, timesteps),
          rate_bias_hz(new_rate_bias_hz),
          rate_gain_hz_per_current(new_rate_gain_hz_per_current) {
        this->integration_method_on_GPU = 0;
        this->N_conductance = 0;
        this->conductance_exp_values = 0;
    }

    __device__ void SetEnabledSynapsis(bool new_excited, bool new_inhibitory, bool new_i_ext) {
        this->Excited = new_excited;
        this->Inhibitory = new_inhibitory;
        this->I_EXT = new_i_ext;
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

    __device__ float HashUniform01(int neuron_index, int time_step) const {
        unsigned int x = static_cast<unsigned int>(neuron_index) * 747796405u +
                         static_cast<unsigned int>(time_step + 1) * 2891336453u +
                         0x9e3779b9u;
        x ^= x >> 16;
        x *= 2246822519u;
        x ^= x >> 13;
        x *= 3266489917u;
        x ^= x >> 16;
        return static_cast<float>(x & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
    }

    __device__ virtual void UpdateState(int time, float timestep) {
        const int neuron_count = this->Neuron_State_Vector->NumberofNeurons;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < neuron_count) {
            const float arrival_exc = this->Excited ? this->Neuron_State_Vector->AuxStateGPU[index] : 0.0f;
            const float arrival_inh = this->Inhibitory ? this->Neuron_State_Vector->AuxStateGPU[index + neuron_count] : 0.0f;
            const float input_current = this->I_EXT ? this->Neuron_State_Vector->AuxStateGPU[index + neuron_count * 2] : 0.0f;
            float rate_hz = this->rate_bias_hz +
                            this->rate_gain_hz_per_current * input_current +
                            arrival_exc - arrival_inh;
            if (rate_hz < 0.0f) {
                rate_hz = 0.0f;
            }
            this->Neuron_State_Vector->Vector_of_State_VariableGPU[index] = rate_hz;
            const float probability = fminf(1.0f, rate_hz * timestep * 0.001f);
            if (HashUniform01(index, time) < probability) {
                this->Neuron_State_Vector->InternalSpikeGPU[index] = true;
                this->Neuron_State_Vector->LastSpikeGPU[index] = 0;
            } else {
                this->Neuron_State_Vector->LastSpikeGPU[index] += 1;
            }
            this->Neuron_State_Vector->LastUpdateGPU[index] = time;
            index += blockDim.x * gridDim.x;
        }
    }
};

#endif
