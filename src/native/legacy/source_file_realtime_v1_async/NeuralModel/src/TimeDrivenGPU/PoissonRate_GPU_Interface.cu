#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/PoissonRate_GPU_Interface.cuh"

#include "../source_file_realtime_v1_async/Current/inc/CurrentSynapse.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/PoissonRate_GPU.cuh"
#include "../source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "../source_file_realtime_v1_async/error/cudaerror.h"

PoissonRate_GPU_Interface::PoissonRate_GPU_Interface(int timestep)
    : TimeDrivenNeuronModelGPU_Interface(timestep),
      CurrentSynapeModel(0),
      NeuronModelOnGPU(0),
      timestepdouble(0.0f) {
    this->StateVector = new Neuron_State_Vector_Interface(N_NeuronStateVariables);
    this->setModelName(std::string("PoissonRate_GPU"));
}

PoissonRate_GPU_Interface::~PoissonRate_GPU_Interface() {
    if (this->CurrentSynapeModel != 0) {
        delete this->CurrentSynapeModel;
        this->CurrentSynapeModel = 0;
    }
    this->DestroyGPUModel();
}

__global__ void DeletePoissonRateClassGPU(PoissonRate_GPU** GPU_NeuralModel) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        delete (*GPU_NeuralModel);
    }
}

void PoissonRate_GPU_Interface::DestroyGPUModel() {
    if (this->NeuronModelOnGPU != 0) {
        DeletePoissonRateClassGPU<<<1, 1>>>(this->NeuronModelOnGPU);
        cudaFree(this->NeuronModelOnGPU);
        this->NeuronModelOnGPU = 0;
    }
}

Neuron_State_Vector* PoissonRate_GPU_Interface::InitState() {
    return this->StateVector;
}

InternalSpike* PoissonRate_GPU_Interface::ProcessSpike(Interconnections* inter, int time) {
    (void)time;
    if (inter->type == 0 && this->Excited) {
        this->State_GPU->AuxStateCPU[inter->TargetNeuronModelIndex] += inter->weight;
    } else if (inter->type == 1 && this->Inhibitory) {
        this->State_GPU->AuxStateCPU[this->State_GPU->NumberofNeuron + inter->TargetNeuronModelIndex] += inter->weight;
    }
    return 0;
}

void PoissonRate_GPU_Interface::ProcessCurrent(Interconnections* inter, Neuron* Target, float current) {
    this->CurrentSynapeModel->SetInputCurrentPerSynapse(Target->index_in_NeuronModel, inter->subindex_type, current);
    const float total_current = this->CurrentSynapeModel->GetTotalInputCurrentPerNeuron(Target->index_in_NeuronModel);
    this->State_GPU->AuxStateCPU[2 * this->State_GPU->NumberofNeuron + Target->index_in_NeuronModel] = total_current;
}

__global__ void PoissonRate_GPU_Interface_Update(PoissonRate_GPU** NeuralModelGPU, int currenttime, float timestepsize) {
    (*NeuralModelGPU)->UpdateState(currenttime, timestepsize);
}

void PoissonRate_GPU_Interface::UpdateState(int index, int time, Simulation* simulation) {
    (void)index;
    Neuron_State_Vector_Interface* state = this->State_GPU;
    const int neuron_count = state->NumberofNeuron;
    const size_t copy_size = sizeof(float) * neuron_count * this->N_TimedependentInput;
    const size_t zero_size = sizeof(float) * neuron_count * (this->N_TimedependentInput - 1);
    HANDLE_ERROR(cudaMemcpyAsync(state->AuxStateGPU, state->AuxStateCPU, copy_size, cudaMemcpyHostToDevice, this->computeStream));
    HANDLE_ERROR(cudaMemsetAsync(state->InternalSpikeGPU, 0, sizeof(bool) * neuron_count, this->computeStream));
    PoissonRate_GPU_Interface_Update<<<this->gridsize, this->blocksize, 0, this->computeStream>>>(
        this->NeuronModelOnGPU, time, simulation->basetimesteps);
    HANDLE_ERROR(cudaMemcpyAsync(state->InternalSpikeCPU, state->InternalSpikeGPU, sizeof(bool) * neuron_count, cudaMemcpyDeviceToHost, this->computeStream));
    HANDLE_ERROR(cudaEventRecord(this->sync_event, this->computeStream));
    HANDLE_ERROR(cudaEventSynchronize(this->sync_event));
    memset(state->AuxStateCPU, 0, zero_size);
}

void PoissonRate_GPU_Interface::InitStateVector(int NumberofNeuron, int GPUindex) {
    this->NumberOfNeuron = NumberofNeuron;
    this->GPU_ID = GPUindex % NumberOfGPU;
    HANDLE_ERROR(cudaSetDevice(GPUIndex[GPU_ID]));
    HANDLE_ERROR(cudaGetDeviceProperties(&this->deviceProp, GPUIndex[GPU_ID]));
    HANDLE_ERROR(cudaEventCreate(&this->sync_event));
    HANDLE_ERROR(cudaStreamCreate(&this->copyStream));
    HANDLE_ERROR(cudaStreamCreate(&this->computeStream));
    this->State_GPU = static_cast<Neuron_State_Vector_Interface*>(this->StateVector);
    float init[] = {0.0f, 0.0f};
    float sigma[] = {0.0f, 0.0f};
    this->State_GPU->InitNeuronStateGPU(NumberofNeuron, init, sigma, this->N_TimedependentInput, this->deviceProp);
    this->InitializeClassGPU2(NumberofNeuron);
    this->InitializeVectorNeuronState_GPU2();
    this->CurrentSynapeModel = new CurrentSynapse(NumberofNeuron);
    this->InitializeInputCurrentSynapseStructure();
}

__global__ void InitializePoissonRateClassGPU(PoissonRate_GPU** NeuralModelOnGPU,
                                             float rate_bias_hz,
                                             float rate_gain_hz_per_current,
                                             int Timestep,
                                             float timesteps) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        (*NeuralModelOnGPU) = new PoissonRate_GPU(rate_bias_hz, rate_gain_hz_per_current, Timestep, timesteps);
    }
}

void PoissonRate_GPU_Interface::InitializeClassGPU2(int NumberofNeuron) {
    HANDLE_ERROR(cudaMalloc(&(this->NeuronModelOnGPU), sizeof(PoissonRate_GPU**)));
    this->blocksize = 128;
    this->gridsize = this->deviceProp.multiProcessorCount * 16;
    const int needed_blocks = (NumberofNeuron + this->blocksize - 1) / this->blocksize;
    if (needed_blocks < this->gridsize) {
        this->gridsize = needed_blocks;
    }
    InitializePoissonRateClassGPU<<<1, 1>>>(
        this->NeuronModelOnGPU,
        this->rate_bias_hz,
        this->rate_gain_hz_per_current,
        this->getTimestepSize(),
        this->timestepdouble);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}

__global__ void InitializePoissonRateVectorGPU(PoissonRate_GPU** NeuralModelGPU,
                                              int NumberofVariable,
                                              float* InitState,
                                              float* AuxStateGPU,
                                              float* StateGPU,
                                              int* LastUpdateGPU,
                                              int* LastSpikeGPU,
                                              bool* InternalSpikeGPU,
                                              int NumberofNeurons) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        (*NeuralModelGPU)->InitializeNeuronStateVector(
            NumberofVariable,
            InitState,
            AuxStateGPU,
            StateGPU,
            LastUpdateGPU,
            LastSpikeGPU,
            InternalSpikeGPU,
            NumberofNeurons);
    }
}

void PoissonRate_GPU_Interface::InitializeVectorNeuronState_GPU2() {
    Neuron_State_Vector_Interface* state = static_cast<Neuron_State_Vector_Interface*>(this->StateVector);
    InitializePoissonRateVectorGPU<<<1, 1>>>(
        this->NeuronModelOnGPU,
        state->NumberofStateVariable,
        state->InitialStateGPU,
        state->AuxStateGPU,
        state->Vector_of_StateVariableGPU,
        state->LastUpdateGPU,
        state->LastSpikingGPU,
        state->InternalSpikeGPU,
        state->NumberofNeuron);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}

void PoissonRate_GPU_Interface::InitializeInputCurrentSynapseStructure() {
    if (this->CurrentSynapeModel != 0) {
        this->CurrentSynapeModel->InitializeInputCurrentPerSynapseStructure();
    }
}

__global__ void SetPoissonRateSynapseOnGPU(PoissonRate_GPU** NeuralModelGPU, bool Exc, bool Inh, bool I_EXT) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        (*NeuralModelGPU)->SetEnabledSynapsis(Exc, Inh, I_EXT);
    }
}

void PoissonRate_GPU_Interface::CheckType(Interconnections* inter) {
    if (inter->type == 0) {
        this->Excited = true;
    } else if (inter->type == 1) {
        this->Inhibitory = true;
    } else if (inter->type == 3) {
        this->I_EXT = true;
        inter->subindex_type = this->CurrentSynapeModel->N_connections[inter->TargetNeuron->index_in_NeuronModel];
        this->CurrentSynapeModel->IncrementNInputCurrentSynapsesPerNeuron(inter->TargetNeuron->index_in_NeuronModel);
    }
    SetPoissonRateSynapseOnGPU<<<1, 1>>>(this->NeuronModelOnGPU, this->Excited, this->Inhibitory, this->I_EXT);
}

void PoissonRate_GPU_Interface::SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep) {
    std::map<std::string, boost::any>::iterator iter = parametermap.find("rate_bias_hz");
    if (iter != parametermap.end()) {
        this->rate_bias_hz = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
    iter = parametermap.find("poisson_rate_bias_hz");
    if (iter != parametermap.end()) {
        this->rate_bias_hz = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
    iter = parametermap.find("rate_gain_hz_per_current");
    if (iter != parametermap.end()) {
        this->rate_gain_hz_per_current = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
    iter = parametermap.find("poisson_rate_gain_hz_per_current");
    if (iter != parametermap.end()) {
        this->rate_gain_hz_per_current = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
}

std::map<std::string, boost::any> PoissonRate_GPU_Interface::getParameters() {
    std::map<std::string, boost::any> parametermap;
    parametermap["rate_bias_hz"] = this->rate_bias_hz;
    parametermap["rate_gain_hz_per_current"] = this->rate_gain_hz_per_current;
    return parametermap;
}
