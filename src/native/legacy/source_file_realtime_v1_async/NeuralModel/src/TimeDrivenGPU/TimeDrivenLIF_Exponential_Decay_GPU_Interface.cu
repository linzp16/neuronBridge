#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Exponential_Decay_GPU_Interface.cuh"

#include "../source_file_realtime_v1_async/Current/inc/CurrentSynapse.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Exponential_Decay_GPU.cuh"
#include "../source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "../source_file_realtime_v1_async/error/cudaerror.h"
#include "../source_file_realtime_v1_async/Intergration/inc/GPU_Method/IntegrationMethodGPU_Interface.cuh"
#include "../source_file_realtime_v1_async/ModelFactory/Integration_method_GPU_Interface_Facory.cuh"

#include <vector>

TimeDrivenLIF_Exponential_Decay_GPU_Interface::TimeDrivenLIF_Exponential_Decay_GPU_Interface(int timestep)
    : TimeDrivenNeuronModelGPU_Interface(timestep),
      CurrentSynapeModel(0),
      NeuronModelOnGPU(0),
      timestepdouble(0.0f) {
    this->StateVector = new Neuron_State_Vector_Interface(N_NeuronStateVariables);
    this->setModelName(std::string("TimeDrivenLIF_Exponential_Decay_GPU"));
}

TimeDrivenLIF_Exponential_Decay_GPU_Interface::~TimeDrivenLIF_Exponential_Decay_GPU_Interface() {
    if (this->CurrentSynapeModel != 0) {
        delete this->CurrentSynapeModel;
        this->CurrentSynapeModel = 0;
    }
    this->DestroyGPUModel();
}

__global__ void DeleteLifDecayClassGPU(TimeDrivenLIF_Exponential_Decay_GPU** GPU_NeuralModel) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        delete (*GPU_NeuralModel);
    }
}

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::DestroyGPUModel() {
    if (this->NeuronModelOnGPU != 0) {
        DeleteLifDecayClassGPU<<<1, 1>>>(this->NeuronModelOnGPU);
        cudaFree(this->NeuronModelOnGPU);
        this->NeuronModelOnGPU = 0;
    }
}

Neuron_State_Vector* TimeDrivenLIF_Exponential_Decay_GPU_Interface::InitState() {
    return this->StateVector;
}

InternalSpike* TimeDrivenLIF_Exponential_Decay_GPU_Interface::ProcessSpike(Interconnections* inter, int time) {
    (void)time;
    if (inter->type == 0 && this->Excited) {
        this->State_GPU->AuxStateCPU[inter->TargetNeuronModelIndex] += inter->weight;
    } else if (inter->type == 1 && this->Inhibitory) {
        this->State_GPU->AuxStateCPU[this->State_GPU->NumberofNeuron + inter->TargetNeuronModelIndex] += inter->weight;
    }
    return 0;
}

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::ProcessCurrent(Interconnections* inter, Neuron* Target, float current) {
    this->CurrentSynapeModel->SetInputCurrentPerSynapse(Target->index_in_NeuronModel, inter->subindex_type, current);
    const float total_current = this->CurrentSynapeModel->GetTotalInputCurrentPerNeuron(Target->index_in_NeuronModel);
    this->State_GPU->AuxStateCPU[2 * this->State_GPU->NumberofNeuron + Target->index_in_NeuronModel] = total_current;
}

__global__ void LifDecay_GPU_Interface_Update(TimeDrivenLIF_Exponential_Decay_GPU** NeuralModelGPU, int currenttime, float timestepsize) {
    (*NeuralModelGPU)->UpdateState(currenttime, timestepsize);
}

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::UpdateState(int index, int time, Simulation* simulation) {
    (void)index;
    Neuron_State_Vector_Interface* state = this->State_GPU;
    const int neuron_count = state->NumberofNeuron;
    const size_t copy_size = sizeof(float) * neuron_count * this->N_TimedependentInput;
    const size_t zero_size = sizeof(float) * neuron_count * (this->N_TimedependentInput - 1);
    HANDLE_ERROR(cudaMemcpyAsync(state->AuxStateGPU, state->AuxStateCPU, copy_size, cudaMemcpyHostToDevice, this->computeStream));
    HANDLE_ERROR(cudaMemsetAsync(state->InternalSpikeGPU, 0, sizeof(bool) * neuron_count, this->computeStream));
    LifDecay_GPU_Interface_Update<<<this->gridsize, this->blocksize, 0, this->computeStream>>>(
        this->NeuronModelOnGPU, time, simulation->basetimesteps);
    HANDLE_ERROR(cudaMemcpyAsync(state->InternalSpikeCPU, state->InternalSpikeGPU, sizeof(bool) * neuron_count, cudaMemcpyDeviceToHost, this->computeStream));
    if (this->StateVector->IsMonitored) {
        HANDLE_ERROR(cudaMemcpyAsync(
            state->Vector_of_StateVariable,
            state->Vector_of_StateVariableGPU,
            state->NumberofNeuron * state->NumberofStateVariable * sizeof(float),
            cudaMemcpyDeviceToHost,
            this->computeStream));
        HANDLE_ERROR(cudaMemcpyAsync(
            state->LastUpdate,
            state->LastUpdateGPU,
            state->NumberofNeuron * sizeof(int),
            cudaMemcpyDeviceToHost,
            this->computeStream));
        HANDLE_ERROR(cudaMemcpyAsync(
            state->LastSpike,
            state->LastSpikingGPU,
            state->NumberofNeuron * sizeof(int),
            cudaMemcpyDeviceToHost,
            this->computeStream));
    }
    HANDLE_ERROR(cudaEventRecord(this->sync_event, this->computeStream));
    HANDLE_ERROR(cudaEventSynchronize(this->sync_event));
    memset(state->AuxStateCPU, 0, zero_size);
}

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::InitStateVector(int NumberofNeuron, int GPUindex) {
    this->NumberOfNeuron = NumberofNeuron;
    this->GPU_ID = GPUindex % NumberOfGPU;
    HANDLE_ERROR(cudaSetDevice(GPUIndex[GPU_ID]));
    HANDLE_ERROR(cudaGetDeviceProperties(&this->deviceProp, GPUIndex[GPU_ID]));
    HANDLE_ERROR(cudaEventCreate(&this->sync_event));
    HANDLE_ERROR(cudaStreamCreate(&this->copyStream));
    HANDLE_ERROR(cudaStreamCreate(&this->computeStream));
    this->State_GPU = static_cast<Neuron_State_Vector_Interface*>(this->StateVector);
    float new_init[] = {this->V_rest + this->init[0], this->init[1], this->init[2]};
    float new_sigma[] = {this->sigma[0], this->sigma[1], this->sigma[2]};
    this->State_GPU->InitNeuronStateGPU(NumberofNeuron, new_init, new_sigma, this->N_TimedependentInput, this->deviceProp);
    this->InitializeClassGPU2(NumberofNeuron);
    this->InitializeVectorNeuronState_GPU2();
    this->CurrentSynapeModel = new CurrentSynapse(NumberofNeuron);
    this->InitializeInputCurrentSynapseStructure();
}

__global__ void InitializeLifDecayClassGPU(TimeDrivenLIF_Exponential_Decay_GPU** NeuralModelOnGPU,
                                          char* int_Name,
                                          float V_rest,
                                          float tau,
                                          float V_th,
                                          float R,
                                          float V_reset,
                                          int t_ref,
                                          float g_tau,
                                          float E,
                                          int Timestep,
                                          float timesteps,
                                          void** d_parm) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        (*NeuralModelOnGPU) = new TimeDrivenLIF_Exponential_Decay_GPU(
            int_Name, V_rest, tau, V_th, R, V_reset, t_ref, g_tau, E, Timestep, timesteps, d_parm);
    }
}

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::InitializeClassGPU2(int NumberofNeuron) {
    HANDLE_ERROR(cudaMalloc(&(this->NeuronModelOnGPU), sizeof(TimeDrivenLIF_Exponential_Decay_GPU**)));
    std::vector<char> integration_name_host(
        this->integrationMethodGPUInterface->name.begin(),
        this->integrationMethodGPUInterface->name.end());
    integration_name_host.push_back('\0');
    char* integration_name_gpu = nullptr;
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&integration_name_gpu), integration_name_host.size() * sizeof(char)));
    HANDLE_ERROR(cudaMemcpy(
        integration_name_gpu,
        integration_name_host.data(),
        integration_name_host.size() * sizeof(char),
        cudaMemcpyHostToDevice));
    this->blocksize = 128;
    this->gridsize = this->deviceProp.multiProcessorCount * 16;
    const int needed_blocks = (NumberofNeuron + this->blocksize - 1) / this->blocksize;
    if (needed_blocks < this->gridsize) {
        this->gridsize = needed_blocks;
    }
    const int total_thread = this->gridsize * this->blocksize;
    this->integrationMethodGPUInterface->InitIntegrationMethodOnGPU(this->NumberOfNeuron, total_thread);
    InitializeLifDecayClassGPU<<<1, 1>>>(
        this->NeuronModelOnGPU,
        integration_name_gpu,
        this->V_rest,
        this->tau,
        this->V_th,
        this->R,
        this->V_reset,
        this->t_ref,
        this->g_tau,
        this->E,
        this->getTimestepSize(),
        this->timestepdouble,
        this->integrationMethodGPUInterface->d_param);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaFree(integration_name_gpu));
}

__global__ void InitializeLifDecayVectorGPU(TimeDrivenLIF_Exponential_Decay_GPU** NeuralModelGPU,
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

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::InitializeVectorNeuronState_GPU2() {
    Neuron_State_Vector_Interface* state = static_cast<Neuron_State_Vector_Interface*>(this->StateVector);
    InitializeLifDecayVectorGPU<<<1, 1>>>(
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

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::InitializeInputCurrentSynapseStructure() {
    if (this->CurrentSynapeModel != 0) {
        this->CurrentSynapeModel->InitializeInputCurrentPerSynapseStructure();
    }
}

__global__ void SetLifDecaySynapseOnGPU(TimeDrivenLIF_Exponential_Decay_GPU** NeuralModelGPU, bool Exc, bool Inh, bool I_EXT) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        (*NeuralModelGPU)->SetEnabledSynapsis(Exc, Inh, I_EXT);
    }
}

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::CheckType(Interconnections* inter) {
    if (inter->type == 0) {
        this->Excited = true;
    } else if (inter->type == 1) {
        this->Inhibitory = true;
    } else if (inter->type == 3) {
        this->I_EXT = true;
        inter->subindex_type = this->CurrentSynapeModel->N_connections[inter->TargetNeuron->index_in_NeuronModel];
        this->CurrentSynapeModel->IncrementNInputCurrentSynapsesPerNeuron(inter->TargetNeuron->index_in_NeuronModel);
    }
    SetLifDecaySynapseOnGPU<<<1, 1>>>(this->NeuronModelOnGPU, this->Excited, this->Inhibitory, this->I_EXT);
}

void TimeDrivenLIF_Exponential_Decay_GPU_Interface::SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep) {
    std::map<std::string, boost::any>::iterator iter = parametermap.find("V_rest");
    if (iter != parametermap.end()) { this->V_rest = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("tau");
    if (iter != parametermap.end()) { this->tau = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("V_th");
    if (iter != parametermap.end()) { this->V_th = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("R");
    if (iter != parametermap.end()) { this->R = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("V_reset");
    if (iter != parametermap.end()) { this->V_reset = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("t_ref");
    if (iter != parametermap.end()) { this->t_ref = boost::any_cast<int>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("g_tau");
    if (iter != parametermap.end()) { this->g_tau = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("E");
    if (iter != parametermap.end()) { this->E = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("basetimestep");
    if (iter != parametermap.end()) { this->timestepdouble = boost::any_cast<float>(iter->second); parametermap.erase(iter); }
    iter = parametermap.find("int_method");
    if (iter != parametermap.end()) {
        if (this->integrationMethodGPUInterface != 0) {
            delete this->integrationMethodGPUInterface;
        }
        ModelDescription temp = boost::any_cast<ModelDescription>(iter->second);
        temp.ModelParameter["step"] = basetimestep;
        this->integrationMethodGPUInterface =
            Integration_method_GPU_Interface_Factory<TimeDrivenLIF_Exponential_Decay_GPU_Interface>::create_Integration_method_GPU_Interface(
                temp,
                this);
        parametermap.erase(iter);
    } else {
        ModelDescription temp_modelDescription;
        temp_modelDescription.ModelName = "ForwardEulerMethod";
        temp_modelDescription.ModelParameter["step"] = basetimestep;
        this->integrationMethodGPUInterface =
            Integration_method_GPU_Interface_Factory<TimeDrivenLIF_Exponential_Decay_GPU_Interface>::create_Integration_method_GPU_Interface(
                temp_modelDescription,
                this);
    }
}

std::map<std::string, boost::any> TimeDrivenLIF_Exponential_Decay_GPU_Interface::getParameters() {
    std::map<std::string, boost::any> parametermap;
    parametermap["V_rest"] = this->V_rest;
    parametermap["tau"] = this->tau;
    parametermap["V_th"] = this->V_th;
    parametermap["R"] = this->R;
    parametermap["V_reset"] = this->V_reset;
    parametermap["t_ref"] = this->t_ref;
    parametermap["g_tau"] = this->g_tau;
    parametermap["E"] = this->E;
    if (this->integrationMethodGPUInterface != 0) {
        ModelDescription temp;
        temp.ModelParameter = this->integrationMethodGPUInterface->getParameters();
        temp.ModelName = boost::any_cast<std::string>(temp.ModelParameter["name"]);
        parametermap["int_method"] = temp;
    }
    return parametermap;
}
