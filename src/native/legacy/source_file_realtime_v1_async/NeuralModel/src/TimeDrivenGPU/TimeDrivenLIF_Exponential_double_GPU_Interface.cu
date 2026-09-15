#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Exponential_double_GPU_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector_Interface.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Exponential_double_GPU.cuh"
#include <iostream>
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "../source_file_realtime_v1_async/Current/inc/CurrentSynapse.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/error/cudaerror.h"
#include "../source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include "../source_file_realtime_v1_async/Intergration/inc/GPU_Method/IntegrationMethodGPU_Interface.cuh"
#include "../source_file_realtime_v1_async/ModelFactory/Integration_method_GPU_Interface_Facory.cuh"
#include "../source_file_realtime_v1_async/mainprogram/inc/BenchProfiling.h"
#include <cstdlib>
#include <vector>

namespace {

bool ForceExplicitGpuCopyPath() {
    const char* value = std::getenv("CEREBELLUM_FORCE_GPU_EXPLICIT_COPY");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

bool EnableGpuKernelDeviceTiming() {
    const char* value = std::getenv("CEREBELLUM_PROFILE_GPU_DEVICE_TIMING");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

bool PrintGpuLaunchConfig() {
    const char* value = std::getenv("CEREBELLUM_PRINT_GPU_LAUNCH_CONFIG");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

bool ForceGpuSyncAfterUpdate() {
    const char* value = std::getenv("CEREBELLUM_FORCE_GPU_SYNC_AFTER_UPDATE");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

}  // namespace

TimeDrivenLIF_Exponential_double_GPU_Interface::TimeDrivenLIF_Exponential_double_GPU_Interface(int timestep)
	: TimeDrivenNeuronModelGPU_Interface(timestep), CurrentSynapeModel(0), NeuronModelOnGPU(0), timestepdouble(0.0f) {
	this->StateVector = new Neuron_State_Vector_Interface(N_NeuronStateVariables);
	std::string name = std::string("TimeDrivenLIF_Exponential_double_GPU");
	this->setModelName(name);

}

TimeDrivenLIF_Exponential_double_GPU_Interface::~TimeDrivenLIF_Exponential_double_GPU_Interface() {
	if (this->CurrentSynapeModel != 0) {
		delete this->CurrentSynapeModel;
		this->CurrentSynapeModel = 0;
	}
	if (this->kernel_profile_start_event != nullptr) {
		cudaEventDestroy(this->kernel_profile_start_event);
		this->kernel_profile_start_event = nullptr;
	}
	if (this->kernel_profile_stop_event != nullptr) {
		cudaEventDestroy(this->kernel_profile_stop_event);
		this->kernel_profile_stop_event = nullptr;
	}
	if (this->update_profile_start_event != nullptr) {
		cudaEventDestroy(this->update_profile_start_event);
		this->update_profile_start_event = nullptr;
	}
	// 销毁 GPU 上的神经元模型
	this->DestroyGPUModel();
}

__global__ void DeleteClassGPU(TimeDrivenLIF_Exponential_double_GPU** GPU_NeuralModel) {
	// 使用单线程释放 GPU 上的神经元模型
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		delete (*GPU_NeuralModel);
	}
}

void TimeDrivenLIF_Exponential_double_GPU_Interface::DestroyGPUModel() {
	// 调用全局函数释放 GPU 上的神经元模型
	if (this->NeuronModelOnGPU != 0) {
        DeleteClassGPU <<<1, 1 >>> (this->NeuronModelOnGPU);
		cudaFree(this->NeuronModelOnGPU);
		this->NeuronModelOnGPU = 0;
	}

}

Neuron_State_Vector* TimeDrivenLIF_Exponential_double_GPU_Interface::InitState() {
	return this->StateVector;
}

InternalSpike* TimeDrivenLIF_Exponential_double_GPU_Interface::ProcessSpike(Interconnections* inter, int time) {
	// 根据突触类型把输入累加到对应电导槽位
	if (inter->type == 0 && this->Excited) {
		this->State_GPU->AuxStateCPU[0 * this->State_GPU->NumberofNeuron + inter->TargetNeuronModelIndex] += inter->weight;
	}
	else if (inter->type == 1 && this->Inhibitory) {
		this->State_GPU->AuxStateCPU[1 * this->State_GPU->NumberofNeuron + inter->TargetNeuronModelIndex] += inter->weight;
	}
    return 0;
}

void TimeDrivenLIF_Exponential_double_GPU_Interface::ProcessCurrent(Interconnections* inter, Neuron* Target, float current) {
	//设置Target传入突触的电流大小
	this->CurrentSynapeModel->SetInputCurrentPerSynapse(Target->index_in_NeuronModel, inter->subindex_type, current);
	//计算总电流
	float total_current = this->CurrentSynapeModel->GetTotalInputCurrentPerNeuron(Target->index_in_NeuronModel);
	//设置AuxStateCPU中的电流大小
	this->State_GPU->AuxStateCPU[2 * this->State_GPU->NumberofNeuron + Target->index_in_NeuronModel] = total_current;
}

//用全局函数封装GPU上的更新函数
__global__ void TimeDrivenLIF_Exponential_double_GPU_Interface_Update(TimeDrivenLIF_Exponential_double_GPU** NeuralModelGPU, int currenttime, float timestepsize) {
	(*NeuralModelGPU)->UpdateState(currenttime, timestepsize);
}

void TimeDrivenLIF_Exponential_double_GPU_Interface::UpdateState(int index, int time, Simulation* simulation) {
	const long long total_start = bench_profile::now_ns();
	auto* stateGPU = this->State_GPU;
	int neuronNum = stateGPU->NumberofNeuron;
	int timedepInput = this->N_TimedependentInput;
	size_t copySize = sizeof(float) * neuronNum * timedepInput;
	size_t zeroSize = sizeof(float) * neuronNum * (timedepInput - 1);
	const bool use_mapped_host_memory = this->deviceProp.canMapHostMemory && !ForceExplicitGpuCopyPath();
	const bool profile_kernel_device_time =
		EnableGpuKernelDeviceTiming() &&
		this->kernel_profile_start_event != nullptr &&
		this->kernel_profile_stop_event != nullptr;
	if (this->update_profile_start_event != nullptr) {
		HANDLE_ERROR(cudaEventRecord(this->update_profile_start_event, this->computeStream));
	}
	if (use_mapped_host_memory) {
		memset(this->State_GPU->InternalSpikeCPU, 0, sizeof(bool) * this->State_GPU->NumberofNeuron);
		const long long kernel_start = bench_profile::now_ns();
		if (profile_kernel_device_time) {
			HANDLE_ERROR(cudaEventRecord(this->kernel_profile_start_event, this->computeStream));
		}
		TimeDrivenLIF_Exponential_double_GPU_Interface_Update <<<this->gridsize, this->blocksize, 0, this->computeStream >>> (this->NeuronModelOnGPU, time, simulation->basetimesteps);
		if (profile_kernel_device_time) {
			HANDLE_ERROR(cudaEventRecord(this->kernel_profile_stop_event, this->computeStream));
		}
		bench_profile::gpu_update_kernel_ns.fetch_add(bench_profile::now_ns() - kernel_start, std::memory_order_relaxed);
	}
	else {
		const long long h2d_start = bench_profile::now_ns();
		HANDLE_ERROR(cudaMemcpyAsync(
			stateGPU->AuxStateGPU,
			stateGPU->AuxStateCPU,
			copySize,
			cudaMemcpyHostToDevice,
			this->computeStream));
		bench_profile::gpu_update_memcpy_h2d_ns.fetch_add(bench_profile::now_ns() - h2d_start, std::memory_order_relaxed);
		HANDLE_ERROR(cudaMemsetAsync(this->State_GPU->InternalSpikeGPU, 0, sizeof(bool) * this->State_GPU->NumberofNeuron, this->computeStream));
		const long long kernel_start = bench_profile::now_ns();
		if (profile_kernel_device_time) {
			HANDLE_ERROR(cudaEventRecord(this->kernel_profile_start_event, this->computeStream));
		}
		TimeDrivenLIF_Exponential_double_GPU_Interface_Update << <this->gridsize, this->blocksize, 0, this->computeStream >> > (this->NeuronModelOnGPU, time, simulation->basetimesteps);
		if (profile_kernel_device_time) {
			HANDLE_ERROR(cudaEventRecord(this->kernel_profile_stop_event, this->computeStream));
		}
		bench_profile::gpu_update_kernel_ns.fetch_add(bench_profile::now_ns() - kernel_start, std::memory_order_relaxed);
		const long long d2h_start = bench_profile::now_ns();
		HANDLE_ERROR(cudaMemcpyAsync(
			this->State_GPU->InternalSpikeCPU,
			this->State_GPU->InternalSpikeGPU,
			sizeof(bool) * this->State_GPU->NumberofNeuron,
			cudaMemcpyDeviceToHost,
			this->computeStream));
		bench_profile::gpu_update_d2h_ns.fetch_add(bench_profile::now_ns() - d2h_start, std::memory_order_relaxed);
	}
	if (this->StateVector->IsMonitored) {
		const long long monitor_d2h_start = bench_profile::now_ns();
		HANDLE_ERROR(cudaMemcpyAsync(
			this->State_GPU->Vector_of_StateVariable,
			this->State_GPU->Vector_of_StateVariableGPU,
			this->State_GPU->NumberofNeuron * this->State_GPU->NumberofStateVariable * sizeof(float),
			cudaMemcpyDeviceToHost,
			this->computeStream));
		HANDLE_ERROR(cudaMemcpyAsync(
			this->State_GPU->LastUpdate,
			this->State_GPU->LastUpdateGPU,
			this->State_GPU->NumberofNeuron * sizeof(int),
			cudaMemcpyDeviceToHost,
			this->computeStream));
		HANDLE_ERROR(cudaMemcpyAsync(
			this->State_GPU->LastSpike,
			this->State_GPU->LastSpikingGPU,
			this->State_GPU->NumberofNeuron * sizeof(int),
			cudaMemcpyDeviceToHost,
			this->computeStream));
		bench_profile::gpu_update_d2h_ns.fetch_add(bench_profile::now_ns() - monitor_d2h_start, std::memory_order_relaxed);
	}
	if (profile_kernel_device_time) {
		HANDLE_ERROR(cudaEventSynchronize(this->kernel_profile_stop_event));
		float kernel_elapsed_ms = 0.0f;
		HANDLE_ERROR(cudaEventElapsedTime(&kernel_elapsed_ms, this->kernel_profile_start_event, this->kernel_profile_stop_event));
		const long long kernel_elapsed_ns = static_cast<long long>(kernel_elapsed_ms * 1000000.0f);
		bench_profile::gpu_update_kernel_device_ns.fetch_add(kernel_elapsed_ns, std::memory_order_relaxed);
	}
	const long long record_start = bench_profile::now_ns();
	HANDLE_ERROR(cudaEventRecord(this->sync_event, this->computeStream));
	bench_profile::gpu_update_event_record_ns.fetch_add(bench_profile::now_ns() - record_start, std::memory_order_relaxed);
	const long long sync_start = bench_profile::now_ns();
	HANDLE_ERROR(cudaEventSynchronize(this->sync_event));
	bench_profile::gpu_update_sync_ns.fetch_add(bench_profile::now_ns() - sync_start, std::memory_order_relaxed);
	if (this->update_profile_start_event != nullptr) {
		float stream_elapsed_ms = 0.0f;
		HANDLE_ERROR(cudaEventElapsedTime(&stream_elapsed_ms, this->update_profile_start_event, this->sync_event));
		bench_profile::gpu_update_stream_device_ns.fetch_add(
			static_cast<long long>(stream_elapsed_ms * 1000000.0f),
			std::memory_order_relaxed);
	}
	const long long memset_start = bench_profile::now_ns();
	memset(stateGPU->AuxStateCPU, 0, zeroSize);
	bench_profile::gpu_update_memset_ns.fetch_add(bench_profile::now_ns() - memset_start, std::memory_order_relaxed);
	bench_profile::gpu_update_total_ns.fetch_add(bench_profile::now_ns() - total_start, std::memory_order_relaxed);
	bench_profile::gpu_update_count.fetch_add(1, std::memory_order_relaxed);
}

void TimeDrivenLIF_Exponential_double_GPU_Interface::InitStateVector(int NumberofNeuron, int GPUindex) {
	// 将状态数据同步到 GPU
	//this->GPU_ID = openmpindex % NumberofGPU;
	this->NumberOfNeuron = NumberofNeuron;
	this->GPU_ID = GPUindex % NumberOfGPU;
	HANDLE_ERROR(cudaSetDevice(GPUIndex[GPU_ID]));
	HANDLE_ERROR(cudaGetDeviceProperties(&this->deviceProp, GPUIndex[GPUindex % NumberOfGPU]));
	HANDLE_ERROR(cudaEventCreate(&this->sync_event));
	HANDLE_ERROR(cudaEventCreate(&this->kernel_profile_start_event));
	HANDLE_ERROR(cudaEventCreate(&this->kernel_profile_stop_event));
	HANDLE_ERROR(cudaEventCreate(&this->update_profile_start_event));
	HANDLE_ERROR(cudaStreamCreate(&copyStream));
	HANDLE_ERROR(cudaStreamCreate(&computeStream));
	//将State_GPU的指针指向StateVector（统一父类与子类的StateVector）
	this->State_GPU = (Neuron_State_Vector_Interface*) this->StateVector;
	//分配初始化向量 (电压，兴奋g，抑制g，电流)
	float new_init[] = { this->V_reset + this->init[0], 0.0, 0.0, 0.0};
	float new_sigma[] = { this->sigma[0], this->sigma[1], this->sigma[2], this->sigma[3] };
	//初始化接口状态向量
	this->State_GPU->InitNeuronStateGPU(NumberofNeuron, new_init, new_sigma, this->N_TimedependentInput, this->deviceProp);
	// 初始化 GPU 上的神经元模型
	this->InitializeClassGPU2(NumberofNeuron);
	// 初始化 GPU 上的状态向量
	InitializeVectorNeuronState_GPU2();
	this->CurrentSynapeModel = new CurrentSynapse(NumberofNeuron);
	// 初始化输入电流模型
	InitializeInputCurrentSynapseStructure();
}

__global__ void LIFTimeDrivenNeuralModel_GPU_C_Interface_InitializeClassGPU2(TimeDrivenLIF_Exponential_double_GPU** NeuralModelOnGPU, char* int_Name,
	float V_rest, float tau, float V_th, float R, float V_reset, int t_ref, float gexc_tau, float Exc,
	float ginh_tau, float Einhibitory, int Timestep, float timesteps, void** d_parm, int N_neurons) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		(*NeuralModelOnGPU) = new TimeDrivenLIF_Exponential_double_GPU(int_Name, V_rest, tau, V_th, R, V_reset, t_ref, gexc_tau, Exc, ginh_tau, Einhibitory, Timestep, timesteps, d_parm, N_neurons);
	}
}

void TimeDrivenLIF_Exponential_double_GPU_Interface::InitializeClassGPU2(int NumberofNeuron) {
	// 为 GPU 上的神经元模型分配内存
	HANDLE_ERROR(cudaMalloc(&(this->NeuronModelOnGPU), sizeof(TimeDrivenLIF_Exponential_double_GPU**)));
	std::vector<char> integrationNameHost(
		this->integrationMethodGPUInterface->name.begin(),
		this->integrationMethodGPUInterface->name.end());
	integrationNameHost.push_back('\0');
	char* integrationNameGPU = nullptr;
	HANDLE_ERROR(cudaMalloc((void**)&integrationNameGPU, integrationNameHost.size() * sizeof(char)));
	HANDLE_ERROR(cudaMemcpy(
		integrationNameGPU,
		integrationNameHost.data(),
		integrationNameHost.size() * sizeof(char),
		cudaMemcpyHostToDevice));
	// 计算 block size
	this->blocksize = 128;
	// 计算 grid size
	this->gridsize = this->deviceProp.multiProcessorCount * 16;
	if ((NumberofNeuron + this->blocksize - 1) / this->blocksize < this->gridsize) {
		this->gridsize = (NumberofNeuron + this->blocksize - 1) / this->blocksize;
	}
	// 统计总线程数
	int TotalThread = this->gridsize * this->blocksize;
	if (PrintGpuLaunchConfig()) {
		int active_blocks_per_sm = 0;
		HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&active_blocks_per_sm,
			TimeDrivenLIF_Exponential_double_GPU_Interface_Update,
			this->blocksize,
			0));
		const int theoretical_active_threads_per_sm = active_blocks_per_sm * this->blocksize;
		const double theoretical_occupancy =
			this->deviceProp.maxThreadsPerMultiProcessor > 0
				? static_cast<double>(theoretical_active_threads_per_sm) /
				      static_cast<double>(this->deviceProp.maxThreadsPerMultiProcessor)
				: 0.0;
		const int needed_blocks = (NumberofNeuron + this->blocksize - 1) / this->blocksize;
		std::cout
			<< "[gpu_lif_double_launch] gpu_id=" << this->GPU_ID
			<< " model_neurons=" << NumberofNeuron
			<< " block=" << this->blocksize
			<< " grid=" << this->gridsize
			<< " launch_threads=" << TotalThread
			<< " needed_blocks=" << needed_blocks
			<< " mp_count=" << this->deviceProp.multiProcessorCount
			<< " max_threads_per_sm=" << this->deviceProp.maxThreadsPerMultiProcessor
			<< " occ_blocks_per_sm=" << active_blocks_per_sm
			<< " occ_threads_per_sm=" << theoretical_active_threads_per_sm
			<< " theoretical_occupancy=" << theoretical_occupancy
			<< '\n';
	}
	// 为 GPU 端方法名字符串分配内存
	this->integrationMethodGPUInterface->InitIntegrationMethodOnGPU(this->NumberOfNeuron, TotalThread);
	// 调用全局函数释放 GPU 上的神经元模型
	LIFTimeDrivenNeuralModel_GPU_C_Interface_InitializeClassGPU2 <<<1, 1 >>> (this->NeuronModelOnGPU, integrationNameGPU, this->V_rest, this->tau, this->V_th, this->R, this->V_reset, this->t_ref, this->gexc_tau, this->Eexc, this->ginh_tau, this->Einhibitory, this->getTimestepSize(), this->timestepdouble, this->integrationMethodGPUInterface->d_param, this->NumberOfNeuron);
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());
	// 释放 GPU 端方法名字符串缓存
    HANDLE_ERROR(cudaFree(integrationNameGPU));

}

__global__ void initializeVectorNeuronState_GPU2(TimeDrivenLIF_Exponential_double_GPU** NeuralModelGPU, int NumberofVariable, float* InitState, float* AuxStateGPU, float* StateGPU, int* LastUpdateGPU, int* LastSpikeGPU, bool* InternalSpikeGPU, int NumberofNeurons) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		(*NeuralModelGPU)->InitializeNeuronStateVector(NumberofVariable, InitState, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeGPU, InternalSpikeGPU, NumberofNeurons);
	}
}

void TimeDrivenLIF_Exponential_double_GPU_Interface::InitializeVectorNeuronState_GPU2() {
	Neuron_State_Vector_Interface* State = (Neuron_State_Vector_Interface*) this->StateVector;
	// 调用全局函数初始化 GPU 上的状态向量
    initializeVectorNeuronState_GPU2 <<<1, 1 >>> (this->NeuronModelOnGPU,  State->NumberofStateVariable, State->InitialStateGPU, State->AuxStateGPU, State->Vector_of_StateVariableGPU, State->LastUpdateGPU, State->LastSpikingGPU, State->InternalSpikeGPU, State->NumberofNeuron);
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

}



__global__ void SetSynapseOnGPU(TimeDrivenLIF_Exponential_double_GPU** NeuralModelGPU, bool Exc, bool Inh, bool I_EXT) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		(*NeuralModelGPU)->SetEnabledSynapsis(Exc, Inh, I_EXT);
	}
}


void TimeDrivenLIF_Exponential_double_GPU_Interface::CheckType(Interconnections* inter) {
	int Type = inter->type;
	if (Type == 0 && this->Excited == false) {
		this->Excited = true;
		SetSynapseOnGPU<<<1, 1>>>(this->NeuronModelOnGPU, this->Excited, this->Inhibitory, this->I_EXT);
	}
	else if (Type == 1 && this->Inhibitory == false) {
		this->Inhibitory = true;
		SetSynapseOnGPU <<<1, 1 >>> (this->NeuronModelOnGPU, this->Excited, this->Inhibitory, this->I_EXT);
	}
	else if (Type == 3) {
		this->I_EXT = true;
		// 设置输入电流突触的子类型序号
		inter->subindex_type = this->CurrentSynapeModel->N_connections[inter->TargetNeuron->index_in_NeuronModel];
		// 增加每个神经元的输入电流突触计数
		this->CurrentSynapeModel->IncrementNInputCurrentSynapsesPerNeuron(inter->TargetNeuron->index_in_NeuronModel);
		SetSynapseOnGPU <<<1, 1 >>> (this->NeuronModelOnGPU, this->Excited, this->Inhibitory, this->I_EXT);
	}
	else if (Type == 2) {
		std::cout << "Error: NMDA is not supported in this model" << std::endl;
	}
	else if (Type == 0 && this->Excited == true) {
		return;
	}
	else if (Type == 1 && this->Inhibitory == true) {
		return;
	}
	else {
		std::cout << "Error: Type of synapse is not supported in this model" << std::endl;
	}
}


void TimeDrivenLIF_Exponential_double_GPU_Interface::InitializeInputCurrentSynapseStructure() {
	if (this->CurrentSynapeModel != 0) {
		this->CurrentSynapeModel->InitializeInputCurrentPerSynapseStructure();
	}
}


void TimeDrivenLIF_Exponential_double_GPU_Interface::SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep) {
	//在字典中查找参数
	//查找静息电位
	std::map<std::string, boost::any>::iterator iter = parametermap.find("V_rest");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->V_rest = new_parameter;
		parametermap.erase(iter);
	}
	// 写入 tau
	iter = parametermap.find("tau");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->tau = new_parameter;
		parametermap.erase(iter);
	}
	// 写入阈值
	iter = parametermap.find("V_th");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->V_th = new_parameter;
		parametermap.erase(iter);
	}
	// 写入输入电阻
	iter = parametermap.find("R");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->R = new_parameter;
		parametermap.erase(iter);
	}
	// 写入重置电位
	iter = parametermap.find("V_reset");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->V_reset = new_parameter;
		parametermap.erase(iter);
	}
	// 写入不应期
	iter = parametermap.find("t_ref");
	if (iter != parametermap.end()) {
		int new_parameter = boost::any_cast<int>(iter->second);
		this->t_ref = new_parameter;
		parametermap.erase(iter);
	}
	// 写入兴奋性电导时间常数
	iter = parametermap.find("gexc_tau");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->gexc_tau = new_parameter;
		parametermap.erase(iter);
	}
	// 写入兴奋性反转膜电位
	iter = parametermap.find("Eexc");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->Eexc = new_parameter;
		parametermap.erase(iter);
	}
	// 写入抑制性电导时间常数
	iter = parametermap.find("ginh_tau");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->ginh_tau = new_parameter;
		parametermap.erase(iter);
	}
	// 写入抑制性反转膜电位
	iter = parametermap.find("Einh");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->Einhibitory = new_parameter;
		parametermap.erase(iter);
	}

	iter = parametermap.find("basetimestep");
	if (iter != parametermap.end()) {
		float new_parameter = boost::any_cast<float>(iter->second);
		this->timestepdouble = new_parameter;
		parametermap.erase(iter);
	}

	// 写入积分方法参数
	iter = parametermap.find("random_mu");
	if (iter != parametermap.end()) {
		std::array<float, 4> new_parameter = boost::any_cast<std::array<float, 4>>(iter->second);
		this->init = new_parameter;
		parametermap.erase(iter);
	}

	iter = parametermap.find("random_sigma");
	if (iter != parametermap.end()) {
		std::array<float, 4> new_parameter = boost::any_cast<std::array<float, 4>>(iter->second);
		this->sigma = new_parameter;
		parametermap.erase(iter);
	}

	// 创建一段字符串缓存

	iter = parametermap.find("int_method");
	if (iter != parametermap.end()) {
		if (this->integrationMethodGPUInterface != 0) {
			delete this->integrationMethodGPUInterface;
		}
		ModelDescription temp = boost::any_cast<ModelDescription>(iter->second);
		temp.ModelParameter["step"] = basetimestep;
		// 为神经元模型创建字符串缓存
		this->integrationMethodGPUInterface = Integration_method_GPU_Interface_Factory<TimeDrivenLIF_Exponential_double_GPU_Interface>::create_Integration_method_GPU_Interface(temp, this);
		parametermap.erase(iter);

	}
	else {
		// 创建一个默认的前向欧拉法字符串
		ModelDescription temp_modelDescription;
		temp_modelDescription.ModelName = "ForwardEulerMethod";
		temp_modelDescription.ModelParameter["step"] = basetimestep;
		this->integrationMethodGPUInterface = Integration_method_GPU_Interface_Factory<TimeDrivenLIF_Exponential_double_GPU_Interface>::create_Integration_method_GPU_Interface(temp_modelDescription, this);
	}



}

std::map<std::string, boost::any> TimeDrivenLIF_Exponential_double_GPU_Interface::getParameters() {
	// 返回一个参数字典
	std::map<std::string, boost::any> parametermap;
	parametermap["V_rest"] = boost::any(this->V_rest);
	parametermap["tau"] = boost::any(this->tau);
	parametermap["V_th"] = boost::any(this->V_th);
	parametermap["R"] = boost::any(this->R);
	parametermap["V_reset"] = boost::any(this->V_reset);
	parametermap["t_ref"] = boost::any(this->t_ref);
	parametermap["gexc_tau"] = boost::any(this->gexc_tau);
	parametermap["Eexc"] = boost::any(this->Eexc);
	parametermap["ginh_tau"] = boost::any(this->ginh_tau);
	parametermap["Einh"] = boost::any(this->Einhibitory);
	parametermap["basetimestep"] = boost::any(this->timestepdouble);
	parametermap["random_mu"] = this->init;
	parametermap["random_sigma"] = this->sigma;
	ModelDescription temp;
	temp.ModelParameter = this->integrationMethodGPUInterface->getParameters();
	temp.ModelName = boost::any_cast<std::string>(temp.ModelParameter["name"]);
	parametermap["int_method"] = temp;
	return parametermap;

}
