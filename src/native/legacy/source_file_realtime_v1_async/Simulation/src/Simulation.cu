#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "debug_monitor/SimulationDebugMonitor.h"
#include "simulation_dense/SimulationCommonHost.h"
#include "simulation_dense/DenseSubnetworkModel.h"
#include "../source_file_realtime_v1_async/EventQueue/inc/TimingWheelEventQueue.h"
#include "../source_file_realtime_v1_async/Event/inc/EndSimulationTime.h"
#include "../source_file_realtime_v1_async/Event/inc/TimeEventUpdateNeuron.h"
#include "../source_file_realtime_v1_async/InputSpikeDriver/inc/ArrayInputSpikeDriver.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuronModel.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include "../source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include "../source_file_realtime_v1_async/Event/inc/Synchronize/SynchronizeActivityEvent.h"
#include "../source_file_realtime_v1_async/communication/inc/OutputSpikeDriver.h"
#include "../source_file_realtime_v1_async/Event/inc/Spike/Spike.h"
#include <cuda_runtime.h>
#include "../source_file_realtime_v1_async/error/cudaerror.h"
#include "../source_file_realtime_v1_async/Event/inc/Synchronize/SaveWeightEvent.h"
#include "simulation_dense/SimulationWeightIO.h"
#include "../source_file_realtime_v1_async/communication/inc/DriverType.h"
#include "../source_file_realtime_v1_async/Event/inc/Synchronize/CommunicationEvent.h"
#include "../source_file_realtime_v1_async/Event/inc/Outer/OuterUpdateEvent.h"
#include "../source_file_realtime_v1_async/Event/inc/InputConv/UpdateInputConvEvent.h"
#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeBuffer.h"
#include "../source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"
#include "../source_file_realtime_v1_async/InputConv/inc/InputConvFrameQueue.h"
#include "../source_file_realtime_v1_async/ModelFactory/InputConvModelFactory.h"
#include "../source_file_realtime_v1_async/ModelFactory/OuterDynamicModelFactory.h"
#include "../source_file_realtime_v1_async/mainprogram/inc/BenchProfiling.h"
#if SNN_WITH_ZMQ
#include "../source_file_realtime_v1_async/communication/inc/ZMQInputOutputSpikeDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/ZMQAsyncInputOutputSpikeDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/ZMQInputConvFrameDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/ZMQAsyncInputConvFrameDriver.h"
#endif

namespace {

bool HasDenseMonitorTargets(const Simulation* simulation) {
	if (simulation == NULL) {
		return false;
	}
	for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
		if (simulation->dense_subnetworks[index] != NULL &&
			simulation->dense_subnetworks[index]->HasMonitorTargets()) {
			return true;
		}
	}
	return false;
}

}  // namespace

void Simulation::AddExternalSpikeActivity(const std::vector<int>& EventTime, const std::vector<int>& EventNeuronID) {
	npgr::sim_support::AddExternalSpikeActivityTranslated(this, EventTime, EventNeuronID);
}

void Simulation::AddExternalCurrentActivity(const std::vector<int>& EventTime, const std::vector<int>& EventNeuronID, const std::vector<float>& Current) {
	npgr::sim_support::AddExternalCurrentActivityTranslated(this, EventTime, EventNeuronID, Current);
}



void Simulation::EndSimulation(int QueueID) {
	this->SimulationEnd[QueueID] = true;
}


Simulation::Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list, const std::list<ConnectionDescription>& connection_list, const std::list<LearningRuleDescription>& learning_rule_list, int simulationsteps, float timestep, int NumberOfQueue, EventQueueType eventQueueType, int timingWheelSize)
	: Simulation(neuron_layer_list, connection_list, learning_rule_list, std::list<OuterDynamicDescription>(), std::list<InputConvDescription>(), simulationsteps, timestep, NumberOfQueue, eventQueueType, timingWheelSize) {}

Simulation::Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list, const std::list<ConnectionDescription>& connection_list, const std::list<LearningRuleDescription>& learning_rule_list, const std::list<OuterDynamicDescription>& outer_dynamic_list, int simulationsteps, float timestep, int NumberOfQueue, EventQueueType eventQueueType, int timingWheelSize) :
	Simulation(neuron_layer_list, connection_list, learning_rule_list, outer_dynamic_list, std::list<InputConvDescription>(), simulationsteps, timestep, NumberOfQueue, eventQueueType, timingWheelSize) {}

Simulation::Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list, const std::list<ConnectionDescription>& connection_list, const std::list<LearningRuleDescription>& learning_rule_list, const std::list<OuterDynamicDescription>& outer_dynamic_list, const std::list<OuterDynamicConnectionDescription>& outer_dynamic_connection_list, int simulationsteps, float timestep, int NumberOfQueue, EventQueueType eventQueueType, int timingWheelSize) :
	Simulation(neuron_layer_list, connection_list, learning_rule_list, outer_dynamic_list, outer_dynamic_connection_list, std::list<InputConvDescription>(), simulationsteps, timestep, NumberOfQueue, eventQueueType, timingWheelSize) {}

Simulation::Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list, const std::list<ConnectionDescription>& connection_list, const std::list<LearningRuleDescription>& learning_rule_list, const std::list<OuterDynamicDescription>& outer_dynamic_list, const std::list<InputConvDescription>& input_conv_list, int simulationsteps, float timestep, int NumberOfQueue, EventQueueType eventQueueType, int timingWheelSize) :
	Simulation(neuron_layer_list, connection_list, learning_rule_list, outer_dynamic_list, std::list<OuterDynamicConnectionDescription>(), input_conv_list, simulationsteps, timestep, NumberOfQueue, eventQueueType, timingWheelSize) {}

Simulation::Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list, const std::list<ConnectionDescription>& connection_list, const std::list<LearningRuleDescription>& learning_rule_list, const std::list<OuterDynamicDescription>& outer_dynamic_list, const std::list<OuterDynamicConnectionDescription>& outer_dynamic_connection_list, const std::list<InputConvDescription>& input_conv_list, int simulationsteps, float timestep, int NumberOfQueue, EventQueueType eventQueueType, int timingWheelSize) :
	network(0), EventHeap(0), totaltimesteps(simulationsteps), basetimesteps(timestep), currenttime(0), NumberOfQueue(0), SimulationEnd(0), inputSpikeDriver(0), inputCurrentDriver(0), output_spike_driver(0), file_output_spike_driver(0), file_outer_dynamic_state_driver(0), SyncThread(0), PauseThread(0), DelayMin(-1), file_output_weight_driver(0), zmq_input_output_spike_driver(0), zmq_async_input_output_spike_driver(0), WeightSaveInterval(0), CommunivationInterval(0), neuronMonitorExist(false), RealTimeRestrictionObject(0), RealtimeEnabled(false), RealtimeSlotSteps(0), PrintSimulationTime(false), eventQueueType(eventQueueType), timingWheelSize(timingWheelSize), drivers_initialized(false), outer_dynamic_spike_buffer(0), latest_outer_dynamic_state(), latest_outer_dynamic_time_step(0), has_latest_outer_dynamic_state(false), debug_monitor(0), debug_monitor_config()
{
	// 采用公用构造仿真
	npgr::sim_support::ConstructDenseAwareSimulation(
		this,
		neuron_layer_list,
		connection_list,
		learning_rule_list,
		outer_dynamic_list,
		outer_dynamic_connection_list,
		input_conv_list,
		NumberOfQueue);
}

Simulation::~Simulation() {
	if (this->debug_monitor != NULL) {
		this->debug_monitor->Flush(NULL);
		delete this->debug_monitor;
		this->debug_monitor = NULL;
	}
	if (this->network != NULL) {
		delete this->network;
	}
	if (this->EventHeap != NULL) {
		delete this->EventHeap;
	}
	if (this->inputSpikeDriver != NULL) {
		delete this->inputSpikeDriver;
	}
	if (this->inputCurrentDriver != NULL) {
		delete this->inputCurrentDriver;
	}
	if (this->output_spike_driver != NULL) {
		delete this->output_spike_driver;
	}
	if (this->file_output_spike_driver != NULL) {
		delete this->file_output_spike_driver;
	}
	if (this->file_outer_dynamic_state_driver != NULL) {
		delete this->file_outer_dynamic_state_driver;
	}
	if (this->file_output_weight_driver != NULL) {
		delete this->file_output_weight_driver;
	}
	for (std::size_t index = 0; index < this->OuterDynamicModelList.size(); ++index) {
		delete this->OuterDynamicModelList[index];
	}
	this->OuterDynamicModelList.clear();
	for (std::size_t index = 0; index < this->InputConvModelList.size(); ++index) {
		delete this->InputConvModelList[index];
	}
	this->InputConvModelList.clear();
	for (std::size_t index = 0; index < this->InputConvFrameSourceList.size(); ++index) {
		delete this->InputConvFrameSourceList[index];
	}
	this->InputConvFrameSourceList.clear();
	this->InputConvFrameSourceIndexByName.clear();
	this->InputConvFrameBindings.clear();
	if (this->outer_dynamic_spike_buffer != NULL) {
		delete this->outer_dynamic_spike_buffer;
	}
	#if SNN_WITH_ZMQ
	if (this->zmq_input_output_spike_driver != NULL) {
		delete this->zmq_input_output_spike_driver;
	}
	if (this->zmq_async_input_output_spike_driver != NULL) {
		delete this->zmq_async_input_output_spike_driver;
	}
	#endif
	if (this->RealTimeRestrictionObject != NULL) {
		delete this->RealTimeRestrictionObject;
	}
	if (this->currenttime != NULL) {
		delete[] this->currenttime;
	}
	if (this->SimulationEnd != NULL) {
		delete[] this->SimulationEnd;
	}
	if (this->PauseThread != NULL) {
		delete[] this->PauseThread;
	}
	if (this->SyncThread != NULL) {
		delete[] this->SyncThread;
	}
	for (std::size_t index = 0; index < this->dense_subnetworks.size(); ++index) {
		delete this->dense_subnetworks[index];
	}
	this->dense_subnetworks.clear();

	this->InputSpikeDriverList.clear();
	this->InputCurrentDriverList.clear();
	this->OutputSpikeDriverList.clear();
	this->WeightDriverList.clear();
	this->OuterDynamicStateDriverList.clear();

}

void Simulation::CreateDenseSubnetworks(const std::list<NeuronLayerDescription>& neuron_layer_list) {
	(void)neuron_layer_list;
}

void Simulation::ResetDenseSubnetworks() {
	npgr::sim_support::ResetDenseSubnetworks(this);
}

void Simulation::CreateInputConv(const std::list<InputConvDescription>& input_conv_list) {
	this->InputConvDescriptionList.clear();
	this->InputConvMainCurrentSourceNeuronIds.clear();
	this->InputConvFrameBindings.clear();
	for (std::list<InputConvDescription>::const_iterator it = input_conv_list.begin(); it != input_conv_list.end(); ++it) {
		InputConvModel* model = InputConvModelFactory::createInputConvModel(*it);
		if (model != NULL) {
			const int inputconv_index = static_cast<int>(this->InputConvModelList.size());
			model->setInputConvIndex(inputconv_index);
			model->Initialize(*it, this);
			this->InputConvModelList.push_back(model);
			this->InputConvDescriptionList.push_back(*it);
			this->InputConvMainCurrentSourceNeuronIds.push_back(std::vector<int>());
			this->InputConvFrameBindings.push_back(InputConvFrameSourceBinding());
		}
	}
}

void Simulation::ResetInputConvModels() {
	for (std::size_t index = 0; index < this->InputConvModelList.size(); ++index) {
		if (this->InputConvModelList[index] != NULL) {
			this->InputConvModelList[index]->Reset(this);
		}
	}
}

void Simulation::FlushInputConvMainNetworkCurrent(InputConvModel* model, int time_step) {
	npgr::sim_support::FlushInputConvMainNetworkCurrent(this, model, time_step);
}

bool Simulation::BindDenseSubnetworkInterfaceCurrentDeviceSource(int subnetwork_index,
                                                                 const float* device_current,
                                                                 int interface_count,
                                                                 std::string* reason) {
	return npgr::sim_support::BindDenseSubnetworkInterfaceCurrentDeviceSource(
		this,
		subnetwork_index,
		device_current,
		interface_count,
		reason);
}

bool Simulation::IsDenseManagedModel(const NeuronModel* model) const {
	return npgr::sim_support::IsDenseManagedModel(this, model);
}

int Simulation::TranslateOriginalNeuronIdToMain(int original_neuron_id) const {
	return npgr::sim_support::TranslateOriginalNeuronIdToMain(this, original_neuron_id);
}

int Simulation::GetDenseSubnetworkCount() const {
	return npgr::sim_support::GetDenseSubnetworkCount(this);
}

bool Simulation::GetDenseSubnetworkName(int subnetwork_index, std::string& name) const {
	return npgr::sim_support::GetDenseSubnetworkName(this, subnetwork_index, name);
}

int Simulation::FindDenseSubnetworkByName(const std::string& name) const {
	return npgr::sim_support::FindDenseSubnetworkByName(this, name);
}

bool Simulation::GetDenseSubnetworkDebugSnapshot(int subnetwork_index, npgr::DenseSubnetworkDebugSnapshot& snapshot) const {
	return npgr::sim_support::GetDenseSubnetworkDebugSnapshot(this, subnetwork_index, snapshot);
}

bool Simulation::GetDenseSubnetworkWeights(int subnetwork_index, std::vector<float>& weights) const {
	return npgr::sim_support::GetDenseSubnetworkWeights(this, subnetwork_index, weights);
}

bool Simulation::ResetDenseSubnetwork(int subnetwork_index) {
	return npgr::sim_support::ResetDenseSubnetwork(this, subnetwork_index);
}

void Simulation::EnableRealtime(int slotSteps, double maxAdvanceSeconds, float firstSection, float secondSection, float thirdSection) {
	// 开启实时限制
	this->RealtimeEnabled = true;
	//插入第一个Communication事件
	this->RealtimeSlotSteps = (slotSteps > 0) ? slotSteps : ((this->CommunivationInterval > 0) ? this->CommunivationInterval : 1);
	//转化为实时处理的秒数
	const double slotSeconds = static_cast<double>(this->RealtimeSlotSteps) * static_cast<double>(this->basetimesteps)* 0.001;
	// 设置看门狗参数
	this->RealTimeRestrictionObject->SetParameterWatchDog(slotSeconds, maxAdvanceSeconds, firstSection, secondSection, thirdSection);
	this->RealTimeRestrictionObject->SetSleepPeriod(0.01);
}

void Simulation::DisableRealtime() {
	this->RealtimeEnabled = false;
	this->RealtimeSlotSteps = 0;
	if (this->RealTimeRestrictionObject != NULL) {
		this->RealTimeRestrictionObject->StopWatchDog();
	}
}

void Simulation::SetPrintSimulationTime(bool enable) {
	this->PrintSimulationTime = enable;
}

void Simulation::RecreateEventQueue() {
	npgr::sim_support::RecreateEventQueue(this);
}

void Simulation::BindDriversOnce() {
	npgr::sim_support::BindDriversOnce(this);
}

void Simulation::ScheduleInitialEvents() {
	npgr::sim_support::ScheduleInitialEvents(this);
}

void Simulation::ResetForNextRound(bool preserve_weights) {
	npgr::sim_support::ResetForNextRound(this, preserve_weights);
}




void Simulation::InitSimulation() {
	npgr::sim_support::BindDriversOnce(this);
	npgr::sim_support::ResetForNextRound(this, true);
	if (this->debug_monitor_config.enabled ||
		(this->network != NULL && this->network->isMonitor) ||
		HasDenseMonitorTargets(this)) {
		std::string reason;
		if (!this->EnableDebugMonitor(this->debug_monitor_config, &reason)) {
			std::cerr << "EnableDebugMonitor failed: " << reason << std::endl;
		}
	}
}

void Simulation::RunSimulationStep(int runsteps) {
	Event* newEvent;
	int time;
	int OpenmpIndex; //线程索引

    #pragma omp parallel num_threads(this->NumberOfQueue) if(this->NumberOfQueue > 1) \
    private(newEvent, OpenmpIndex, time)
	{
        const long long thread_setup_start_ns = bench_profile::now_ns();
		OpenmpIndex = omp_get_thread_num();
		time = 0.0;
        bench_profile::run_step_parallel_setup_ns.fetch_add(
            bench_profile::now_ns() - thread_setup_start_ns, std::memory_order_relaxed);
		if (NumberOfGPU > 0) {
            const long long set_gpu_start_ns = bench_profile::now_ns();
			this->SetOpenMPThreadToGPU(OpenmpIndex);
            bench_profile::run_step_set_gpu_thread_ns.fetch_add(
                bench_profile::now_ns() - set_gpu_start_ns, std::memory_order_relaxed);
		}
		//初始化EventHeap信号灯变量
		this->SimulationEnd[OpenmpIndex] = false;
		// 插入相对当前仿真时刻的结束事件，而不是重复使用绝对时间 runsteps。
		const long long end_event_insert_start_ns = bench_profile::now_ns();
		this->EventHeap->Insert_a_Event(
			new EndSimulationTime(this->currenttime[OpenmpIndex] + runsteps, OpenmpIndex),
			OpenmpIndex);
		bench_profile::run_step_end_event_insert_ns.fetch_add(
			bench_profile::now_ns() - end_event_insert_start_ns, std::memory_order_relaxed);
		//循环执行事件
		while (!this->SimulationEnd[OpenmpIndex]) {
			//判断是否需要同步
			if (this->SyncThread[OpenmpIndex]) {
				//重置线程同步索引
				this->SyncThread[OpenmpIndex] = false;
				//执行线程同步操作
				this->SynchronizeThread();
			}



			//从堆顶获取一个事件
			const long long remove_event_start_ns = bench_profile::now_ns();
			newEvent = this->EventHeap->Remove_a_Event(OpenmpIndex);
			bench_profile::run_step_remove_event_ns.fetch_add(
				bench_profile::now_ns() - remove_event_start_ns, std::memory_order_relaxed);
			if (newEvent == NULL) {
				std::cout << "Internal error: empty event queue on thread " << OpenmpIndex << std::endl;
				this->SimulationEnd[OpenmpIndex] = true;
				continue;
			}
			bench_profile::run_step_event_count.fetch_add(1, std::memory_order_relaxed);

			bool skipEvent = false;

			//判断事件是否需要跳过
			if (newEvent->getTime() < this->currenttime[OpenmpIndex]) {
				skipEvent = true;
				std::cout << "Internal error: Bad spike time. Spike: " << newEvent->getTime()*this->basetimesteps << " Current: " << this->currenttime[OpenmpIndex] << std::endl;
			}
			else {
				this->currenttime[OpenmpIndex] = newEvent->getTime();
				if (OpenmpIndex == 0 && this->currenttime[OpenmpIndex] > time) {
					time = this->currenttime[OpenmpIndex];
					if (this->PrintSimulationTime) {
						std::cout << "[SimulationStep] time = "
							<< static_cast<double>(time) * static_cast<double>(this->basetimesteps)
							<< " ms" << std::endl;
					}
				}
			}

			if (!skipEvent) {
				const long long process_event_start_ns = bench_profile::now_ns();
				newEvent->ProcessEvent(this);
				bench_profile::run_step_process_event_ns.fetch_add(
					bench_profile::now_ns() - process_event_start_ns, std::memory_order_relaxed);

			}

			const long long delete_event_start_ns = bench_profile::now_ns();
			delete newEvent;
			bench_profile::run_step_delete_event_ns.fetch_add(
				bench_profile::now_ns() - delete_event_start_ns, std::memory_order_relaxed);
			newEvent = NULL;
		}

	}
	const long long monitor_capture_start_ns = bench_profile::now_ns();
	this->CaptureDebugMonitorStep(this->currenttime != NULL ? this->currenttime[0] : 0, NULL);
	bench_profile::run_step_monitor_capture_ns.fetch_add(
		bench_profile::now_ns() - monitor_capture_start_ns, std::memory_order_relaxed);
}

void Simulation::RunSimulationRealtime(int runsteps) {
	if (!this->RealtimeEnabled || this->RealTimeRestrictionObject == NULL || this->RealtimeSlotSteps <= 0) {
		this->RunSimulationStep(runsteps);
		return;
	}
	Event* newEvent;
	int OpenmpIndex;
	int time;
	const int realtimeSlotSteps = this->RealtimeSlotSteps;
	// 定义看门狗线程	std::thread watchdogThread;
	// 阻塞等待看门狗线程启动	std::atomic<bool> watchdogStarted(false);

#pragma omp parallel num_threads(this->NumberOfQueue) if(this->NumberOfQueue > 1) private(newEvent, OpenmpIndex, time)
	{
		OpenmpIndex = omp_get_thread_num();
		int nextRealtimeBoundary = realtimeSlotSteps;
		time = 0;
		if (NumberOfGPU > 0) {
			this->SetOpenMPThreadToGPU(OpenmpIndex);
		}
		//初始化线程信号灯变量
		this->SimulationEnd[OpenmpIndex] = false;
		// 插入相对当前仿真时刻的结束事件，而不是重复使用绝对时间 runsteps。
		this->EventHeap->Insert_a_Event(
			new EndSimulationTime(this->currenttime[OpenmpIndex] + runsteps, OpenmpIndex),
			OpenmpIndex);
		//循环执行事件
		while (!this->SimulationEnd[OpenmpIndex]) {
			//获取约束等级
			RealTimeRestrictionLevel restrictionLevel = this->RealTimeRestrictionObject->GetRestrictionLevel();
			if (this->SyncThread[OpenmpIndex]) {
				this->SyncThread[OpenmpIndex] = false;
				this->SynchronizeThread(restrictionLevel);
				restrictionLevel = this->RealTimeRestrictionObject->GetRestrictionLevel();
			}
	//逐一在InputSpikeDriverList中加载输入
			newEvent = this->EventHeap->Remove_a_Event(OpenmpIndex);
			if (newEvent == NULL) {
				this->SimulationEnd[OpenmpIndex] = true;
				continue;
			}
			//判断事件是否合法
			bool skipEvent = false;
			if (newEvent->getTime() < this->currenttime[OpenmpIndex]) {
				skipEvent = true;
				std::cout << "Internal error: Bad spike time. Spike: " << newEvent->getTime() << " Current: " << this->currenttime[OpenmpIndex] << std::endl;
			}//打印当前时间
			else {
				this->currenttime[OpenmpIndex] = newEvent->getTime();
				if (OpenmpIndex == 0 && this->currenttime[OpenmpIndex] > time) {
					time = this->currenttime[OpenmpIndex];
					if (this->PrintSimulationTime) {
						std::cout << "[SimulationRealtime] time = "
							<< static_cast<double>(time) * static_cast<double>(this->basetimesteps)
							<< " ms" << std::endl;
					}
				}
			}

			if (OpenmpIndex == 0) {
				while (this->currenttime[0] >= nextRealtimeBoundary) {
					//判断是否需要启动看门狗线程
					if (!watchdogStarted.load()) {
						this->RealTimeRestrictionObject->StartWatchDog();
						watchdogThread = std::thread([this]() {
							this->RealTimeRestrictionObject->Watchdog();
						});
						//标记看门狗已启动
						watchdogStarted.store(true);
					}
					this->RealTimeRestrictionObject->NextStepWatchDog();
					nextRealtimeBoundary += realtimeSlotSteps;
				}
			}

			while (watchdogStarted.load() &&
				this->RealTimeRestrictionObject->GetRestrictionLevel() == SIMULATION_TOO_FAST) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}

			if (!skipEvent) {
				//获取约束等级
				restrictionLevel = this->RealTimeRestrictionObject->GetRestrictionLevel();
				newEvent->ProcessEvent(this, restrictionLevel);
			}

			delete newEvent;
			newEvent = NULL;
		}
	}

	if (watchdogStarted.load()) {
		this->RealTimeRestrictionObject->StopWatchDog();
	}
	if (watchdogThread.joinable()) {
		watchdogThread.join();
	}
	this->CaptureDebugMonitorStep(this->currenttime != NULL ? this->currenttime[0] : 0, NULL);
}

void Simulation::SynchronizeThread() {
    const long long start_ns = bench_profile::now_ns();
	//定义一个屏障同步指令
#pragma omp barrier
//定义单线程执行指令
#pragma omp single
	{
		if (!this->PauseThread[0] && !this->SimulationEnd[0]) {
			Event* event = this->EventHeap->Remove_a_syn_Event();
			if (event != NULL) {
				event->ProcessEvent(this);
				delete event;
			}
		}
	}

    bench_profile::sync_ns.fetch_add(bench_profile::now_ns() - start_ns, std::memory_order_relaxed);
    bench_profile::sync_count.fetch_add(1, std::memory_order_relaxed);
}

void Simulation::SynchronizeThread(RealTimeRestrictionLevel restrictionLevel) {
    const long long start_ns = bench_profile::now_ns();
#pragma omp barrier
#pragma omp single
	{
		if (!this->PauseThread[0] && !this->SimulationEnd[0]) {
			Event* event = this->EventHeap->Remove_a_syn_Event();
			if (event != NULL) {
				event->ProcessEvent(this, restrictionLevel);
				delete event;
			}
		}
	}
    bench_profile::sync_ns.fetch_add(bench_profile::now_ns() - start_ns, std::memory_order_relaxed);
    bench_profile::sync_count.fetch_add(1, std::memory_order_relaxed);
}


void Simulation::WriteSpike(Spike* spike) {
	Neuron* neuron = spike->SourceNeuron;
	//判断是否为输出神经元
	if (neuron->IsOutput) {
		if (this->outer_dynamic_spike_buffer != NULL) {
			this->outer_dynamic_spike_buffer->RecordSpike(spike->getTime(), neuron->Neuron_index);
		}
		for (std::list<OutputSpikeDriver*>::iterator it = this->OutputSpikeDriverList.begin(); it != this->OutputSpikeDriverList.end(); it++) {
			
			(*it)->WriteSpike(spike, this->basetimesteps);
			
		}
	}

}

void Simulation::SetOpenMPThreadToGPU(int threadNum) {
	if (NumberOfGPU > 0) {
		HANDLE_ERROR(cudaSetDevice(GPUIndex[threadNum % NumberOfGPU]));
	}
}

int Simulation::GetWeightSaveStep() {
	return this->WeightSaveInterval;
}

void Simulation::AddInputSpikeDriver(InputSpikeDriver* newInputSpikeDriver) {
	this->InputSpikeDriverList.push_back(newInputSpikeDriver);
}

void Simulation::AddOutputSpikeDriver(OutputSpikeDriver* newOutputSpikeDriver) {
	this->OutputSpikeDriverList.push_back(newOutputSpikeDriver);
}

void Simulation::AddInputCurrentDriver(InputCurrentDriver* newInputCurrentDriver) {
	this->InputCurrentDriverList.push_back(newInputCurrentDriver);
}

void Simulation::AddOutputWeightDriver(OutputWeightDriver* WeightDriver) {
	this->WeightDriverList.push_back(WeightDriver);
}

void Simulation::AddOuterDynamicStateDriver(OuterDynamicStateDriver* driver) {
	this->OuterDynamicStateDriverList.push_back(driver);
}

void Simulation::AddFileOutputSpikeDriver(std::string filename) {
	this->file_output_spike_driver = new FileOutputSpikeDriver(filename.c_str());
}

void Simulation::AddFileOuterDynamicStateDriver(std::string filename) {
	this->file_outer_dynamic_state_driver = new FileOuterDynamicStateDriver(filename.c_str());
}

void Simulation::AddFileOutputWeightDriver(std::string FileName, int WeightSaveInterval) {
	this->file_output_weight_driver = new FileOutputWeightDriver(FileName.c_str());
	this->WeightSaveInterval = WeightSaveInterval;
}

void Simulation::AddZMQInputOutputSpikeDriver(enum DriverType driverType, std::string address, unsigned short port, int communicationInterval) {
#if SNN_WITH_ZMQ
	this->zmq_input_output_spike_driver = new ZMQInputOutputSpikeDriver(driverType, address, port);
	this->CommunivationInterval = communicationInterval;
	this->InputSpikeDriverList.push_back(this->zmq_input_output_spike_driver);
	this->OutputSpikeDriverList.push_back(this->zmq_input_output_spike_driver);
#else
	(void) driverType;
	(void) address;
	(void) port;
	(void) communicationInterval;
	std::cout << "ZMQ support is disabled in this build." << std::endl;
#endif

}

void Simulation::AddZMQAsyncInputOutputSpikeDriver(std::string subscribeAddress, unsigned short publishPort, unsigned short subscribePort, std::string publishTopic, std::string subscribeTopic, int communicationInterval) {
#if SNN_WITH_ZMQ
	this->zmq_async_input_output_spike_driver = new ZMQAsyncInputOutputSpikeDriver(
		publishPort,
		subscribeAddress,
		subscribePort,
		publishTopic,
		subscribeTopic);
	this->CommunivationInterval = communicationInterval;
	this->InputSpikeDriverList.push_back(this->zmq_async_input_output_spike_driver);
	this->OutputSpikeDriverList.push_back(this->zmq_async_input_output_spike_driver);
#else
	(void)subscribeAddress;
	(void)publishPort;
	(void)subscribePort;
	(void)publishTopic;
	(void)subscribeTopic;
	(void)communicationInterval;
	std::cout << "ZMQ support is disabled in this build." << std::endl;
#endif
}

int Simulation::AddInputConvFrameSource(const std::string& source_name,
                                        InputConvFrameDriver* driver) {
	if (source_name.empty() || driver == NULL) {
		return -1;
	}
	std::map<std::string, int>::const_iterator found =
		this->InputConvFrameSourceIndexByName.find(source_name);
	if (found != this->InputConvFrameSourceIndexByName.end()) {
		const int existing_index = found->second;
		if (existing_index >= 0 &&
			existing_index < static_cast<int>(this->InputConvFrameSourceList.size())) {
			delete this->InputConvFrameSourceList[static_cast<std::size_t>(existing_index)];
			this->InputConvFrameSourceList[static_cast<std::size_t>(existing_index)] = driver;
			return existing_index;
		}
	}
	const int source_index = static_cast<int>(this->InputConvFrameSourceList.size());
	this->InputConvFrameSourceList.push_back(driver);
	this->InputConvFrameSourceIndexByName[source_name] = source_index;
	return source_index;
}

int Simulation::AddAsyncInputConvFrameSource(const std::string& source_name,
                                             AsyncInputConvFrameDriver* driver,
                                             bool start_immediately) {
	const int source_index = this->AddInputConvFrameSource(source_name, driver);
	if (source_index >= 0 && driver != NULL && start_immediately) {
		driver->Start();
	}
	return source_index;
}

bool Simulation::BindInputConvFrameSource(int inputconv_index,
                                          const std::string& source_name,
                                          int source_camera_index,
                                          std::string* reason) {
	if (inputconv_index < 0 ||
		inputconv_index >= static_cast<int>(this->InputConvModelList.size())) {
		if (reason != NULL) {
			*reason = "InputConv index is out of range";
		}
		return false;
	}
	std::map<std::string, int>::const_iterator found =
		this->InputConvFrameSourceIndexByName.find(source_name);
	if (found == this->InputConvFrameSourceIndexByName.end()) {
		if (reason != NULL) {
			*reason = "InputConv frame source was not found";
		}
		return false;
	}
	if (this->InputConvFrameBindings.size() < this->InputConvModelList.size()) {
		this->InputConvFrameBindings.resize(this->InputConvModelList.size());
	}
	InputConvFrameSourceBinding binding;
	binding.source_index = found->second;
	binding.source_camera_index = source_camera_index;
	this->InputConvFrameBindings[static_cast<std::size_t>(inputconv_index)] = binding;
	return true;
}

bool Simulation::BindInputConvFrameSource(const std::string& inputconv_name,
                                          const std::string& source_name,
                                          int source_camera_index,
                                          std::string* reason) {
	for (std::size_t index = 0; index < this->InputConvDescriptionList.size(); ++index) {
		const std::string candidate =
			this->InputConvDescriptionList[index].ModelName + "#" + std::to_string(index);
		if (candidate == inputconv_name || this->InputConvDescriptionList[index].ModelName == inputconv_name) {
			return this->BindInputConvFrameSource(static_cast<int>(index),
												 source_name,
												 source_camera_index,
												 reason);
		}
	}
	if (reason != NULL) {
		*reason = "InputConv name was not found";
	}
	return false;
}

bool Simulation::HasInputConvFrameSourceBinding(int inputconv_index) const {
	if (inputconv_index < 0 ||
		inputconv_index >= static_cast<int>(this->InputConvFrameBindings.size())) {
		return false;
	}
	const InputConvFrameSourceBinding& binding =
		this->InputConvFrameBindings[static_cast<std::size_t>(inputconv_index)];
	return binding.source_index >= 0 &&
		binding.source_index < static_cast<int>(this->InputConvFrameSourceList.size()) &&
		this->InputConvFrameSourceList[static_cast<std::size_t>(binding.source_index)] != NULL;
}

bool Simulation::LoadInputConvFrame(int inputconv_index,
                                    int time_step,
                                    InputConvFrame* frame,
                                    std::string* reason) {
	if (frame == NULL) {
		if (reason != NULL) {
			*reason = "InputConv frame destination must not be null";
		}
		return false;
	}
	if (!this->HasInputConvFrameSourceBinding(inputconv_index)) {
		if (reason != NULL) {
			*reason = "InputConv is not bound to a dynamic frame source";
		}
		return false;
	}
	InputConvModel* model = this->InputConvModelList[static_cast<std::size_t>(inputconv_index)];
	if (model == NULL || !model->UsesDynamicFrameInput()) {
		if (reason != NULL) {
			*reason = "InputConv model does not support dynamic frame input";
		}
		return false;
	}
	const InputConvFrameSourceBinding& binding =
		this->InputConvFrameBindings[static_cast<std::size_t>(inputconv_index)];
	InputConvFrameRequest request =
		model->GetFrameRequest(time_step, inputconv_index, binding.source_camera_index);
	InputConvFrameDriver* driver =
		this->InputConvFrameSourceList[static_cast<std::size_t>(binding.source_index)];
	return driver->LoadFrame(request, frame, reason);
}

bool Simulation::AddExternalInputConvFrames(const std::string& source_name,
                                            const std::vector<InputConvFrame>& frames,
                                            std::string* reason) {
	if (source_name.empty()) {
		if (reason != NULL) {
			*reason = "InputConv frame source name must not be empty";
		}
		return false;
	}
	QueuedInputConvFrameDriver* queue_driver = NULL;
	std::map<std::string, int>::const_iterator found =
		this->InputConvFrameSourceIndexByName.find(source_name);
	if (found == this->InputConvFrameSourceIndexByName.end()) {
		queue_driver = new QueuedInputConvFrameDriver();
		this->AddInputConvFrameSource(source_name, queue_driver);
	} else {
		const int source_index = found->second;
		if (source_index < 0 ||
			source_index >= static_cast<int>(this->InputConvFrameSourceList.size())) {
			if (reason != NULL) {
				*reason = "InputConv frame source index is invalid";
			}
			return false;
		}
		queue_driver = dynamic_cast<QueuedInputConvFrameDriver*>(
			this->InputConvFrameSourceList[static_cast<std::size_t>(source_index)]);
		if (queue_driver == NULL) {
			if (reason != NULL) {
				*reason = "InputConv frame source is not a queued source";
			}
			return false;
		}
	}
	queue_driver->PushFrames(frames);
	return true;
}

bool Simulation::ClearInputConvFrameQueue(const std::string& source_name,
                                          std::string* reason) {
	std::map<std::string, int>::const_iterator found =
		this->InputConvFrameSourceIndexByName.find(source_name);
	if (found == this->InputConvFrameSourceIndexByName.end()) {
		if (reason != NULL) {
			*reason = "InputConv frame source was not found";
		}
		return false;
	}
	const int source_index = found->second;
	if (source_index < 0 ||
		source_index >= static_cast<int>(this->InputConvFrameSourceList.size())) {
		if (reason != NULL) {
			*reason = "InputConv frame source index is invalid";
		}
		return false;
	}
	this->InputConvFrameSourceList[static_cast<std::size_t>(source_index)]->Clear();
	return true;
}

bool Simulation::GetInputConvFrameSourceStats(const std::string& source_name,
                                              InputConvFrameSourceStats* stats,
                                              std::string* reason) const {
	if (stats == NULL) {
		if (reason != NULL) {
			*reason = "InputConv frame source stats destination must not be null";
		}
		return false;
	}
	std::map<std::string, int>::const_iterator found =
		this->InputConvFrameSourceIndexByName.find(source_name);
	if (found == this->InputConvFrameSourceIndexByName.end()) {
		if (reason != NULL) {
			*reason = "InputConv frame source was not found";
		}
		return false;
	}
	return this->GetInputConvFrameSourceStats(found->second, stats, reason);
}

bool Simulation::GetInputConvFrameSourceStats(int source_index,
                                              InputConvFrameSourceStats* stats,
                                              std::string* reason) const {
	if (stats == NULL) {
		if (reason != NULL) {
			*reason = "InputConv frame source stats destination must not be null";
		}
		return false;
	}
	if (source_index < 0 ||
		source_index >= static_cast<int>(this->InputConvFrameSourceList.size()) ||
		this->InputConvFrameSourceList[static_cast<std::size_t>(source_index)] == NULL) {
		if (reason != NULL) {
			*reason = "InputConv frame source index is invalid";
		}
		return false;
	}
	*stats = this->InputConvFrameSourceList[static_cast<std::size_t>(source_index)]->GetStats();
	return true;
}

int Simulation::AddZMQInputConvFrameSource(const std::string& source_name,
                                           std::string address,
                                           unsigned short port,
                                           int max_payload_bytes) {
#if SNN_WITH_ZMQ
	return this->AddInputConvFrameSource(
		source_name,
		new ZMQInputConvFrameDriver(address, port, max_payload_bytes));
#else
	(void)source_name;
	(void)address;
	(void)port;
	(void)max_payload_bytes;
	std::cout << "ZMQ support is disabled in this build." << std::endl;
	return -1;
#endif
}

int Simulation::AddZMQAsyncInputConvFrameSource(const std::string& source_name,
                                                std::string subscribe_address,
                                                unsigned short subscribe_port,
                                                std::string topic,
                                                int max_payload_bytes,
                                                int max_buffered_frames_per_camera) {
#if SNN_WITH_ZMQ
	return this->AddAsyncInputConvFrameSource(
		source_name,
		new ZMQAsyncInputConvFrameDriver(subscribe_address,
		                                 subscribe_port,
		                                 topic,
		                                 max_payload_bytes,
		                                 max_buffered_frames_per_camera),
		true);
#else
	(void)source_name;
	(void)subscribe_address;
	(void)subscribe_port;
	(void)topic;
	(void)max_payload_bytes;
	(void)max_buffered_frames_per_camera;
	std::cout << "ZMQ support is disabled in this build." << std::endl;
	return -1;
#endif
}

void Simulation::SaveWeight() {
	std::cout << "Saving weights..." << std::endl;
	for (std::list<OutputWeightDriver*>::iterator it = this->WeightDriverList.begin(); it != this->WeightDriverList.end(); it++) {
		(*it)->WriteWeight(this, this->currenttime[0]);
	}
}


void Simulation::LoadInput(CommunicationEvent* c_event) {
	//逐一在InputSpikeDriverList中加载输入
	for (std::list<InputSpikeDriver*>::iterator it = this->InputSpikeDriverList.begin(); it != this->InputSpikeDriverList.end(); it++) {
		if (!(*it)->isFinished) {
			(*it)->LoadInputSpike(this->EventHeap, this->network, c_event->getTime());
		}
	}
}

void Simulation::PublishOutput(CommunicationEvent* c_event) {
	//逐一在OutputSpikeDriverList中发布输出
	for (std::list<OutputSpikeDriver*>::iterator it = this->OutputSpikeDriverList.begin(); it != this->OutputSpikeDriverList.end(); it++) {
		if ((*it)->IsBuffered()) {
			(*it)->FlushBuffers();
		}
	}

}

void Simulation::LoadWeight(const char* filename) {
	std::string reason;
	if (!npgr::sim_support::LoadSimulationWeights(this, filename, &reason)) {
		std::cerr << "LoadWeight failed: " << reason << std::endl;
	}
}

void Simulation::SaveWeightToFile(const char* filename) {
	std::string reason;
	if (!npgr::sim_support::SaveSimulationWeights(this, filename, &reason)) {
		std::cerr << "SaveWeightToFile failed: " << reason << std::endl;
	}
}

bool Simulation::GetConnectionWeight(int original_connection_index,
                                     float* weight,
                                     std::string* reason) const {
	return npgr::sim_support::GetConnectionWeight(this, original_connection_index, weight, reason);
}

bool Simulation::SetConnectionWeight(int original_connection_index,
                                     float weight,
                                     std::string* reason) {
	return npgr::sim_support::SetConnectionWeight(this, original_connection_index, weight, reason);
}

bool Simulation::EnableDebugMonitor(const npgr::DebugMonitorConfig& config, std::string* reason) {
	this->debug_monitor_config = config;
	this->debug_monitor_config.enabled = true;
	if (this->debug_monitor == NULL) {
		this->debug_monitor = new npgr::SimulationDebugMonitor();
	}
	return this->debug_monitor->Initialize(this, this->debug_monitor_config, reason);
}

bool Simulation::DisableDebugMonitor(std::string* reason) {
	if (this->debug_monitor != NULL) {
		this->debug_monitor->Flush(reason);
		delete this->debug_monitor;
		this->debug_monitor = NULL;
	}
	this->debug_monitor_config.enabled = false;
	return true;
}

bool Simulation::FlushDebugMonitor(std::string* reason) {
	if (this->debug_monitor == NULL) {
		return true;
	}
	return this->debug_monitor->Flush(reason);
}

void Simulation::SetDebugMonitorConfig(const npgr::DebugMonitorConfig& config) {
	this->debug_monitor_config = config;
}

bool Simulation::CaptureDebugMonitorStep(int time_step, std::string* reason) {
	if (this->debug_monitor == NULL) {
		return true;
	}
	return this->debug_monitor->CaptureStep(time_step, reason);
}

bool Simulation::EnableInputConvMonitor(int inputconv_index, std::string* reason) {
	if (inputconv_index < 0 || inputconv_index >= static_cast<int>(this->InputConvModelList.size())) {
		if (reason != NULL) {
			*reason = "InputConv monitor index is out of range";
		}
		return false;
	}
	if (std::find(this->debug_monitor_config.monitored_inputconv_indices.begin(),
	              this->debug_monitor_config.monitored_inputconv_indices.end(),
	              inputconv_index) == this->debug_monitor_config.monitored_inputconv_indices.end()) {
		this->debug_monitor_config.monitored_inputconv_indices.push_back(inputconv_index);
	}
	this->debug_monitor_config.enabled = true;
	this->debug_monitor_config.record_inputconv_outputs = true;
	if (this->debug_monitor == NULL || !this->debug_monitor->enabled()) {
		return this->EnableDebugMonitor(this->debug_monitor_config, reason);
	}
	return this->debug_monitor->EnableInputConvMonitor(inputconv_index, reason);
}

bool Simulation::EnableInputConvMonitor(const std::string& inputconv_name, std::string* reason) {
	for (std::size_t index = 0; index < this->InputConvDescriptionList.size(); ++index) {
		const std::string candidate =
			this->InputConvDescriptionList[index].ModelName + "#" + std::to_string(index);
		if (candidate == inputconv_name || this->InputConvDescriptionList[index].ModelName == inputconv_name) {
			return this->EnableInputConvMonitor(static_cast<int>(index), reason);
		}
	}
	if (reason != NULL) {
		*reason = "InputConv monitor name was not found";
	}
	return false;
}

bool Simulation::DisableInputConvMonitor(int inputconv_index, std::string* reason) {
	this->debug_monitor_config.monitored_inputconv_indices.erase(
		std::remove(this->debug_monitor_config.monitored_inputconv_indices.begin(),
		            this->debug_monitor_config.monitored_inputconv_indices.end(),
		            inputconv_index),
		this->debug_monitor_config.monitored_inputconv_indices.end());
	if (this->debug_monitor != NULL) {
		return this->debug_monitor->DisableInputConvMonitor(inputconv_index, reason);
	}
	return true;
}

bool Simulation::DisableInputConvMonitor(const std::string& inputconv_name, std::string* reason) {
	for (std::size_t index = 0; index < this->InputConvDescriptionList.size(); ++index) {
		const std::string candidate =
			this->InputConvDescriptionList[index].ModelName + "#" + std::to_string(index);
		if (candidate == inputconv_name || this->InputConvDescriptionList[index].ModelName == inputconv_name) {
			return this->DisableInputConvMonitor(static_cast<int>(index), reason);
		}
	}
	if (reason != NULL) {
		*reason = "InputConv monitor name was not found";
	}
	return false;
}

void Simulation::WriteOuterDynamicState(int time_step,
                                        const OuterDynamicJointState& state,
                                        const OuterDynamicModel* source) {
	int component_index = -1;
	std::string component_name("outer_dynamic");
	if (source != NULL) {
		for (std::size_t index = 0; index < this->OuterDynamicModelList.size(); ++index) {
			if (this->OuterDynamicModelList[index] == source) {
				component_index = static_cast<int>(index);
				component_name = source->name();
				break;
			}
		}
	}
	{
		std::lock_guard<std::mutex> lock(this->outer_dynamic_state_mutex);
		this->latest_outer_dynamic_state = state;
		this->latest_outer_dynamic_time_step = time_step;
		this->has_latest_outer_dynamic_state = true;
		bool updated = false;
		for (std::size_t index = 0; index < this->latest_outer_dynamic_states.size(); ++index) {
			OuterDynamicStateSnapshot& snapshot = this->latest_outer_dynamic_states[index];
			if (snapshot.component_index == component_index && snapshot.component_name == component_name) {
				snapshot.state = state;
				snapshot.time_step = time_step;
				updated = true;
				break;
			}
		}
		if (!updated) {
			OuterDynamicStateSnapshot snapshot;
			snapshot.component_index = component_index;
			snapshot.component_name = component_name;
			snapshot.state = state;
			snapshot.time_step = time_step;
			this->latest_outer_dynamic_states.push_back(snapshot);
		}
	}
	for (std::list<OuterDynamicStateDriver*>::iterator it = this->OuterDynamicStateDriverList.begin(); it != this->OuterDynamicStateDriverList.end(); it++) {
		(*it)->WriteJointState(time_step, this->basetimesteps, state);
	}
}

bool Simulation::GetLatestOuterDynamicState(OuterDynamicJointState& state, int& time_step) const {
	std::lock_guard<std::mutex> lock(this->outer_dynamic_state_mutex);
	if (!this->has_latest_outer_dynamic_state) {
		return false;
	}
	state = this->latest_outer_dynamic_state;
	time_step = this->latest_outer_dynamic_time_step;
	return true;
}

void Simulation::CreateOuterDynamic(const std::list<OuterDynamicDescription>& outer_dynamic_list) {
	int outer_dynamic_index = 0;
	for (std::list<OuterDynamicDescription>::const_iterator it = outer_dynamic_list.begin(); it != outer_dynamic_list.end(); it++, outer_dynamic_index++) {
		OuterDynamicModel* model = OuterDynamicModelFactory::createOuterDynamicModel(*it);
		const std::string resolved_name =
			it->name.empty() ? (it->ModelName + "#" + std::to_string(outer_dynamic_index)) : it->name;
		if (this->FindOuterDynamicByName(resolved_name) >= 0) {
			delete model;
			throw std::runtime_error("Duplicate OuterDynamic name: " + resolved_name);
		}
		model->setName(resolved_name);
		model->Initialize(*it, this);
		this->OuterDynamicModelList.push_back(model);
	}
}

bool Simulation::GetLatestOuterDynamicStates(std::vector<OuterDynamicStateSnapshot>& states) const {
	std::lock_guard<std::mutex> lock(this->outer_dynamic_state_mutex);
	if (this->latest_outer_dynamic_states.empty()) {
		return false;
	}
	states = this->latest_outer_dynamic_states;
	return true;
}

int Simulation::FindOuterDynamicByName(const std::string& name) const {
	if (name.empty()) {
		return -1;
	}
	for (std::size_t index = 0; index < this->OuterDynamicModelList.size(); ++index) {
		const OuterDynamicModel* model = this->OuterDynamicModelList[index];
		if (model != NULL && model->name() == name) {
			return static_cast<int>(index);
		}
	}
	return -1;
}

OuterDynamicModel* Simulation::GetOuterDynamicByName(const std::string& name) {
	const int index = this->FindOuterDynamicByName(name);
	if (index < 0) {
		return NULL;
	}
	return this->OuterDynamicModelList[static_cast<std::size_t>(index)];
}

const OuterDynamicModel* Simulation::GetOuterDynamicByName(const std::string& name) const {
	const int index = this->FindOuterDynamicByName(name);
	if (index < 0) {
		return NULL;
	}
	return this->OuterDynamicModelList[static_cast<std::size_t>(index)];
}

void Simulation::QueueOuterDynamicInputSpike(int event_time, int neuron_id) {
	std::lock_guard<std::mutex> lock(this->outer_dynamic_input_mutex);
	this->outer_dynamic_input_times.push_back(event_time);
	this->outer_dynamic_input_neurons.push_back(neuron_id);
}

void Simulation::FlushOuterDynamicInputSpikes(int current_time) {
	std::vector<int> event_times;
	std::vector<int> event_neurons;
	{
		std::lock_guard<std::mutex> lock(this->outer_dynamic_input_mutex);
		if (this->outer_dynamic_input_times.empty()) {
			return;
		}

		std::vector<int> remaining_times;
		std::vector<int> remaining_neurons;
		remaining_times.reserve(this->outer_dynamic_input_times.size());
		remaining_neurons.reserve(this->outer_dynamic_input_neurons.size());

		for (std::size_t i = 0; i < this->outer_dynamic_input_times.size(); ++i) {
			if (this->outer_dynamic_input_times[i] > current_time) {
				event_times.push_back(this->outer_dynamic_input_times[i]);
				event_neurons.push_back(this->outer_dynamic_input_neurons[i]);
			}
		}

		this->outer_dynamic_input_times.swap(remaining_times);
		this->outer_dynamic_input_neurons.swap(remaining_neurons);
	}

	if (!event_times.empty()) {
		this->inputSpikeDriver->LoadInputSpike(
			this->EventHeap,
			this->network,
			static_cast<int>(event_times.size()),
			event_times.data(),
			event_neurons.data());
	}
}




