/*
 * Simulation.h
 *
 * Top-level event-driven simulation controller for the legacy SNN runtime.
 * Simulation owns the event queue, compiled Network, input/output drivers,
 * optional outer-dynamics models, optional input-convolution models, and the
 * optional GPU dense-subnetwork bridge.
 */
#ifndef SIMULATION_H
#define SIMULATION_H

#include "../source_file_realtime_v1_async/EventQueue/inc/EventQueue.h"
#include "../source_file_realtime_v1_async/InputCurrentDriver/inc/ArrayInputCurrentDriver.h"
#include "../source_file_realtime_v1_async/InputSpikeDriver/inc/ArrayInputSpikeDriver.h"
#include "../source_file_realtime_v1_async/Network/inc/Network.h"
#include "../source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "../source_file_realtime_v1_async/Simulation/inc/InputConvDescription.h"
#include "../source_file_realtime_v1_async/InputConv/inc/InputConvFrameDriver.h"
#include "../source_file_realtime_v1_async/InputConv/inc/AsyncInputConvFrameDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/ArrayOutputSpikeDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/FileOuterDynamicStateDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/FileOutputSpikeDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/FileOutputWeightDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/OuterDynamicStateDriver.h"
#include "simulation_dense/SimulationConnectionWeightRef.h"
#include "debug_monitor/DebugMonitorTypes.h"
#include "RealTimeRestriction.h"
#include <atomic>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class OutputSpikeDriver;
class Spike;
class Neuron;
class CommunicationEvent;
class ZMQInputOutputSpikeDriver;
class ZMQAsyncInputOutputSpikeDriver;
class OuterDynamicModel;
class InputConvModel;
class OuterDynamicSpikeBuffer;
class OuterDynamicStateDriver;

namespace npgr {
class DenseSubnetworkModel;
class SimulationDebugMonitor;
struct DenseSubnetworkDebugSnapshot;
}

enum EventQueueType { EVENT_QUEUE_HEAP, EVENT_QUEUE_TIMING_WHEEL };

// Latest observable state for one named OuterDynamic instance. The legacy
// single-state fields remain below for compatibility with existing callers.
struct OuterDynamicStateSnapshot {
    int component_index = -1;
    std::string component_name;
    OuterDynamicJointState state;
    int time_step = 0;
};

class Simulation {
public:
    // Compiled legacy network owned by the simulation.
    Network* network;
    // Event scheduler used by all simulation queues.
    EventQueue* EventHeap;
    // Absolute end time for an initialized run.
    int totaltimesteps;
    // Base timestep size in milliseconds.
    float basetimesteps;
    // Current timestep for each event queue.
    int* currenttime;
    // Number of OpenMP/event queues.
    int NumberOfQueue;
    // Per-queue stop flags set by EndSimulationTime events.
    bool* SimulationEnd;

    // Built-in array-backed input and output drivers.
    ArrayInputSpikeDriver* inputSpikeDriver;
    ArrayInputCurrentDriver* inputCurrentDriver;
    ArrayOutputSpikeDriver* output_spike_driver;

    // Optional file-backed output drivers.
    FileOutputSpikeDriver* file_output_spike_driver;
    FileOuterDynamicStateDriver* file_outer_dynamic_state_driver;

    // Registered driver lists consumed by synchronization/communication events.
    std::list<InputSpikeDriver*> InputSpikeDriverList;
    std::list<InputCurrentDriver*> InputCurrentDriverList;
    std::list<OutputSpikeDriver*> OutputSpikeDriverList;
    std::list<OutputWeightDriver*> WeightDriverList;
    std::list<OuterDynamicStateDriver*> OuterDynamicStateDriverList;

    // Per-queue synchronization and pause flags.
    bool* SyncThread;
    bool* PauseThread;
    // Minimum cross-queue propagation delay; negative means no cross-queue sync is needed.
    int DelayMin;

    // Optional file-backed weight writer and ZMQ communication drivers.
    FileOutputWeightDriver* file_output_weight_driver;
    ZMQInputOutputSpikeDriver* zmq_input_output_spike_driver;
    ZMQAsyncInputOutputSpikeDriver* zmq_async_input_output_spike_driver;
    // Weight-save interval in simulation steps.
    int WeightSaveInterval;
    // Communication interval in simulation steps. 
    int CommunivationInterval;

    // True when at least one neuron monitor exists in the network.
    bool neuronMonitorExist;
    // Real-time throttle and watchdog state.
    RealTimeRestriction* RealTimeRestrictionObject;
    bool RealtimeEnabled;
    int RealtimeSlotSteps;
    bool PrintSimulationTime;
    std::thread watchdogThread;
    std::atomic<bool> watchdogStarted{ false };

    // Event queue implementation selector.
    EventQueueType eventQueueType;
    int timingWheelSize;
    // Prevents driver registration from being repeated across resets.
    bool drivers_initialized;

    // Optional outer-dynamics and input-convolution modules.
    std::vector<OuterDynamicModel*> OuterDynamicModelList;
    std::vector<InputConvModel*> InputConvModelList;
    // Descriptions retained so Simulation can resolve output bindings after
    // all optional dense subnetworks have been registered.
    std::vector<InputConvDescription> InputConvDescriptionList;
    // Per-InputConv hidden InputCurrentNeuronModel source ids used when an
    // InputConv output targets the main network as external current.
    std::vector<std::vector<int> > InputConvMainCurrentSourceNeuronIds;
    // Dynamic frame sources are named so multiple cameras or producers can be
    // bound to multiple InputConv models without changing model-specific code.
    std::vector<InputConvFrameDriver*> InputConvFrameSourceList;
    std::map<std::string, int> InputConvFrameSourceIndexByName;
    // Each InputConv may bind to exactly one source/camera pair.
    std::vector<InputConvFrameSourceBinding> InputConvFrameBindings;
    OuterDynamicSpikeBuffer* outer_dynamic_spike_buffer;
    std::vector<int> outer_dynamic_input_times;
    std::vector<int> outer_dynamic_input_neurons;
    std::mutex outer_dynamic_input_mutex;
    OuterDynamicJointState latest_outer_dynamic_state;
    int latest_outer_dynamic_time_step;
    bool has_latest_outer_dynamic_state;
    std::vector<OuterDynamicStateSnapshot> latest_outer_dynamic_states;
    mutable std::mutex outer_dynamic_state_mutex;

    // GPU dense subnetworks, when enabled.
    std::vector<npgr::DenseSubnetworkModel*> dense_subnetworks;
    // Unified debug monitor used for main-network, dense, InputConv, and outer-dynamics samples.
    npgr::SimulationDebugMonitor* debug_monitor;
    npgr::DebugMonitorConfig debug_monitor_config;
    // Maps original neuron IDs to the compact main-network IDs used after dense extraction.
    std::vector<int> original_to_main_neuron_id;
    // Maps original connection_list flattened indices to their live runtime owner.
    std::vector<npgr::RuntimeConnectionWeightRef> original_connection_weight_refs;

    // Runs the event loop for a relative number of simulation steps.
    void RunSimulationStep(int runsteps);
    // Runs the event loop while applying real-time pacing restrictions.
    void RunSimulationRealtime(int runsteps);

    // Constructs a simulation with only SNN layers, connections, and learning rules.
    Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list,
               const std::list<ConnectionDescription>& connection_list,
               const std::list<LearningRuleDescription>& learning_rule_list,
               int simulationsteps,
               float timestep,
               int NumberOfQueue,
               EventQueueType eventQueueType = EVENT_QUEUE_HEAP,
               int timingWheelSize = 0);

    // Constructs a simulation with additional outer-dynamics models.
    Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list,
               const std::list<ConnectionDescription>& connection_list,
               const std::list<LearningRuleDescription>& learning_rule_list,
               const std::list<OuterDynamicDescription>& outer_dynamic_list,
               int simulationsteps,
               float timestep,
               int NumberOfQueue,
               EventQueueType eventQueueType = EVENT_QUEUE_HEAP,
               int timingWheelSize = 0);

    // Constructs a simulation with outer-dynamics models and network-to-joint input routes.
    Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list,
               const std::list<ConnectionDescription>& connection_list,
               const std::list<LearningRuleDescription>& learning_rule_list,
               const std::list<OuterDynamicDescription>& outer_dynamic_list,
               const std::list<OuterDynamicConnectionDescription>& outer_dynamic_connection_list,
               int simulationsteps,
               float timestep,
               int NumberOfQueue,
               EventQueueType eventQueueType = EVENT_QUEUE_HEAP,
               int timingWheelSize = 0);

    // Constructs a simulation with outer-dynamics and input-convolution modules.
    Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list,
               const std::list<ConnectionDescription>& connection_list,
               const std::list<LearningRuleDescription>& learning_rule_list,
               const std::list<OuterDynamicDescription>& outer_dynamic_list,
               const std::list<InputConvDescription>& input_conv_list,
               int simulationsteps,
               float timestep,
               int NumberOfQueue,
               EventQueueType eventQueueType = EVENT_QUEUE_HEAP,
               int timingWheelSize = 0);

    // Constructs a simulation with outer-dynamics input routes and input-convolution modules.
    Simulation(const std::list<NeuronLayerDescription>& neuron_layer_list,
               const std::list<ConnectionDescription>& connection_list,
               const std::list<LearningRuleDescription>& learning_rule_list,
               const std::list<OuterDynamicDescription>& outer_dynamic_list,
               const std::list<OuterDynamicConnectionDescription>& outer_dynamic_connection_list,
               const std::list<InputConvDescription>& input_conv_list,
               int simulationsteps,
               float timestep,
               int NumberOfQueue,
               EventQueueType eventQueueType = EVENT_QUEUE_HEAP,
               int timingWheelSize = 0);

    // Releases network, drivers, queues, and optional runtime modules.
    ~Simulation();

    // Queues externally supplied spikes into the event queue.
    void AddExternalSpikeActivity(const std::vector<int>& EventTime, const std::vector<int>& EventNeuronID);
    // Queues externally supplied currents into the event queue.
    void AddExternalCurrentActivity(const std::vector<int>& EventTime,
                                    const std::vector<int>& EventNeuronID,
                                    const std::vector<float>& Current);
    // Registers drivers and schedules the first events for a run.
    void InitSimulation();
    // Clears dynamic state and prepares the event queue for another run.
    void ResetForNextRound(bool preserve_weights = true);
    // Marks one queue as complete.
    void EndSimulation(int QueueIndex);
    // Synchronizes queues for cross-queue spike propagation.
    void SynchronizeThread();
    // Synchronizes queues while honoring a real-time restriction level.
    void SynchronizeThread(RealTimeRestrictionLevel restrictionLevel);

    // Registers an output spike driver.
    void AddOutputSpikeDriver(OutputSpikeDriver* OutputSpikeDriver);
    // Writes one emitted spike to all output drivers.
    void WriteSpike(Spike* spike);
    // Binds the current OpenMP worker to a CUDA device.
    void SetOpenMPThreadToGPU(int openMpID);
    // Returns the configured weight-save interval.
    int GetWeightSaveStep();
    // Registers an input spike driver.
    void AddInputSpikeDriver(InputSpikeDriver* newInputSpikeDriver);
    // Registers an input current driver.
    void AddInputCurrentDriver(InputCurrentDriver* newInputCurrentDriver);
    // Registers a weight-output driver.
    void AddOutputWeightDriver(OutputWeightDriver* WeightDriver);
    // Registers an outer-dynamics state-output driver.
    void AddOuterDynamicStateDriver(OuterDynamicStateDriver* driver);

    // Creates and registers a file spike-output driver.
    void AddFileOutputSpikeDriver(std::string FileName);
    // Creates and registers a file outer-dynamics state driver.
    void AddFileOuterDynamicStateDriver(std::string FileName);
    // Creates and registers a file weight-output driver.
    void AddFileOutputWeightDriver(std::string FileName, int WeightSaveInterval);
    // Creates a request/response ZMQ spike communication driver.
    void AddZMQInputOutputSpikeDriver(enum DriverType Type,
                                      std::string address,
                                      unsigned short port,
                                      int CommunicationInterval);
    // Creates an asynchronous pub/sub ZMQ spike communication driver.
    void AddZMQAsyncInputOutputSpikeDriver(std::string subscribeAddress,
                                           unsigned short publishPort,
                                           unsigned short subscribePort,
                                           std::string publishTopic,
                                           std::string subscribeTopic,
                                           int CommunicationInterval);
    // Registers a named dynamic frame source. Simulation owns the driver.
    int AddInputConvFrameSource(const std::string& source_name,
                                InputConvFrameDriver* driver);
    // Registers a named asynchronous frame source and optionally starts it.
    int AddAsyncInputConvFrameSource(const std::string& source_name,
                                     AsyncInputConvFrameDriver* driver,
                                     bool start_immediately = true);
    // Binds one InputConv model to one source/camera pair.
    bool BindInputConvFrameSource(int inputconv_index,
                                  const std::string& source_name,
                                  int source_camera_index = 0,
                                  std::string* reason = nullptr);
    bool BindInputConvFrameSource(const std::string& inputconv_name,
                                  const std::string& source_name,
                                  int source_camera_index = 0,
                                  std::string* reason = nullptr);
    bool HasInputConvFrameSourceBinding(int inputconv_index) const;
    bool LoadInputConvFrame(int inputconv_index,
                            int time_step,
                            InputConvFrame* frame,
                            std::string* reason = nullptr);
    // Pushes frames into a named memory queue source, creating it on demand.
    bool AddExternalInputConvFrames(const std::string& source_name,
                                    const std::vector<InputConvFrame>& frames,
                                    std::string* reason = nullptr);
    bool ClearInputConvFrameQueue(const std::string& source_name,
                                  std::string* reason = nullptr);
    bool GetInputConvFrameSourceStats(const std::string& source_name,
                                      InputConvFrameSourceStats* stats,
                                      std::string* reason = nullptr) const;
    bool GetInputConvFrameSourceStats(int source_index,
                                      InputConvFrameSourceStats* stats,
                                      std::string* reason = nullptr) const;
    int AddZMQInputConvFrameSource(const std::string& source_name,
                                   std::string address,
                                   unsigned short port,
                                   int max_payload_bytes = 0);
    int AddZMQAsyncInputConvFrameSource(const std::string& source_name,
                                        std::string subscribe_address,
                                        unsigned short subscribe_port,
                                        std::string topic,
                                        int max_payload_bytes = 0,
                                        int max_buffered_frames_per_camera = 8);

    // Writes the current network weights through all registered weight drivers.
    void SaveWeight();
    // Publishes buffered output spikes for a communication event.
    void PublishOutput(CommunicationEvent* c_event);
    // Loads external input for a communication event.
    void LoadInput(CommunicationEvent* c_event);
    // Loads weights from a file into the network.
    void LoadWeight(const char* filename);
    // Saves network weights to a file.
    void SaveWeightToFile(const char* filename);
    // Reads the live weight addressed by the original connection_list flattened index.
    bool GetConnectionWeight(int original_connection_index,
                             float* weight,
                             std::string* reason = nullptr) const;
    // Writes the live weight addressed by the original connection_list flattened index.
    bool SetConnectionWeight(int original_connection_index,
                             float weight,
                             std::string* reason = nullptr);

    // Enables the unified debug monitor for main network, dense subnetworks,
    // InputConv modules, and outer-dynamics state.
    bool EnableDebugMonitor(const npgr::DebugMonitorConfig& config,
                            std::string* reason = nullptr);
    bool DisableDebugMonitor(std::string* reason = nullptr);
    bool FlushDebugMonitor(std::string* reason = nullptr);
    void SetDebugMonitorConfig(const npgr::DebugMonitorConfig& config);
    bool CaptureDebugMonitorStep(int time_step, std::string* reason = nullptr);
    bool EnableInputConvMonitor(int inputconv_index, std::string* reason = nullptr);
    bool EnableInputConvMonitor(const std::string& inputconv_name, std::string* reason = nullptr);
    bool DisableInputConvMonitor(int inputconv_index, std::string* reason = nullptr);
    bool DisableInputConvMonitor(const std::string& inputconv_name, std::string* reason = nullptr);

    // Enables real-time pacing and watchdog thresholds.
    void EnableRealtime(int slotSteps,
                        double maxAdvanceSeconds,
                        float firstSection,
                        float secondSection,
                        float thirdSection);
    // Disables real-time pacing.
    void DisableRealtime();
    // Enables or disables printing of simulation time progress.
    void SetPrintSimulationTime(bool enable);

    // Creates outer-dynamics models from descriptions.
    void CreateOuterDynamic(const std::list<OuterDynamicDescription>& outer_dynamic_list);
    // Finds an outer-dynamics model by user-visible name, or returns -1.
    int FindOuterDynamicByName(const std::string& name) const;
    // Returns an outer-dynamics model by name, or null when absent.
    OuterDynamicModel* GetOuterDynamicByName(const std::string& name);
    const OuterDynamicModel* GetOuterDynamicByName(const std::string& name) const;
    // Creates input-convolution models from descriptions.
    void CreateInputConv(const std::list<InputConvDescription>& input_conv_list);
    // Resets input-convolution model state.
    void ResetInputConvModels();
    // Publishes one InputConv model's latest output into main-network current routes.
    void FlushInputConvMainNetworkCurrent(InputConvModel* model, int time_step);

    // Binds a device-resident current buffer to a dense subnetwork interface.
    bool BindDenseSubnetworkInterfaceCurrentDeviceSource(int subnetwork_index,
                                                         const float* device_current,
                                                         int interface_count,
                                                         std::string* reason = nullptr);

    // Queues an outer-dynamics input spike for later flushing.
    void QueueOuterDynamicInputSpike(int event_time, int neuron_id);
    // Flushes queued outer-dynamics input spikes up to current_time.
    void FlushOuterDynamicInputSpikes(int current_time);
    // Stores the latest outer-dynamics joint state for consumers.
    void WriteOuterDynamicState(int time_step,
                                const OuterDynamicJointState& state,
                                const OuterDynamicModel* source = nullptr);
    // Reads the latest outer-dynamics joint state if one exists.
    bool GetLatestOuterDynamicState(OuterDynamicJointState& state, int& time_step) const;
    // Reads one latest state per reporting OuterDynamic instance.
    bool GetLatestOuterDynamicStates(std::vector<OuterDynamicStateSnapshot>& states) const;

    // Registers built-in and optional drivers once.
    void BindDriversOnce();
    // Inserts initial end, update, synchronization, and communication events.
    void ScheduleInitialEvents();
    // Recreates the selected event queue implementation.
    void RecreateEventQueue();

    // Creates GPU dense subnetworks from dense-marked neuron layers.
    void CreateDenseSubnetworks(const std::list<NeuronLayerDescription>& neuron_layer_list);
    // Resets all dense subnetworks.
    void ResetDenseSubnetworks();
    // Returns true when a neuron model is owned by a dense subnetwork.
    bool IsDenseManagedModel(const NeuronModel* model) const;
    // Converts an original neuron ID into its post-dense-extraction main-network ID.
    int TranslateOriginalNeuronIdToMain(int original_neuron_id) const;
    // Returns the number of dense subnetworks.
    int GetDenseSubnetworkCount() const;
    // Writes the dense subnetwork name into name.
    bool GetDenseSubnetworkName(int subnetwork_index, std::string& name) const;
    // Finds a dense subnetwork by name and returns its index, or -1 when absent.
    int FindDenseSubnetworkByName(const std::string& name) const;
    // Builds a detailed dense runtime debug snapshot.
    bool GetDenseSubnetworkDebugSnapshot(int subnetwork_index,
                                         npgr::DenseSubnetworkDebugSnapshot& snapshot) const;
    // Exports dense subnetwork synaptic weights.
    bool GetDenseSubnetworkWeights(int subnetwork_index, std::vector<float>& weights) const;
    // Resets one dense subnetwork.
    bool ResetDenseSubnetwork(int subnetwork_index);
};

#endif
