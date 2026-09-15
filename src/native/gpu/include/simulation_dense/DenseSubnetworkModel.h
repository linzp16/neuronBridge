/*
 * DenseSubnetworkModel.h
 *
 * Owns one GPU dense subnetwork and exposes it as a legacy neuron-model
 * boundary. It gathers spikes/currents from legacy interface connections,
 * advances the GPU runtime, then schedules emitted dense spikes back into the
 * legacy event queue.
 */
#ifndef NPGR_DENSE_SUBNETWORK_MODEL_H
#define NPGR_DENSE_SUBNETWORK_MODEL_H

#include "bridge/LegacyNetworkBridge.h"
#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"

#include <mutex>
#include <string>
#include <vector>

class Simulation;
class NeuronModel;
class EventQueue;
class Interconnections;
class Neuron;

namespace npgr {

class DenseInterfaceSpikeNeuronModel;
class DenseInterfaceCurrentNeuronModel;

struct DenseSubnetworkHostProfiling {
    // Number of DenseSubnetworkModel::AdvanceStep calls.
    long long advance_count = 0;
    // End-to-end time spent inside DenseSubnetworkModel::AdvanceStep.
    double advance_total_ms = 0.0;
    // Time spent rebuilding/uploading interface state before GPU execution.
    double flush_interface_ms = 0.0;
    // Time spent executing the dense GPU runtime event.
    double event_execute_ms = 0.0;
    // Time spent expanding dense output firings back into legacy events.
    double output_schedule_ms = 0.0;
    // Time spent in the dense debug-capture hook called by AdvanceStep.
    double debug_capture_ms = 0.0;
    // Number of output spikes scheduled from dense back into the legacy queue.
    long long output_spike_count = 0;
    // Number of current ProcessCurrent calls received by dense interface models.
    long long current_process_count = 0;
    // Time spent inside DenseInterfaceCurrentNeuronModel::ProcessCurrent.
    double current_process_ms = 0.0;
};

class DenseSubnetworkModel {
public:
    DenseSubnetworkModel();
    ~DenseSubnetworkModel();

    // Builds a dense subnetwork from an already prepared runtime spec and binding.
    bool InitializeBlackBox(Simulation* simulation,
                            const sim_support::DenseSubnetworkBuildSpec& spec,
                            const LegacyNetworkBinding& binding,
                            const RuntimeConfig& config,
                            int queue_index,
                            int update_timestep,
                            std::string* reason = nullptr);
    // Binds output routes after all dense subnetworks and their interface
    // models exist. This enables dense-to-dense boundary connections.
    bool BindOutputConnections(const LegacyNetworkBinding& binding, std::string* reason = nullptr);
    // Reinitializes runtime buffers and clears pending interface state.
    bool Reset(std::string* reason = nullptr);
    // Adds one spike-derived conductance delta to an interface slot.
    bool QueueInterfaceSpikeSlot(int interface_slot_index,
                                 float weight,
                                 bool inhibitory,
                                 int time_step,
                                 std::string* reason = nullptr);
    // Stores the latest current value for one legacy-to-dense current connection.
    bool SetInterfaceCurrentConnection(int current_connection_index,
                                       float current,
                                       int time_step,
                                       std::string* reason = nullptr);
    // Binds a device output buffer to dense pending channels through fixed D2D copy records.
    bool BindInterfaceDeviceSource(const float* device_values,
                                   int value_count,
                                   const std::vector<DeviceToDeviceInterfaceCopySpec>& copy_specs,
                                   std::string* reason = nullptr);
    // Binds a device buffer that supplies interface currents directly on the GPU.
    bool BindInterfaceCurrentDeviceSource(const float* device_current,
                                          int interface_count,
                                          std::string* reason = nullptr);
    // Prepares and uploads all pending interface state for the given timestep.
    bool FlushInterfaceStateBuffers(int time_step, std::string* reason = nullptr);
    // Runs one dense timestep and schedules dense outputs into the legacy queue.
    bool AdvanceStep(int time_step, EventQueue* event_queue, std::string* reason = nullptr);
    // Enables or disables full firing export in debug snapshots.
    void SetFullFiringExportEnabled(bool enabled);
    bool full_firing_export_enabled() const;

    // Returns true if the supplied legacy model belongs to this dense subnetwork.
    bool OwnsInternalModel(const NeuronModel* model) const;
    // Event queue index used by this dense subnetwork.
    int queue_index() const;
    // Dense update period in simulation steps.
    int update_timestep() const;
    // Current timestep of the owning simulation queue.
    int current_time_step() const;
    // Interface neuron model used for spike-delivery boundary connections.
    DenseInterfaceSpikeNeuronModel* interface_spike_model() const;
    // Interface neuron model used for current-delivery boundary connections.
    DenseInterfaceCurrentNeuronModel* interface_current_model() const;
    // User-facing dense subnetwork name.
    const std::string& name() const;
    // Debug helper that exposes only the route count, not the retained routing table.
    int OutputRouteCountForSourceNeuron(int source_neuron_id) const;
    bool HasMonitorTargets() const;
    int OriginalGlobalNeuronIdForLocal(int local_neuron_id) const;
    // Translates a user-visible original neuron id into the compact dense-local
    // id used by runtime buffers after model-based layout compaction.
    int LocalNeuronIdForOriginalGlobal(int original_neuron_id) const;
    bool IsMonitorCandidateLocalNeuron(int local_neuron_id) const;
    // Captures a detailed runtime snapshot for diagnostics.
    DenseSubnetworkDebugSnapshot BuildDebugSnapshot(std::string* reason = nullptr) const;
    // Captures only the monitor state groups requested by the active config.
    bool ExportMonitorState(bool record_state,
                            bool record_spikes,
                            bool record_pending_channels,
                            bool record_weights,
                            DenseSubnetworkMonitorState* out,
                            std::string* reason = nullptr) const;
    // Returns the dense synaptic weights stored in the debug mirror.
    std::vector<float> GetSynapticWeights() const;
    // Downloads the live GPU synaptic weights for save/load snapshots.
    bool DownloadLiveSynapticWeights(std::vector<float>* weights, std::string* reason = nullptr) const;
    // Uploads live GPU synaptic weights from a save/load snapshot.
    bool UploadLiveSynapticWeights(const std::vector<float>& weights, std::string* reason = nullptr);
    // Reads one live GPU synaptic weight by dense runtime synapse index.
    bool GetLiveSynapseWeight(int synapse_index, float* weight, std::string* reason = nullptr) const;
    // Writes one live GPU synaptic weight by dense runtime synapse index.
    bool SetLiveSynapseWeight(int synapse_index, float weight, std::string* reason = nullptr);
    // Clears host-side timing counters used by dense-through-Simulation benchmarks.
    static void ResetHostProfiling();
    // Returns host-side timing counters used by dense-through-Simulation benchmarks.
    static DenseSubnetworkHostProfiling HostProfilingSnapshot();
    // Records current-interface delivery time from DenseInterfaceCurrentNeuronModel.
    static void RecordCurrentProcessTime(long long elapsed_ns);

private:
    // Owning simulation, used for current time and monitor capture.
    Simulation* simulation_;
    // Dense subnetwork name retained after build-spec release.
    std::string name_;
    // Runtime configuration copied at initialization.
    RuntimeConfig config_;
    // GPU propagation and neuron update runtime.
    DenseSubnetworkRuntimeGpu runtime_;
    // Synthetic legacy model that receives spike interface connections.
    DenseInterfaceSpikeNeuronModel* interface_spike_model_;
    // Synthetic legacy model that receives current interface connections.
    DenseInterfaceCurrentNeuronModel* interface_current_model_;
    // Event queue index used for dense update events.
    int queue_index_;
    // Number of simulation steps between dense updates.
    int update_timestep_;
    // Number of external interface slots retained after build-spec release.
    int interface_slot_count_;
    // True when the interface slot accepts current rather than spike delivery.
    std::vector<unsigned char> interface_uses_current_;
    // Legacy output connections retained in dense-output order.
    std::vector<Interconnections*> output_connections_;
    // Event queue index for every retained output connection.
    std::vector<int> output_queue_indices_;
    // CSR starts for dense source neuron -> output connection indices.
    std::vector<int> output_route_start_by_source_neuron_;
    // CSR payload storing output connection indices for dense output routing.
    std::vector<int> output_route_binding_indices_;
    // Host debug mirror of dense synaptic weights after build-spec release.
    mutable std::vector<float> debug_synaptic_weights_;
    // Dense-local neuron id -> original user-visible neuron id for monitor output.
    std::vector<int> monitor_local_to_original_ids_;
    // Dense-local monitor candidates collected from layer.isMonitored.
    std::vector<unsigned char> monitor_candidate_mask_;
    // Output connections allocated for black-box dense specs.
    std::vector<Interconnections*> owned_output_connections_;
    // Protects interface buffers and runtime snapshot reads.
    mutable std::mutex input_mutex_;
};

}  // namespace npgr

#endif
