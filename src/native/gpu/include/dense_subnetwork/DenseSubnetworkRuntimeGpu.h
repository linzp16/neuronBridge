/*
 * DenseSubnetworkRuntimeGpu.h
 *
 * GPU-backed execution engine for a dense subnetwork. The runtime owns dense
 * neuron state, channelized pending-input buffers, and output-spike views
 * used by the legacy bridge. The GPU main path is implemented by
 * GpuPropagationRuntime and DenseUnifiedNeuronRuntime; FiringTable and
 * DelayRing are host-side diagnostic mirrors.
 */
#ifndef NPGR_DENSE_SUBNETWORK_RUNTIME_GPU_H
#define NPGR_DENSE_SUBNETWORK_RUNTIME_GPU_H

#include "dense_subnetwork/DenseSubnetworkFiringTable.h"
#include "dense_subnetwork/DenseUnifiedNeuronRuntime.h"
#include "gpu_runtime/GpuPropagationRuntime.h"
#include "simulation_dense/DenseBuildShared.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace npgr {

struct DenseInterfaceSpikeInput {
    // External input slot that receives this spike contribution.
    int interface_slot_index = -1;
    // Simulation step at which the input arrives.
    int time_step = 0;
    // Synaptic weight to add to the slot.
    float weight = 0.0f;
    // True when the contribution should update inhibitory conductance.
    bool inhibitory = false;
};

struct DenseInterfaceInputArrival {
    // External input slot that received activity.
    int interface_slot_index = -1;
    // Simulation step at which the activity arrived.
    int time_step = 0;
};

struct DenseInterfaceInputBatchItem {
    // External input slot index.
    int interface_slot_index = -1;
    // Pending channel selected from the target dense neuron model.
    std::uint8_t pending_channel = static_cast<std::uint8_t>(PendingChannel::ExcitatoryConductance);
    // Channel value staged for upload.
    float value = 0.0f;
    // Model-defined scaling applied at upload.
    float scale = 1.0f;
};

struct DenseOutputSpike {
    // Index into DenseSubnetworkBuildSpec output_* arrays.
    int output_index = -1;
    // Dense source neuron that fired.
    int source_neuron = -1;
    // Legacy target neuron ID outside the dense subnetwork.
    int external_target_id = -1;
    // Simulation step at which the dense neuron fired.
    int time_step = 0;
    // Delay to apply before delivery to the legacy target.
    int delay = 0;
    // Weight of the dense-to-legacy output connection.
    float weight = 0.0f;
    // Synapse type for the legacy output connection.
    int synapse_type = 0;
};

struct DenseOutputRouteEntry {
    int source_neuron = -1;
    // Runtime-owned output records for this dense source neuron.
    // Build-only output_* arrays are not retained after initialization.
    std::vector<DenseOutputSpike> output_spikes;
};

struct DenseSubnetworkDebugSnapshot {
    // Dense subnetwork name.
    std::string name;
    // Simulation step captured by this snapshot.
    int time_step = 0;
    // Runtime configuration in effect for the snapshot.
    RuntimeConfig config;
    // Timing counters collected by the dense runtime.
    ProfilingStats profiling;
    // Whether this binary was built with CUDA support.
    bool cuda_build_enabled = false;
    // Whether CUDA device buffers are available.
    bool gpu_backend_ready = false;
    // Whether the CARLsim-like GPU path is active.
    bool carlsim_like_gpu_active = false;
    // Output-neuron firing IDs visible to the legacy network.
    std::vector<int> visible_output_firing_ids;
    // All firing neuron IDs for the current step when full export is enabled.
    std::vector<int> full_firing_ids;
    // Host mirror of membrane voltage.
    std::vector<float> membrane_v;
    // Host mirror of excitatory conductance.
    std::vector<float> gexc;
    // Host mirror of inhibitory conductance.
    std::vector<float> ginh;
    // Host mirror of per-neuron firing flags.
    std::vector<unsigned char> fired;
    // Channel-major pending input buffer.
    std::vector<float> pending_channels;
    // Synaptic weights in dense layout order.
    std::vector<float> synaptic_weights;
    // Pending learning increments for synaptic weights.
    std::vector<float> synaptic_learning_pending_dwt;
    // Per-model debug state exported by neuron runtimes.
    std::vector<DenseNeuronDebugSnapshot> model_debug_states;
    // Dense output spikes emitted during the captured step.
    std::vector<DenseOutputSpike> emitted_output_spikes;
    // Recent firing history retained by the delay ring.
    std::vector<FiringStepView> firing_history_ring;
};

struct DenseSubnetworkMonitorState {
    // Simulation step captured by the lightweight monitor export.
    int time_step = 0;
    // Optional per-model State/Debug fields requested by DebugMonitorConfig::record_state.
    std::vector<DenseNeuronDebugSnapshot> model_debug_states;
    // Optional dense-local neuron ids that fired in the captured step.
    std::vector<int> current_firing_ids;
    // Optional host copy of pending[channel * stride + neuron].
    std::vector<float> pending_channels;
    int pending_channel_count = 0;
    int pending_channel_stride = 0;
    // Optional live dense synaptic weights in dense runtime order.
    std::vector<float> synaptic_weights;
};

class DenseSubnetworkRuntimeGpu {
public:
    DenseSubnetworkRuntimeGpu();
    ~DenseSubnetworkRuntimeGpu();

    // Validates the spec and allocates host/device runtime buffers.
    bool Initialize(const sim_support::DenseSubnetworkBuildSpec& spec,
                    const RuntimeConfig& config,
                    std::string* reason = nullptr);
    // Resets runtime state without retaining or re-reading the build spec.
    bool ResetState(std::string* reason = nullptr);
    // Returns true after a successful Initialize call.
    bool IsInitialized() const;

    // Starts a new timestep and clears current-step staging buffers.
    bool BeginStep(int time_step, std::string* reason = nullptr);
    // Queues one spike arriving through a dense interface slot.
    bool QueueInterfaceSpike(const DenseInterfaceSpikeInput& input, std::string* reason = nullptr);
    // Internal bridge entry used by DenseInterfaceCurrentNeuronModel. User code
    // should drive dense current through InputCurrentNeuronModel plus type=3
    // connections, matching the main-network current semantics.
    bool SetInterfaceCurrentConnection(int current_connection_index,
                                       float current,
                                       int time_step,
                                       std::string* reason = nullptr);
    // Collapses queued interface activity into per-slot state for time_step.
    bool PrepareInterfaceInputsForStep(int time_step, std::string* reason = nullptr);
    // Runs propagation, neuron update, and output collection for one step.
    bool RunStep(std::string* reason = nullptr);
    // Allocates host-side channelized interface staging.
    bool AllocateInterfaceStateVectors(std::string* reason = nullptr);
    // Releases host-side channelized interface staging.
    void FreeInterfaceStateVectors();
    // Clears host-side channelized interface staging.
    void ClearInterfaceConductanceVectors();
    // Binds a device buffer that can be copied into dense pending channels
    // without host staging. The copy specs define source index, target neuron,
    // pending channel, scale, and overwrite/accumulate behavior.
    bool BindInterfaceDeviceSource(const float* device_values,
                                   int value_count,
                                   const std::vector<DeviceToDeviceInterfaceCopySpec>& copy_specs,
                                   std::string* reason = nullptr);
    // Binds a device current buffer that can be uploaded without host staging.
    bool BindInterfaceCurrentDeviceSource(const float* device_current,
                                          int interface_count,
                                          std::string* reason = nullptr);
    // Uploads prepared host/device interface channel values into propagation buffers.
    bool UploadInterfaceChannelValues(std::string* reason = nullptr);
    // Runs the unified dense neuron update kernel/runtime.
    bool RunUnifiedNeuronUpdate(std::string* reason = nullptr);
    // Controls whether snapshots include all firing neurons or only visible outputs.
    void SetFullFiringExportEnabled(bool enabled);
    bool full_firing_export_enabled() const;

    // Runtime configuration.
    const RuntimeConfig& config() const;
    // Lower-level propagation runtime.
    const GpuPropagationRuntime& propagation_runtime() const;
    // Host-side diagnostic firing table; not used by the GPU main path.
    const DenseSubnetworkFiringTable& firing_table() const;
    // Output spikes collected for the most recent step.
    const std::vector<DenseOutputSpike>& emitted_output_spikes() const;
    // Dense output-neuron IDs that fired in the current step.
    const std::vector<int>& current_step_output_firing_ids() const;
    // Expands output-neuron firings into legacy-routable output spike records.
    std::vector<DenseOutputSpike> ExpandCurrentStepOutputSpikes() const;
    // Builds a detailed diagnostic snapshot.
    DenseSubnetworkDebugSnapshot BuildDebugSnapshot(std::string* reason = nullptr) const;
    // Downloads only the state groups requested by the unified debug monitor.
    bool ExportMonitorState(bool record_state,
                            bool record_spikes,
                            bool record_pending_channels,
                            bool record_weights,
                            DenseSubnetworkMonitorState* out,
                            std::string* reason = nullptr) const;
    // Downloads live synaptic weights from the GPU propagation runtime.
    bool DownloadSynapticWeights(std::vector<float>* weights, std::string* reason = nullptr) const;
    // Uploads live synaptic weights into the GPU propagation runtime.
    bool UploadSynapticWeights(const std::vector<float>& weights, std::string* reason = nullptr);
    // Reads one live synaptic weight from the GPU propagation runtime.
    bool GetSynapseWeight(int synapse_index, float* weight, std::string* reason = nullptr) const;
    // Writes one live synaptic weight into the GPU propagation runtime.
    bool SetSynapseWeight(int synapse_index, float weight, std::string* reason = nullptr);
    // Returns true when the mandatory CARLsim-like GPU backend is active.
    bool UsesCarlsimLikeGpuBackend() const;
    // Returns true when CUDA device buffers are allocated and usable.
    bool IsGpuBackendReady() const;

private:
    // Refreshes diagnostic host-side mirrors from GPU-owned firing history.
    bool RefreshHostDebugViews(std::string* reason) const;
    // Accumulates one interface contribution into the fixed per-slot host record table.
    bool AccumulateInterfaceBatchItemToHost(int interface_slot_index,
                                            std::uint8_t pending_channel,
                                            float value,
                                            float scale,
                                            std::string* reason);
    // Extracts dense output spikes from the current-step firing set.
    void CollectOutputSpikesForCurrentStep();
    // Caches the dense target neuron id for every external input slot.
    bool InitializeInterfaceTargetNeurons(const sim_support::DenseSubnetworkBuildSpec& spec,
                                          std::string* reason);
    // Builds per-current-connection state used to mirror legacy current semantics.
    bool InitializeInterfaceCurrentConnections(const sim_support::DenseSubnetworkBuildSpec& spec,
                                               std::string* reason);
    // Rebuilds current slots from per-connection current state before upload.
    void RebuildCurrentInterfaceSlotsFromConnections();
    // Builds reverse lookup from dense source neuron to runtime output records.
    void BuildOutputRoutingBySourceNeuron(const sim_support::DenseSubnetworkBuildSpec& spec);

    // Dense subnetwork name retained for diagnostics after build spec release.
    std::string name_;
    // Runtime tuning values consumed by the GPU main path at initialization.
    RuntimeConfig config_;
    // GPU main path: owns synapse layout, delay slices, pending input buffers,
    // firing-history device buffers, and CUDA propagation kernels.
    GpuPropagationRuntime propagation_runtime_;
    // GPU main path: owns dense neuron state and runs neuron update kernels.
    DenseUnifiedNeuronRuntime unified_neuron_runtime_;
    // Host-side diagnostic mirror populated from propagation_runtime_; not used
    // for GPU propagation scheduling.
    mutable DenseSubnetworkFiringTable firing_table_;
    // True after Initialize successfully validates descriptions and allocates buffers.
    bool initialized_;
    // Simulation step currently being prepared or executed.
    int current_time_step_;
    // Cached GPU main-path neuron count extracted from the build spec.
    int neuron_count_;
    // Number of external input interface slots staged before GPU upload.
    int interface_state_count_;
    // Fixed per-slot host records uploaded every step into propagation_runtime_.
    std::vector<InterfaceChannelValue> host_interface_channel_values_;
    // Dense interface slot selected by each current connection.
    std::vector<int> interface_current_connection_slot_indices_;
    // Latest current value for each current connection; new events overwrite it.
    std::vector<float> interface_current_connection_values_;
    // Fixed per-slot D2D copy records reused every step for CUDA-owned interface inputs.
    std::vector<DeviceToDeviceInterfaceCopySpec> device_interface_copy_specs_;
    // Timestep represented by the prepared interface staging buffers.
    int prepared_interface_time_step_;
    // Enables debug snapshots to export all firing neurons instead of visible outputs only.
    bool full_firing_export_enabled_;
    // Runtime-owned copy of synaptic weights for debug snapshots.
    mutable std::vector<float> debug_synaptic_weights_;
    // Pending channel selected for each external interface slot.
    std::vector<std::uint8_t> interface_pending_channels_;
    // Boundary spikes are staged by delivery time.  They are committed only
    // when their timestamp is strictly older than the Dense update step; this
    // prevents a queue-local Dense update from racing a same-time cross-queue
    // propagated-spike event.
    std::vector<DenseInterfaceSpikeInput> pending_interface_spikes_;
    // Model-defined scale selected for each external interface slot.
    std::vector<float> interface_scales_;
    // Interface-slot lookup used when uploading boundary input into the GPU main path.
    std::vector<int> interface_target_neuron_ids_;
    // Compact host-side output routing lookup used after GPU firing export.
    std::vector<DenseOutputRouteEntry> output_routes_by_source_neuron_;
    // Host mirror of current-step dense firings exported from the GPU update.
    std::vector<int> current_step_firing_ids_;
    // Host mirror filtered to output-visible firings exported from the GPU update.
    std::vector<int> current_step_output_firing_ids_;
    // Expanded output spike records routed back to the legacy network.
    std::vector<DenseOutputSpike> emitted_output_spikes_;
    // Optional device pointer supplying interface values directly from CUDA code.
    const float* device_interface_source_;
    // Number of values available at device_interface_source_.
    int device_interface_source_count_;
};

}  // namespace npgr

#endif
