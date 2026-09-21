#include "simulation_dense/DenseSubnetworkModel.h"

#include "simulation_dense/DenseInterfaceCurrentNeuronModel.h"
#include "simulation_dense/DenseInterfaceSpikeNeuronModel.h"
#include "simulation_dense/SimulationCommonHost.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "source_file_realtime_v1_async/EventQueue/inc/EventQueue.h"
#include "source_file_realtime_v1_async/Event/inc/Spike/PropogatedSpikeGroup.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"

#include <atomic>
#include <chrono>

namespace npgr {

namespace {

std::atomic<long long> g_dense_advance_count(0);
std::atomic<long long> g_dense_advance_total_ns(0);
std::atomic<long long> g_dense_flush_interface_ns(0);
std::atomic<long long> g_dense_event_execute_ns(0);
std::atomic<long long> g_dense_output_schedule_ns(0);
std::atomic<long long> g_dense_debug_capture_ns(0);
std::atomic<long long> g_dense_output_spike_count(0);
std::atomic<long long> g_dense_current_process_count(0);
std::atomic<long long> g_dense_current_process_ns(0);

long long DenseNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double DenseNsToMs(long long value) {
    return static_cast<double>(value) / 1000000.0;
}

}  // namespace

DenseSubnetworkModel::DenseSubnetworkModel()
    : simulation_(nullptr),
      interface_spike_model_(nullptr),
      interface_current_model_(nullptr),
      queue_index_(0),
      update_timestep_(1),
      interface_slot_count_(0) {}

DenseSubnetworkModel::~DenseSubnetworkModel() {
    if (interface_spike_model_ != nullptr) {
        delete interface_spike_model_;
        interface_spike_model_ = nullptr;
    }
    if (interface_current_model_ != nullptr) {
        delete interface_current_model_;
        interface_current_model_ = nullptr;
    }
    for (std::size_t index = 0; index < owned_output_connections_.size(); ++index) {
        delete owned_output_connections_[index];
    }
    owned_output_connections_.clear();
}

bool DenseSubnetworkModel::InitializeBlackBox(Simulation* simulation,
                                              const sim_support::DenseSubnetworkBuildSpec& spec,
                                              const LegacyNetworkBinding& binding,
                                              const RuntimeConfig& config,
                                              int queue_index,
                                              int update_timestep,
                                              std::string* reason) {
    simulation_ = simulation;
    name_ = spec.name;
    config_ = config;
    queue_index_ = queue_index;
    update_timestep_ = update_timestep > 0 ? update_timestep : 1;
    interface_slot_count_ = static_cast<int>(spec.input_target_local_ids.size());
    interface_uses_current_.assign(static_cast<std::size_t>(interface_slot_count_), 0);
    for (int index = 0; index < interface_slot_count_; ++index) {
        interface_uses_current_[static_cast<std::size_t>(index)] =
            spec.input_uses_current[static_cast<std::size_t>(index)] != 0 ? 1 : 0;
    }
    debug_synaptic_weights_ = spec.layout.synapses.weight;
    monitor_local_to_original_ids_ = spec.monitor_local_to_original_ids;
    monitor_candidate_mask_ = spec.monitor_candidate_mask;
    output_connections_.clear();
    output_queue_indices_.clear();
    output_route_start_by_source_neuron_.assign(
        static_cast<std::size_t>(spec.layout.stats.neuron_count + 1), 0);
    output_route_binding_indices_.clear();

    if (!runtime_.Initialize(spec, config_, reason)) {
        return false;
    }
    // build the interface_spike_model
    if (interface_spike_model_ == nullptr) {
        interface_spike_model_ = new DenseInterfaceSpikeNeuronModel(this);
    } else {
        interface_spike_model_->SetOwner(this);
    }
    // build the interface_current_model
    if (interface_current_model_ == nullptr) {
        interface_current_model_ = new DenseInterfaceCurrentNeuronModel(this);
    } else {
        interface_current_model_->SetOwner(this);
    }
    // The interface model only records slot timing; channel values are staged in the dense runtime.
    interface_spike_model_->ResetInterfaceBufferState(interface_slot_count_);

    owned_output_connections_.clear();
    // Bind each legacy input connection to its synthetic dense interface model.
    int current_connection_index = 0;
    for (std::size_t index = 0; index < binding.input_connections.size(); ++index) {
        Interconnections* input_connection = binding.input_connections[index];
        const int interface_slot_index =
            index < binding.input_slot_indices.size() ? binding.input_slot_indices[index] : -1;
        if (interface_slot_index < 0 || interface_slot_index >= interface_slot_count_) {
            continue;
        }
        const bool uses_current =
            interface_uses_current_[static_cast<std::size_t>(interface_slot_index)] != 0;
        const int connection_current_index = uses_current ? current_connection_index++ : -1;
        if (input_connection == nullptr) {
            continue;
        }
        input_connection->TargetNeuronModel =
            uses_current
                ? static_cast<NeuronModel*>(interface_current_model_)
                : static_cast<NeuronModel*>(interface_spike_model_);
        if (uses_current) {
            // Current delivery mirrors legacy CurrentSynapse: TargetNeuronModelIndex
            // addresses a stateful current connection, not the deduped slot.
            input_connection->SetTargetNeuronModelIndex(connection_current_index);
        } else {
            input_connection->SetTargetNeuronModelIndex(interface_slot_index);
        }
    }
    if (!binding.output_connections.empty() ||
        !binding.output_route_binding_indices.empty()) {
        return this->BindOutputConnections(binding, reason);
    }
    return true;
}

bool DenseSubnetworkModel::BindOutputConnections(const LegacyNetworkBinding& binding,
                                                 std::string* reason) {
    std::lock_guard<std::mutex> lock(input_mutex_);
    for (std::size_t index = 0; index < owned_output_connections_.size(); ++index) {
        delete owned_output_connections_[index];
    }
    owned_output_connections_.clear();
    output_connections_.clear();
    output_queue_indices_.clear();
    output_route_start_by_source_neuron_ = binding.output_route_start_by_source_neuron;
    output_route_binding_indices_ = binding.output_route_binding_indices;
    for (std::size_t index = 0; index < binding.output_connections.size(); ++index) {
        Interconnections* output_connection = binding.output_connections[index];
        if (output_connection == nullptr) {
            if (reason != nullptr) {
                *reason = "dense output binding contains null interconnection";
            }
            return false;
        }
        // get the output connections by source neuron
        output_connections_.push_back(output_connection);
        output_queue_indices_.push_back(
            index < binding.output_queue_indices.size() ? binding.output_queue_indices[index] : -1);
        if (output_connection->SourceNeuron == nullptr) {
            owned_output_connections_.push_back(output_connection);
        }
    }
    return true;
}

bool DenseSubnetworkModel::Reset(std::string* reason) {
    std::lock_guard<std::mutex> lock(input_mutex_);
    if (interface_spike_model_ != nullptr) {
        interface_spike_model_->ClearInterfaceStateBuffers();
    }
    const bool ok = runtime_.ResetState(reason);
    if (ok && interface_spike_model_ != nullptr) {
        interface_spike_model_->ResetInterfaceBufferState(interface_slot_count_);
    }
    return ok;
}

bool DenseSubnetworkModel::QueueInterfaceSpikeSlot(int interface_slot_index,
                                                   float weight,
                                                   bool inhibitory,
                                                   int time_step,
                                                   std::string* reason) {
    if (interface_spike_model_ == nullptr) {
        if (reason != nullptr) {
            *reason = "interface spike model is not initialized";
        }
        return false;
    }
    if (interface_slot_index < 0 ||
        interface_slot_index >= interface_slot_count_) {
        if (reason != nullptr) {
            *reason = "interface slot index is out of range";
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(input_mutex_);
    if (interface_uses_current_[static_cast<std::size_t>(interface_slot_index)] != 0) {
        if (reason != nullptr) {
            *reason = "interface input is configured for current delivery, not spike delivery";
        }
        return false;
    }
    DenseInterfaceSpikeInput input_event;
    input_event.interface_slot_index = interface_slot_index;
    input_event.time_step = time_step;
    input_event.weight = weight;
    input_event.inhibitory = inhibitory;
    return runtime_.QueueInterfaceSpike(input_event, reason);
}

bool DenseSubnetworkModel::SetInterfaceCurrentConnection(int current_connection_index,
                                                         float current,
                                                         int time_step,
                                                         std::string* reason) {
    if (interface_current_model_ == nullptr) {
        if (reason != nullptr) {
            *reason = "interface current model is not initialized";
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(input_mutex_);
    return runtime_.SetInterfaceCurrentConnection(current_connection_index, current, time_step, reason);
}

bool DenseSubnetworkModel::BindInterfaceCurrentDeviceSource(const float* device_current,
                                                            int interface_count,
                                                            std::string* reason) {
    std::lock_guard<std::mutex> lock(input_mutex_);
    return runtime_.BindInterfaceCurrentDeviceSource(device_current, interface_count, reason);
}

bool DenseSubnetworkModel::BindInterfaceDeviceSource(
    const float* device_values,
    int value_count,
    const std::vector<DeviceToDeviceInterfaceCopySpec>& copy_specs,
    std::string* reason) {
    std::lock_guard<std::mutex> lock(input_mutex_);
    return runtime_.BindInterfaceDeviceSource(device_values, value_count, copy_specs, reason);
}

bool DenseSubnetworkModel::FlushInterfaceStateBuffers(int time_step, std::string* reason) {
    if (!runtime_.PrepareInterfaceInputsForStep(time_step, reason)) {
        return false;
    }
    return runtime_.UploadInterfaceChannelValues(reason);
}

bool DenseSubnetworkModel::AdvanceStep(int time_step, EventQueue* event_queue, std::string* reason) {
    const long long advance_start_ns = DenseNowNs();
    std::lock_guard<std::mutex> lock(input_mutex_);
    // add the interface spikes to the spike buffer
    const long long flush_start_ns = DenseNowNs();
    if (!this->FlushInterfaceStateBuffers(time_step, reason)) {
        return false;
    }
    g_dense_flush_interface_ns.fetch_add(DenseNowNs() - flush_start_ns, std::memory_order_relaxed);
    DenseSubnetworkEvent event(time_step);
    DenseSubnetworkEventResult result;
    // run the subnetwork
    const long long execute_start_ns = DenseNowNs();
    if (!event.Execute(&runtime_, &result, reason)) {
        return false;
    }
    g_dense_event_execute_ns.fetch_add(DenseNowNs() - execute_start_ns, std::memory_order_relaxed);
    if (event_queue == nullptr) {
        if (reason != nullptr) {
            *reason = "event_queue must not be null";
        }
        return false;
    }
    const long long output_start_ns = DenseNowNs();
    for (std::size_t firing_index = 0; firing_index < result.output_firing_ids.size(); ++firing_index) {
        const int source_neuron = result.output_firing_ids[firing_index];
        if (source_neuron < 0 ||
            source_neuron + 1 >= static_cast<int>(output_route_start_by_source_neuron_.size())) {
            continue;
        }
        const int route_begin =
            output_route_start_by_source_neuron_[static_cast<std::size_t>(source_neuron)];
        const int route_end =
            output_route_start_by_source_neuron_[static_cast<std::size_t>(source_neuron + 1)];
        for (int route_index = route_begin; route_index < route_end; ++route_index) {
            if (route_index < 0 ||
                route_index >= static_cast<int>(output_route_binding_indices_.size())) {
                if (reason != nullptr) {
                    *reason = "dense output route contains invalid CSR offset";
                }
                return false;
            }
            const int output_index =
                output_route_binding_indices_[static_cast<std::size_t>(route_index)];
            if (output_index < 0 ||
                output_index >= static_cast<int>(output_connections_.size())) {
                if (reason != nullptr) {
                    *reason = "dense output route contains invalid binding index";
                }
                return false;
            }
            Interconnections* connection = output_connections_[static_cast<std::size_t>(output_index)];
            if (connection == nullptr) {
                if (reason != nullptr) {
                    *reason = "dense output binding contains null interconnection";
                }
                return false;
            }
            const int output_queue_index = output_queue_indices_[static_cast<std::size_t>(output_index)];
            PropogatedSpikeGroup* group =
                new PropogatedSpikeGroup(result.time_step + connection->delay, output_queue_index);
            group->IncludeNewSourceNeuron(1, connection);
            group->SourceNeuron = connection->SourceNeuron;
            if (output_queue_index == queue_index_) {
                event_queue->Insert_a_Event(group, output_queue_index);
            } else {
                // EventQueue heaps are owned by their worker queue and are
                // not safe for direct cross-thread insertion. Use the
                // producer-to-consumer buffer, which is flushed by the
                // synchronization event.
                event_queue->Insert_a_Event_to_Buffer(
                    group, queue_index_, output_queue_index);
            }
            g_dense_output_spike_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_dense_output_schedule_ns.fetch_add(DenseNowNs() - output_start_ns, std::memory_order_relaxed);
    const long long debug_start_ns = DenseNowNs();
    npgr::sim_support::CaptureDenseSubnetworkDebugSnapshot(simulation_, this, time_step);
    g_dense_debug_capture_ns.fetch_add(DenseNowNs() - debug_start_ns, std::memory_order_relaxed);
    g_dense_advance_count.fetch_add(1, std::memory_order_relaxed);
    g_dense_advance_total_ns.fetch_add(DenseNowNs() - advance_start_ns, std::memory_order_relaxed);
    return true;
}

void DenseSubnetworkModel::SetFullFiringExportEnabled(bool enabled) {
    runtime_.SetFullFiringExportEnabled(enabled);
}

bool DenseSubnetworkModel::full_firing_export_enabled() const {
    return runtime_.full_firing_export_enabled();
}

bool DenseSubnetworkModel::OwnsInternalModel(const NeuronModel* model) const {
    (void)model;
    return false;
}

int DenseSubnetworkModel::queue_index() const {
    return queue_index_;
}

int DenseSubnetworkModel::update_timestep() const {
    return update_timestep_;
}

int DenseSubnetworkModel::current_time_step() const {
    if (simulation_ == nullptr || simulation_->currenttime == nullptr || queue_index_ < 0) {
        return 0;
    }
    return simulation_->currenttime[queue_index_];
}

DenseInterfaceSpikeNeuronModel* DenseSubnetworkModel::interface_spike_model() const {
    return interface_spike_model_;
}

DenseInterfaceCurrentNeuronModel* DenseSubnetworkModel::interface_current_model() const {
    return interface_current_model_;
}

const std::string& DenseSubnetworkModel::name() const {
    return name_;
}

int DenseSubnetworkModel::OutputRouteCountForSourceNeuron(int source_neuron_id) const {
    if (source_neuron_id < 0 ||
        source_neuron_id + 1 >= static_cast<int>(output_route_start_by_source_neuron_.size())) {
        return 0;
    }
    const int route_begin =
        output_route_start_by_source_neuron_[static_cast<std::size_t>(source_neuron_id)];
    const int route_end =
        output_route_start_by_source_neuron_[static_cast<std::size_t>(source_neuron_id + 1)];
    return route_end >= route_begin ? route_end - route_begin : 0;
}

bool DenseSubnetworkModel::HasMonitorTargets() const {
    for (std::size_t index = 0; index < monitor_candidate_mask_.size(); ++index) {
        if (monitor_candidate_mask_[index] != 0) {
            return true;
        }
    }
    return false;
}

int DenseSubnetworkModel::OriginalGlobalNeuronIdForLocal(int local_neuron_id) const {
    if (local_neuron_id < 0 ||
        local_neuron_id >= static_cast<int>(monitor_local_to_original_ids_.size())) {
        return -1;
    }
    return monitor_local_to_original_ids_[static_cast<std::size_t>(local_neuron_id)];
}

int DenseSubnetworkModel::LocalNeuronIdForOriginalGlobal(int original_neuron_id) const {
    if (original_neuron_id < 0) {
        return -1;
    }
    for (std::size_t local_id = 0; local_id < monitor_local_to_original_ids_.size(); ++local_id) {
        if (monitor_local_to_original_ids_[local_id] == original_neuron_id) {
            return static_cast<int>(local_id);
        }
    }
    return -1;
}

bool DenseSubnetworkModel::IsMonitorCandidateLocalNeuron(int local_neuron_id) const {
    if (local_neuron_id < 0 ||
        local_neuron_id >= static_cast<int>(monitor_candidate_mask_.size())) {
        return false;
    }
    return monitor_candidate_mask_[static_cast<std::size_t>(local_neuron_id)] != 0;
}

DenseSubnetworkDebugSnapshot DenseSubnetworkModel::BuildDebugSnapshot(std::string* reason) const {
    std::lock_guard<std::mutex> lock(input_mutex_);
    return runtime_.BuildDebugSnapshot(reason);
}

bool DenseSubnetworkModel::ExportMonitorState(bool record_state,
                                              bool record_spikes,
                                              bool record_pending_channels,
                                              bool record_weights,
                                              DenseSubnetworkMonitorState* out,
                                              std::string* reason) const {
    std::lock_guard<std::mutex> lock(input_mutex_);
    return runtime_.ExportMonitorState(record_state, record_spikes, record_pending_channels, record_weights, out, reason);
}

std::vector<float> DenseSubnetworkModel::GetSynapticWeights() const {
    std::lock_guard<std::mutex> lock(input_mutex_);
    return debug_synaptic_weights_;
}

bool DenseSubnetworkModel::DownloadLiveSynapticWeights(std::vector<float>* weights,
                                                       std::string* reason) const {
    std::lock_guard<std::mutex> lock(input_mutex_);
    if (!runtime_.DownloadSynapticWeights(weights, reason)) {
        return false;
    }
    debug_synaptic_weights_ = *weights;
    return true;
}

bool DenseSubnetworkModel::UploadLiveSynapticWeights(const std::vector<float>& weights,
                                                     std::string* reason) {
    std::lock_guard<std::mutex> lock(input_mutex_);
    if (!runtime_.UploadSynapticWeights(weights, reason)) {
        return false;
    }
    debug_synaptic_weights_ = weights;
    return true;
}

bool DenseSubnetworkModel::GetLiveSynapseWeight(int synapse_index,
                                                float* weight,
                                                std::string* reason) const {
    std::lock_guard<std::mutex> lock(input_mutex_);
    return runtime_.GetSynapseWeight(synapse_index, weight, reason);
}

bool DenseSubnetworkModel::SetLiveSynapseWeight(int synapse_index,
                                                float weight,
                                                std::string* reason) {
    std::lock_guard<std::mutex> lock(input_mutex_);
    if (!runtime_.SetSynapseWeight(synapse_index, weight, reason)) {
        return false;
    }
    if (synapse_index >= 0 &&
        synapse_index < static_cast<int>(debug_synaptic_weights_.size())) {
        debug_synaptic_weights_[static_cast<std::size_t>(synapse_index)] = weight;
    }
    return true;
}

void DenseSubnetworkModel::ResetHostProfiling() {
    g_dense_advance_count.store(0, std::memory_order_relaxed);
    g_dense_advance_total_ns.store(0, std::memory_order_relaxed);
    g_dense_flush_interface_ns.store(0, std::memory_order_relaxed);
    g_dense_event_execute_ns.store(0, std::memory_order_relaxed);
    g_dense_output_schedule_ns.store(0, std::memory_order_relaxed);
    g_dense_debug_capture_ns.store(0, std::memory_order_relaxed);
    g_dense_output_spike_count.store(0, std::memory_order_relaxed);
    g_dense_current_process_count.store(0, std::memory_order_relaxed);
    g_dense_current_process_ns.store(0, std::memory_order_relaxed);
}

DenseSubnetworkHostProfiling DenseSubnetworkModel::HostProfilingSnapshot() {
    DenseSubnetworkHostProfiling snapshot;
    snapshot.advance_count = g_dense_advance_count.load(std::memory_order_relaxed);
    snapshot.advance_total_ms = DenseNsToMs(g_dense_advance_total_ns.load(std::memory_order_relaxed));
    snapshot.flush_interface_ms = DenseNsToMs(g_dense_flush_interface_ns.load(std::memory_order_relaxed));
    snapshot.event_execute_ms = DenseNsToMs(g_dense_event_execute_ns.load(std::memory_order_relaxed));
    snapshot.output_schedule_ms = DenseNsToMs(g_dense_output_schedule_ns.load(std::memory_order_relaxed));
    snapshot.debug_capture_ms = DenseNsToMs(g_dense_debug_capture_ns.load(std::memory_order_relaxed));
    snapshot.output_spike_count = g_dense_output_spike_count.load(std::memory_order_relaxed);
    snapshot.current_process_count = g_dense_current_process_count.load(std::memory_order_relaxed);
    snapshot.current_process_ms = DenseNsToMs(g_dense_current_process_ns.load(std::memory_order_relaxed));
    return snapshot;
}

void DenseSubnetworkModel::RecordCurrentProcessTime(long long elapsed_ns) {
    g_dense_current_process_count.fetch_add(1, std::memory_order_relaxed);
    g_dense_current_process_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
}

}  // namespace npgr
