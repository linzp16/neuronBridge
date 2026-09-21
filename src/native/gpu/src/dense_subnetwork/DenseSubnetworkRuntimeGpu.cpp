#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"
#include "dense_subnetwork/DenseSubnetworkOutputExtractor.h"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

namespace npgr {

namespace {

std::vector<DenseNeuronModelSpec> BuildFallbackNeuronModels(
    const sim_support::DenseSubnetworkBuildSpec& runtime_spec) {
    DenseNeuronModelSpec model_spec;
    model_spec.factory_model_id = DenseNeuronModelFactory::kLifExponentialDoubleModelId;
    model_spec.legacy_model_name = "TimeDrivenLIF_Exponential_double";
    model_spec.model_id = 0;
    model_spec.range.begin = 0;
    model_spec.range.count = runtime_spec.layout.stats.neuron_count;
    return std::vector<DenseNeuronModelSpec>(1, model_spec);
}

std::vector<float> FindFirstFloatDebugField(const std::vector<DenseNeuronDebugSnapshot>& snapshots,
                                            const char* primary,
                                            const char* fallback = nullptr) {
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
        std::map<std::string, std::vector<float> >::const_iterator found =
            snapshots[index].float_state_vectors.find(primary);
        if (found != snapshots[index].float_state_vectors.end()) {
            return found->second;
        }
        if (fallback != nullptr) {
            found = snapshots[index].float_state_vectors.find(fallback);
            if (found != snapshots[index].float_state_vectors.end()) {
                return found->second;
            }
        }
    }
    return std::vector<float>();
}

std::vector<unsigned char> FindFirstByteDebugField(const std::vector<DenseNeuronDebugSnapshot>& snapshots,
                                                   const char* name) {
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
        std::map<std::string, std::vector<unsigned char> >::const_iterator found =
            snapshots[index].byte_state_vectors.find(name);
        if (found != snapshots[index].byte_state_vectors.end()) {
            return found->second;
        }
    }
    return std::vector<unsigned char>();
}

}  // namespace

DenseSubnetworkRuntimeGpu::DenseSubnetworkRuntimeGpu()
    : name_(),
      initialized_(false),
      current_time_step_(0),
      neuron_count_(0),
      interface_state_count_(0),
      prepared_interface_time_step_(-1),
      full_firing_export_enabled_(false),
      device_interface_source_(nullptr),
      device_interface_source_count_(0) {}

DenseSubnetworkRuntimeGpu::~DenseSubnetworkRuntimeGpu() {
    this->FreeInterfaceStateVectors();
}

bool DenseSubnetworkRuntimeGpu::Initialize(const sim_support::DenseSubnetworkBuildSpec& spec,
                                           const RuntimeConfig& config,
                                           std::string* reason) {
    if (!spec.IsValid(reason)) {
        return false;
    }
    // set the config
    config_ = config;
    // set the subnetwork name
    name_ = spec.name;
    // init the debuged weight
    debug_synaptic_weights_ = spec.layout.synapses.weight;
    // clear the debug monitor spike buffer(unused now)
    emitted_output_spikes_.clear();
    // clear the debug current spikes
    current_step_firing_ids_.clear();
    // clear the current step's output firing table
    current_step_output_firing_ids_.clear();
    current_time_step_ = 0;
    neuron_count_ = spec.layout.stats.neuron_count;
    // get the input target slot's size
    interface_state_count_ = static_cast<int>(spec.input_target_local_ids.size());
    prepared_interface_time_step_ = -1;
    full_firing_export_enabled_ = false;
    device_interface_source_ = nullptr;
    device_interface_source_count_ = 0;
    interface_target_neuron_ids_.clear();
    interface_pending_channels_.clear();
    interface_scales_.clear();
    output_routes_by_source_neuron_.clear();
    // Init the host firing table(only for debug)
    if (!firing_table_.Initialize(config_.steps_to_keep, reason)) {
        initialized_ = false;
        return false;
    }
    // Init the propagation structure engine
    if (!propagation_runtime_.Initialize(spec.layout, config_, reason)) {
        initialized_ = false;
        return false;
    }
    // buid the subnetwork output structures
    this->BuildOutputRoutingBySourceNeuron(spec);
    // allocate the Interface of the neuron state vector
    if (!this->AllocateInterfaceStateVectors(reason) ||
        !this->InitializeInterfaceTargetNeurons(spec, reason) ||
        !this->InitializeInterfaceCurrentConnections(spec, reason)) {
        initialized_ = false;
        return false;
    }
    if (!propagation_runtime_.IsCarlsimLikeGpuPathActive()) {
        if (reason != nullptr) {
            if (!propagation_runtime_.IsCudaEnabledBuild()) {
                *reason = "dense subnetwork requires CUDA build because GPU kernel is mandatory";
            } else if (!propagation_runtime_.IsDeviceReady()) {
                *reason = "dense subnetwork requires a ready CUDA device backend, but device buffers are unavailable";
            } else {
                *reason = "dense subnetwork failed to activate the  GPU backend";
            }
        }
        initialized_ = false;
        return false;
    }
    std::vector<DenseNeuronModelSpec> model_specs = spec.neuron_models;
    if (model_specs.empty()) {
        model_specs = BuildFallbackNeuronModels(spec);
    }
    // copy the model parameters to the device
    if (!unified_neuron_runtime_.Initialize(model_specs,
                                            spec.layout.stats.neuron_count,
                                            config_.dt_ms,
                                            reason) ||
        !unified_neuron_runtime_.ConfigureOutputNeuronMask(spec.output_neuron_mask, reason) ||
        !unified_neuron_runtime_.AllocateDeviceBuffers(reason) ||
        !unified_neuron_runtime_.UploadHostToDevice(reason)) {
        initialized_ = false;
        return false;
    }
    initialized_ = true;
    return true;
}

bool DenseSubnetworkRuntimeGpu::IsInitialized() const {
    return initialized_;
}

bool DenseSubnetworkRuntimeGpu::ResetState(std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    emitted_output_spikes_.clear();
    current_step_firing_ids_.clear();
    current_step_output_firing_ids_.clear();
    current_time_step_ = 0;
    prepared_interface_time_step_ = -1;
    pending_interface_spikes_.clear();
    device_interface_source_ = nullptr;
    device_interface_source_count_ = 0;
    std::fill(interface_current_connection_values_.begin(),
              interface_current_connection_values_.end(),
              0.0f);
    this->ClearInterfaceConductanceVectors();
    if (!firing_table_.Initialize(config_.steps_to_keep, reason)) {
        return false;
    }
    if (!propagation_runtime_.ResetState(reason)) {
        return false;
    }
    return unified_neuron_runtime_.ResetState(reason);
}

bool DenseSubnetworkRuntimeGpu::AllocateInterfaceStateVectors(std::string* reason) {
    this->FreeInterfaceStateVectors();
    if (interface_state_count_ < 0) {
        if (reason != nullptr) {
            *reason = "interface_state_count must be non-negative";
        }
        return false;
    }
    if (interface_state_count_ == 0) {
        return true;
    }
    // assign the interface state vectors(for the host to device interface neuron)
    host_interface_channel_values_.assign(static_cast<std::size_t>(interface_state_count_),
                                          InterfaceChannelValue{});
    // assign the interface copy specs( for the device to device inputConv copy)
    device_interface_copy_specs_.assign(static_cast<std::size_t>(interface_state_count_),
                                        DeviceToDeviceInterfaceCopySpec{});
    // clean the current interface values
    this->ClearInterfaceConductanceVectors();
    return true;
}

void DenseSubnetworkRuntimeGpu::FreeInterfaceStateVectors() {
    host_interface_channel_values_.clear();
    host_interface_channel_values_.shrink_to_fit();
    device_interface_copy_specs_.clear();
    device_interface_copy_specs_.shrink_to_fit();
    interface_current_connection_slot_indices_.clear();
    interface_current_connection_slot_indices_.shrink_to_fit();
    interface_current_connection_values_.clear();
    interface_current_connection_values_.shrink_to_fit();
}

void DenseSubnetworkRuntimeGpu::ClearInterfaceConductanceVectors() {
    for (InterfaceChannelValue& value : host_interface_channel_values_) {
        value.value = 0.0f;
    }
}

bool DenseSubnetworkRuntimeGpu::BindInterfaceDeviceSource(
    const float* device_values,
    int value_count,
    const std::vector<DeviceToDeviceInterfaceCopySpec>& copy_specs,
    std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    if (device_values == nullptr) {
        if (reason != nullptr) {
            *reason = "device interface source pointer must not be null";
        }
        return false;
    }
    if (value_count <= 0) {
        if (reason != nullptr) {
            *reason = "device interface source count must be positive";
        }
        return false;
    }
    if (copy_specs.empty()) {
        if (reason != nullptr) {
            *reason = "device interface copy specs must not be empty";
        }
        return false;
    }
    for (std::size_t index = 0; index < copy_specs.size(); ++index) {
        const DeviceToDeviceInterfaceCopySpec& spec = copy_specs[index];
        if (spec.source_index < 0 || spec.source_index >= value_count) {
            if (reason != nullptr) {
                *reason = "device interface source index is out of range";
            }
            return false;
        }
        if (spec.target_neuron < 0 || spec.target_neuron >= neuron_count_) {
            if (reason != nullptr) {
                *reason = "device interface target neuron is out of range";
            }
            return false;
        }
    }
    device_interface_source_ = device_values;
    device_interface_source_count_ = value_count;
    device_interface_copy_specs_ = copy_specs;
    return true;
}

bool DenseSubnetworkRuntimeGpu::BindInterfaceCurrentDeviceSource(const float* device_current,
                                                                 int interface_count,
                                                                 std::string* reason) {
    if (interface_count <= 0) {
        if (reason != nullptr) {
            *reason = "device interface current source count must be positive";
        }
        return false;
    }
    std::vector<DeviceToDeviceInterfaceCopySpec> copy_specs(
        static_cast<std::size_t>(interface_count),
        DeviceToDeviceInterfaceCopySpec{});
    for (int interface_index = 0; interface_index < interface_count; ++interface_index) {
        DeviceToDeviceInterfaceCopySpec& copy_record =
            copy_specs[static_cast<std::size_t>(interface_index)];
        copy_record.interface_slot = interface_index;
        copy_record.target_neuron =
            interface_index < static_cast<int>(interface_target_neuron_ids_.size())
                ? interface_target_neuron_ids_[static_cast<std::size_t>(interface_index)]
                : -1;
        copy_record.pending_channel =
            interface_index < static_cast<int>(interface_pending_channels_.size())
                ? interface_pending_channels_[static_cast<std::size_t>(interface_index)]
                : static_cast<std::uint8_t>(PendingChannel::Current);
        copy_record.source_index = interface_index;
        copy_record.scale =
            interface_index < static_cast<int>(interface_scales_.size())
                ? interface_scales_[static_cast<std::size_t>(interface_index)]
                : 1.0f;
        copy_record.overwrite =
            copy_record.pending_channel == static_cast<std::uint8_t>(PendingChannel::Current);
    }
    return this->BindInterfaceDeviceSource(device_current, interface_count, copy_specs, reason);
}

bool DenseSubnetworkRuntimeGpu::InitializeInterfaceTargetNeurons(
    const sim_support::DenseSubnetworkBuildSpec& spec,
    std::string* reason) {
    // first init the initial value
    interface_target_neuron_ids_.assign(static_cast<std::size_t>(interface_state_count_), -1);
    interface_pending_channels_.assign(
        static_cast<std::size_t>(interface_state_count_),
        static_cast<std::uint8_t>(PendingChannel::ExcitatoryConductance));
    interface_scales_.assign(static_cast<std::size_t>(interface_state_count_), 1.0f);
    host_interface_channel_values_.resize(static_cast<std::size_t>(interface_state_count_));
    device_interface_copy_specs_.resize(static_cast<std::size_t>(interface_state_count_));
    // scan all the interface states
    for (int interface_index = 0; interface_index < interface_state_count_; ++interface_index) {
        if (spec.input_target_local_ids[static_cast<std::size_t>(interface_index)] < 0 ||
            spec.input_target_local_ids[static_cast<std::size_t>(interface_index)] >= neuron_count_) {
            if (reason != nullptr) {
                *reason = "interface target neuron is out of range";
            }
            return false;
        }
        // 
        interface_target_neuron_ids_[static_cast<std::size_t>(interface_index)] =
            spec.input_target_local_ids[static_cast<std::size_t>(interface_index)];
        interface_pending_channels_[static_cast<std::size_t>(interface_index)] =
            spec.input_pending_channels[static_cast<std::size_t>(interface_index)];
        interface_scales_[static_cast<std::size_t>(interface_index)] =
            spec.input_scales[static_cast<std::size_t>(interface_index)];
        // init the host to device copy
        InterfaceChannelValue& host_record =
            host_interface_channel_values_[static_cast<std::size_t>(interface_index)];
        host_record.interface_slot = interface_index;
        // get the target neuron id 
        host_record.target_neuron =
            interface_target_neuron_ids_[static_cast<std::size_t>(interface_index)];
        // get the pending channels
        host_record.pending_channel =
            interface_pending_channels_[static_cast<std::size_t>(interface_index)];
        // get the value and scale
        host_record.value = 0.0f;
        host_record.scale = interface_scales_[static_cast<std::size_t>(interface_index)];
        // the current will use the overtwrite and the spike mark will use the accumulate
        host_record.overwrite =
            host_record.pending_channel == static_cast<std::uint8_t>(PendingChannel::Current) ? 1 : 0;
        // init the device to device copy
        DeviceToDeviceInterfaceCopySpec& copy_record =
            device_interface_copy_specs_[static_cast<std::size_t>(interface_index)];
        // get the interface slot
        copy_record.interface_slot = interface_index;
        copy_record.target_neuron =
            interface_target_neuron_ids_[static_cast<std::size_t>(interface_index)];
        copy_record.pending_channel =
            interface_pending_channels_[static_cast<std::size_t>(interface_index)];
        copy_record.source_index = interface_index;
        copy_record.scale = interface_scales_[static_cast<std::size_t>(interface_index)];
        copy_record.overwrite =
            copy_record.pending_channel == static_cast<std::uint8_t>(PendingChannel::Current);
    }
    return propagation_runtime_.SetInterfaceTargetNeurons(interface_target_neuron_ids_, reason);
}

bool DenseSubnetworkRuntimeGpu::InitializeInterfaceCurrentConnections(
    const sim_support::DenseSubnetworkBuildSpec& spec,
    std::string* reason) {
    interface_current_connection_slot_indices_.clear();
    interface_current_connection_values_.clear();
    for (std::size_t input_index = 0; input_index < spec.input_connection_slot_indices.size(); ++input_index) {
        const int slot_index = spec.input_connection_slot_indices[input_index];
        if (slot_index < 0 || slot_index >= interface_state_count_) {
            if (reason != nullptr) {
                *reason = "interface input connection references an invalid slot";
            }
            return false;
        }
        if (slot_index >= static_cast<int>(spec.input_uses_current.size()) ||
            spec.input_uses_current[static_cast<std::size_t>(slot_index)] == 0) {
            continue;
        }
        // Legacy current delivery is stateful per connection. The runtime keeps
        // the latest value for each current connection, then sums all values
        // that feed the same current slot before uploading the step record.
        interface_current_connection_slot_indices_.push_back(slot_index);
        interface_current_connection_values_.push_back(0.0f);
    }
    return true;
}

void DenseSubnetworkRuntimeGpu::RebuildCurrentInterfaceSlotsFromConnections() {
    for (int slot_index = 0; slot_index < interface_state_count_; ++slot_index) {
        if (slot_index < static_cast<int>(interface_pending_channels_.size()) &&
            interface_pending_channels_[static_cast<std::size_t>(slot_index)] ==
                static_cast<std::uint8_t>(PendingChannel::Current)) {
            InterfaceChannelValue& record =
                host_interface_channel_values_[static_cast<std::size_t>(slot_index)];
            record.value = 0.0f;
            record.overwrite = 1;
        }
    }
    for (std::size_t current_index = 0;
         current_index < interface_current_connection_slot_indices_.size();
         ++current_index) {
        const int slot_index = interface_current_connection_slot_indices_[current_index];
        if (slot_index < 0 || slot_index >= interface_state_count_) {
            continue;
        }
        host_interface_channel_values_[static_cast<std::size_t>(slot_index)].value +=
            interface_current_connection_values_[current_index];
    }
}

bool DenseSubnetworkRuntimeGpu::UploadInterfaceChannelValues(std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    if (interface_state_count_ > 0) {
        this->RebuildCurrentInterfaceSlotsFromConnections();
        // Upload host-owned interface records produced by main-network
        // boundary spike/current delivery.
        if (!propagation_runtime_.UploadInterfaceChannelValues(host_interface_channel_values_, reason)) {
            return false;
        }
    }
    if (device_interface_source_ != nullptr && device_interface_source_count_ > 0) {
        // Device-to-device records, such as InputConv -> dense current binding,
        // are independent of synthetic main->dense interface slots.
        if (!propagation_runtime_.UploadInterfaceChannelValuesDevice(
                device_interface_source_,
                device_interface_copy_specs_,
                reason)) {
            return false;
        }
    }
    return true;
}

bool DenseSubnetworkRuntimeGpu::RunUnifiedNeuronUpdate(std::string* reason) {
    const DeviceCommonBuffersView common = propagation_runtime_.GetDeviceCommonBuffersView();
    current_step_output_firing_ids_.clear();
    current_step_firing_ids_.clear();
    if (!unified_neuron_runtime_.Step(common,
                                      current_time_step_,
                                      &current_step_output_firing_ids_,
                                      &current_step_firing_ids_,
                                      full_firing_export_enabled_,
                                      reason)) {
        return false;
    }
    if (!propagation_runtime_.ApplyPostLearningFromDeviceFirings(
            unified_neuron_runtime_.device_current_did_fire(),
            reason)) {
        return false;
    }
    return propagation_runtime_.CommitCurrentDeviceFiringsFromExternal(
        current_time_step_,
        unified_neuron_runtime_.device_full_firing_ids(),
        unified_neuron_runtime_.device_full_firing_count(),
        reason);
}

bool DenseSubnetworkRuntimeGpu::BeginStep(int time_step, std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    current_time_step_ = time_step;
    emitted_output_spikes_.clear();
    current_step_firing_ids_.clear();
    current_step_output_firing_ids_.clear();
    if (!firing_table_.BeginStep(time_step, reason)) {
        return false;
    }
    return propagation_runtime_.BeginStep(time_step, reason);
}

bool DenseSubnetworkRuntimeGpu::QueueInterfaceSpike(const DenseInterfaceSpikeInput& input, std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    if (input.time_step < 0) {
        if (reason != nullptr) {
            *reason = "interface spike input time_step must be non-negative";
        }
        return false;
    }
    if (input.interface_slot_index < 0 || input.interface_slot_index >= interface_state_count_) {
        if (reason != nullptr) {
            *reason = "interface spike input slot index is out of range";
        }
        return false;
    }
    if (input.interface_slot_index < static_cast<int>(interface_pending_channels_.size()) &&
        interface_pending_channels_[static_cast<std::size_t>(input.interface_slot_index)] ==
            static_cast<std::uint8_t>(PendingChannel::Current)) {
        if (reason != nullptr) {
            *reason =
                "current interface slots must be driven by InputCurrentNeuronModel/type=3 connections";
        }
        return false;
    }
    // Do not accumulate directly into the current-step host buffer.  A
    // cross-queue propagated spike with this timestamp may still be processed
    // after the Dense update event on another queue.  Retain the timestamp and
    // commit it from PrepareInterfaceInputsForStep once the step is complete.
    pending_interface_spikes_.push_back(input);
    prepared_interface_time_step_ = -1;
    return true;
}

bool DenseSubnetworkRuntimeGpu::SetInterfaceCurrentConnection(int current_connection_index,
                                                              float current,
                                                              int time_step,
                                                              std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    if (time_step < 0) {
        if (reason != nullptr) {
            *reason = "interface current time_step must be non-negative";
        }
        return false;
    }
    if (current_connection_index < 0 ||
        current_connection_index >= static_cast<int>(interface_current_connection_values_.size())) {
        if (reason != nullptr) {
            *reason = "interface current connection index is out of range";
        }
        return false;
    }
    // Match legacy CurrentSynapse semantics: the same current connection keeps
    // only its latest value, while multiple current connections are summed.
    interface_current_connection_values_[static_cast<std::size_t>(current_connection_index)] = current;
    prepared_interface_time_step_ = -1;
    return true;
}

bool DenseSubnetworkRuntimeGpu::AccumulateInterfaceBatchItemToHost(int interface_slot_index,
                                                                   std::uint8_t pending_channel,
                                                                   float value,
                                                                   float scale,
                                                                   std::string* reason) {
    if (interface_slot_index < 0 ||
        interface_slot_index >= interface_state_count_) {
        if (reason != nullptr) {
            *reason = "interface_input_index is out of range for host interface state buffers";
        }
        return false;
    }
    if (static_cast<int>(host_interface_channel_values_.size()) != interface_state_count_) {
        if (reason != nullptr) {
            *reason = "fixed host interface record table is not initialized";
        }
        return false;
    }
    InterfaceChannelValue& channel_value =
        host_interface_channel_values_[static_cast<std::size_t>(interface_slot_index)];
    channel_value.interface_slot = interface_slot_index;
    channel_value.target_neuron =
        interface_slot_index < static_cast<int>(interface_target_neuron_ids_.size())
            ? interface_target_neuron_ids_[static_cast<std::size_t>(interface_slot_index)]
            : -1;
    channel_value.pending_channel = pending_channel;
    channel_value.value += value;
    channel_value.scale = scale;
    return true;
}

bool DenseSubnetworkRuntimeGpu::PrepareInterfaceInputsForStep(int time_step, std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    if (time_step < 0) {
        if (reason != nullptr) {
            *reason = "interface input prepare time_step must be non-negative";
        }
        return false;
    }
    if (prepared_interface_time_step_ == time_step) {
        return true;
    }
    // Commit only inputs from completed simulation steps.  Same-time inputs
    // remain pending until the next Dense update, after every OpenMP queue has
    // had an opportunity to process its propagated-spike events.
    std::vector<DenseInterfaceSpikeInput> remaining;
    remaining.reserve(pending_interface_spikes_.size());
    for (const DenseInterfaceSpikeInput& input : pending_interface_spikes_) {
        if (input.time_step < time_step) {
            if (!this->AccumulateInterfaceBatchItemToHost(
                    input.interface_slot_index,
                    interface_pending_channels_[static_cast<std::size_t>(input.interface_slot_index)],
                    input.weight,
                    interface_scales_[static_cast<std::size_t>(input.interface_slot_index)],
                    reason)) {
                return false;
            }
        } else {
            remaining.push_back(input);
        }
    }
    pending_interface_spikes_.swap(remaining);
    prepared_interface_time_step_ = time_step;
    return true;
}

bool DenseSubnetworkRuntimeGpu::RunStep(std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime is not initialized";
        }
        return false;
    }
    if (!propagation_runtime_.IsCarlsimLikeGpuPathActive()) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime lost its CARLsim-like GPU backend";
        }
        return false;
    }
    if (!this->PrepareInterfaceInputsForStep(current_time_step_, reason)) {
        return false;
    }
    // upload the interface aux neuronstate to the device
    if (!this->UploadInterfaceChannelValues(reason)) {
        return false;
    }
    // add the interface increaments to the device
    if (!propagation_runtime_.ApplyInterfaceStateVectors(reason)) {
        return false;
    }
    // run the propagation step on the device
    if (!propagation_runtime_.RunPropagationOnly(reason)) {
        return false;
    }
    // update the neuron state ont the device
    if (!this->RunUnifiedNeuronUpdate(reason)) {
        return false;
    }
    emitted_output_spikes_.clear();
    if (prepared_interface_time_step_ == current_time_step_) {
        prepared_interface_time_step_ = -1;
    }
    this->ClearInterfaceConductanceVectors();
    return true;
}

const RuntimeConfig& DenseSubnetworkRuntimeGpu::config() const {
    return config_;
}

const GpuPropagationRuntime& DenseSubnetworkRuntimeGpu::propagation_runtime() const {
    return propagation_runtime_;
}

const DenseSubnetworkFiringTable& DenseSubnetworkRuntimeGpu::firing_table() const {
    this->RefreshHostDebugViews(nullptr);
    return firing_table_;
}

const std::vector<DenseOutputSpike>& DenseSubnetworkRuntimeGpu::emitted_output_spikes() const {
    return emitted_output_spikes_;
}

const std::vector<int>& DenseSubnetworkRuntimeGpu::current_step_output_firing_ids() const {
    return current_step_output_firing_ids_;
}

std::vector<DenseOutputSpike> DenseSubnetworkRuntimeGpu::ExpandCurrentStepOutputSpikes() const {
    return DenseSubnetworkOutputExtractor::Extract(
        output_routes_by_source_neuron_, current_step_output_firing_ids_, current_time_step_);
}

void DenseSubnetworkRuntimeGpu::SetFullFiringExportEnabled(bool enabled) {
    full_firing_export_enabled_ = enabled;
}

bool DenseSubnetworkRuntimeGpu::full_firing_export_enabled() const {
    return full_firing_export_enabled_;
}

DenseSubnetworkDebugSnapshot DenseSubnetworkRuntimeGpu::BuildDebugSnapshot(std::string* reason) const {
    if (!this->RefreshHostDebugViews(reason)) {
        return DenseSubnetworkDebugSnapshot{};
    }
    DenseSubnetworkDebugSnapshot snapshot;
    snapshot.name = name_;
    snapshot.time_step = current_time_step_;
    snapshot.config = config_;
    snapshot.profiling = propagation_runtime_.profiling();
    snapshot.cuda_build_enabled = propagation_runtime_.IsCudaEnabledBuild();
    snapshot.gpu_backend_ready = propagation_runtime_.IsDeviceReady();
    snapshot.carlsim_like_gpu_active = propagation_runtime_.IsCarlsimLikeGpuPathActive();
    snapshot.visible_output_firing_ids = current_step_output_firing_ids_;
    snapshot.full_firing_ids = current_step_firing_ids_;
    snapshot.pending_channels = propagation_runtime_.pending().pending_channels;
    snapshot.synaptic_weights = debug_synaptic_weights_;
    snapshot.synaptic_learning_pending_dwt = propagation_runtime_.learning().syn_pending_dwt;
    snapshot.emitted_output_spikes = this->ExpandCurrentStepOutputSpikes();
    if (!unified_neuron_runtime_.SyncHostStateFromDevice(reason)) {
        return snapshot;
    }
    unified_neuron_runtime_.BuildDebugSnapshots(&snapshot.model_debug_states, nullptr);
    snapshot.membrane_v = FindFirstFloatDebugField(snapshot.model_debug_states, "v", "rate_hz");
    snapshot.gexc = FindFirstFloatDebugField(snapshot.model_debug_states, "gexc", "decay_g");
    snapshot.ginh = FindFirstFloatDebugField(snapshot.model_debug_states, "ginh");
    snapshot.fired = FindFirstByteDebugField(snapshot.model_debug_states, "fired");
    propagation_runtime_.ExportFiringHistoryRing(&snapshot.firing_history_ring, nullptr);
    return snapshot;
}

bool DenseSubnetworkRuntimeGpu::ExportMonitorState(bool record_state,
                                                   bool record_spikes,
                                                   bool record_pending_channels,
                                                   bool record_weights,
                                                   DenseSubnetworkMonitorState* out,
                                                   std::string* reason) const {
    if (out == nullptr) {
        if (reason != nullptr) {
            *reason = "dense monitor output must not be null";
        }
        return false;
    }
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "DenseSubnetworkRuntimeGpu is not initialized";
        }
        return false;
    }

    // Keep monitor capture narrow: each device-to-host transfer is guarded by
    // the caller's DebugMonitorConfig instead of building the full debug view.
    DenseSubnetworkMonitorState result;
    result.time_step = current_time_step_;
    if (record_state) {
        if (!unified_neuron_runtime_.BuildDebugSnapshots(&result.model_debug_states, reason)) {
            return false;
        }
    }
    if (record_spikes) {
        result.current_firing_ids = current_step_firing_ids_;
    }
    if (record_pending_channels) {
        if (!propagation_runtime_.DownloadPendingChannels(&result.pending_channels,
                                                          &result.pending_channel_count,
                                                          &result.pending_channel_stride,
                                                          reason)) {
            return false;
        }
    }
    if (record_weights) {
        if (!propagation_runtime_.DownloadSynapticWeights(&result.synaptic_weights, reason)) {
            return false;
        }
        debug_synaptic_weights_ = result.synaptic_weights;
    }
    *out = result;
    return true;
}

bool DenseSubnetworkRuntimeGpu::DownloadSynapticWeights(std::vector<float>* weights,
                                                        std::string* reason) const {
    if (!propagation_runtime_.DownloadSynapticWeights(weights, reason)) {
        return false;
    }
    debug_synaptic_weights_ = *weights;
    return true;
}

bool DenseSubnetworkRuntimeGpu::UploadSynapticWeights(const std::vector<float>& weights,
                                                      std::string* reason) {
    if (!propagation_runtime_.UploadSynapticWeights(weights, reason)) {
        return false;
    }
    debug_synaptic_weights_ = weights;
    return true;
}

bool DenseSubnetworkRuntimeGpu::GetSynapseWeight(int synapse_index,
                                                 float* weight,
                                                 std::string* reason) const {
    return propagation_runtime_.GetSynapseWeight(synapse_index, weight, reason);
}

bool DenseSubnetworkRuntimeGpu::SetSynapseWeight(int synapse_index,
                                                 float weight,
                                                 std::string* reason) {
    if (!propagation_runtime_.SetSynapseWeight(synapse_index, weight, reason)) {
        return false;
    }
    if (synapse_index >= 0 &&
        synapse_index < static_cast<int>(debug_synaptic_weights_.size())) {
        debug_synaptic_weights_[static_cast<std::size_t>(synapse_index)] = weight;
    }
    return true;
}

bool DenseSubnetworkRuntimeGpu::UsesCarlsimLikeGpuBackend() const {
    return propagation_runtime_.IsCarlsimLikeGpuPathActive();
}

bool DenseSubnetworkRuntimeGpu::IsGpuBackendReady() const {
    return propagation_runtime_.IsDeviceReady();
}

bool DenseSubnetworkRuntimeGpu::RefreshHostDebugViews(std::string* reason) const {
    if (!unified_neuron_runtime_.SyncHostStateFromDevice(reason)) {
        return false;
    }
    std::vector<FiringStepView> history_ring;
    if (!propagation_runtime_.ExportFiringHistoryRing(&history_ring, reason)) {
        return false;
    }

    std::vector<DenseFiringHistoryEntry> dense_history(history_ring.size());
    for (std::size_t index = 0; index < history_ring.size(); ++index) {
        dense_history[index].time_step = history_ring[index].time_step;
        dense_history[index].firing_ids = history_ring[index].firing_ids;
    }
    firing_table_.LoadHistorySnapshot(dense_history, current_time_step_);
    return true;
}

void DenseSubnetworkRuntimeGpu::CollectOutputSpikesForCurrentStep() {
    emitted_output_spikes_ = DenseSubnetworkOutputExtractor::Extract(
        output_routes_by_source_neuron_, current_step_output_firing_ids_, current_time_step_);
}

void DenseSubnetworkRuntimeGpu::BuildOutputRoutingBySourceNeuron(
    const sim_support::DenseSubnetworkBuildSpec& spec) {
    output_routes_by_source_neuron_ =
        DenseSubnetworkOutputExtractor::BuildOutputRoutesBySourceNeuron(spec, neuron_count_);
}

}  // namespace npgr
