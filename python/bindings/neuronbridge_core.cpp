#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <boost/any.hpp>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "debug_monitor/DebugMonitorTypes.h"
#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"
#include "simulation_dense/DenseSubnetworkModel.h"
#include "source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"
#include "source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuronModel.h"
#include "source_file_realtime_v1_async/Neuron/inc/Neuron.h"
#include "source_file_realtime_v1_async/OuterDynamic/inc/PlanarArm2DOFPinocchio.h"
#include "source_file_realtime_v1_async/OuterDynamic/inc/StrictMatlabPlanarArm2DOFOuterDynamic.h"
#include "source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeCounter.h"
#include "source_file_realtime_v1_async/Simulation/inc/InputConvDescription.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "source_file_realtime_v1_async/communication/inc/DriverType.h"
#include "source_file_realtime_v1_async/mainprogram/inc/BenchProfiling.h"
#include "neuron_model/NeuronModelCatalog.h"

namespace py = pybind11;

namespace {

py::dict BenchProfileSnapshotToDict(const bench_profile::Snapshot& snapshot) {
    py::dict data;
    data["event_remove_ns"] = snapshot.event_remove_ns;
    data["event_buffer_insert_ns"] = snapshot.event_buffer_insert_ns;
    data["event_buffer_flush_ns"] = snapshot.event_buffer_flush_ns;
    data["wheel_insert_ns"] = snapshot.wheel_insert_ns;
    data["wheel_remove_ns"] = snapshot.wheel_remove_ns;
    data["wheel_syn_insert_ns"] = snapshot.wheel_syn_insert_ns;
    data["wheel_syn_remove_ns"] = snapshot.wheel_syn_remove_ns;
    data["wheel_first_event_ns"] = snapshot.wheel_first_event_ns;
    data["wheel_first_syn_event_ns"] = snapshot.wheel_first_syn_event_ns;
    data["wheel_advance_queue_ns"] = snapshot.wheel_advance_queue_ns;
    data["wheel_advance_syn_ns"] = snapshot.wheel_advance_syn_ns;
    data["wheel_migrate_queue_ns"] = snapshot.wheel_migrate_queue_ns;
    data["wheel_migrate_syn_ns"] = snapshot.wheel_migrate_syn_ns;
    data["wheel_pop_ready_ns"] = snapshot.wheel_pop_ready_ns;
    data["sync_ns"] = snapshot.sync_ns;
    data["run_step_end_event_insert_ns"] = snapshot.run_step_end_event_insert_ns;
    data["run_step_parallel_setup_ns"] = snapshot.run_step_parallel_setup_ns;
    data["run_step_set_gpu_thread_ns"] = snapshot.run_step_set_gpu_thread_ns;
    data["run_step_remove_event_ns"] = snapshot.run_step_remove_event_ns;
    data["run_step_process_event_ns"] = snapshot.run_step_process_event_ns;
    data["run_step_delete_event_ns"] = snapshot.run_step_delete_event_ns;
    data["run_step_monitor_capture_ns"] = snapshot.run_step_monitor_capture_ns;
    data["run_step_other_ns"] = snapshot.run_step_other_ns;
    data["time_event_total_ns"] = snapshot.time_event_total_ns;
    data["time_event_update_state_ns"] = snapshot.time_event_update_state_ns;
    data["time_event_internal_spike_ns"] = snapshot.time_event_internal_spike_ns;
    data["time_event_reschedule_ns"] = snapshot.time_event_reschedule_ns;
    data["internal_spike_write_spike_ns"] = snapshot.internal_spike_write_spike_ns;
    data["internal_spike_include_ns"] = snapshot.internal_spike_include_ns;
    data["internal_spike_insert_ready_ns"] = snapshot.internal_spike_insert_ready_ns;
    data["internal_spike_rotate_group_ns"] = snapshot.internal_spike_rotate_group_ns;
    data["internal_spike_learning_ns"] = snapshot.internal_spike_learning_ns;
    data["internal_spike_finalize_group_ns"] = snapshot.internal_spike_finalize_group_ns;
    data["gpu_update_total_ns"] = snapshot.gpu_update_total_ns;
    data["gpu_update_memcpy_h2d_ns"] = snapshot.gpu_update_memcpy_h2d_ns;
    data["gpu_update_kernel_ns"] = snapshot.gpu_update_kernel_ns;
    data["gpu_update_kernel_device_ns"] = snapshot.gpu_update_kernel_device_ns;
    data["gpu_update_stream_device_ns"] = snapshot.gpu_update_stream_device_ns;
    data["gpu_update_d2h_ns"] = snapshot.gpu_update_d2h_ns;
    data["gpu_update_sync_ns"] = snapshot.gpu_update_sync_ns;
    data["gpu_update_memset_ns"] = snapshot.gpu_update_memset_ns;
    data["gpu_update_event_record_ns"] = snapshot.gpu_update_event_record_ns;
    data["gpu_internal_spike_scan_ns"] = snapshot.gpu_internal_spike_scan_ns;
    data["remove_count"] = snapshot.remove_count;
    data["run_step_event_count"] = snapshot.run_step_event_count;
    data["time_event_count"] = snapshot.time_event_count;
    data["gpu_update_count"] = snapshot.gpu_update_count;
    data["sync_count"] = snapshot.sync_count;
    return data;
}

struct NativeSimulationConfig {
    int steps = 0;
    float timestep = 0.0f;
    int queues = 1;
    std::string event_queue = "heap";
    int timing_wheel_size = 0;
};

struct NativeNetworkDescription {
    std::list<NeuronLayerDescription> layers;
    std::list<ConnectionDescription> connections;
    std::list<LearningRuleDescription> learning_rules;
    std::list<OuterDynamicDescription> outer_dynamics;
    std::list<OuterDynamicConnectionDescription> outer_dynamic_connections;
    std::list<InputConvDescription> input_convs;

    int layer_count() const {
        return static_cast<int>(layers.size());
    }

    int connection_block_count() const {
        return static_cast<int>(connections.size());
    }

    int learning_rule_count() const {
        return static_cast<int>(learning_rules.size());
    }

    int outer_dynamic_count() const {
        return static_cast<int>(outer_dynamics.size());
    }

    int outer_dynamic_connection_block_count() const {
        return static_cast<int>(outer_dynamic_connections.size());
    }

    int input_conv_count() const {
        return static_cast<int>(input_convs.size());
    }

    int neuron_count() const {
        int total = 0;
        for (const NeuronLayerDescription& layer : layers) {
            total += layer.numberofneuron;
        }
        return total;
    }
};

EventQueueType ParseEventQueueType(const std::string& value) {
    if (value == "heap") {
        return EVENT_QUEUE_HEAP;
    }
    if (value == "timing_wheel") {
        return EVENT_QUEUE_TIMING_WHEEL;
    }
    throw py::value_error("SimulationConfig.event_queue must be 'heap' or 'timing_wheel'");
}

class NativeSimulation {
public:
    NativeSimulation(const NativeNetworkDescription& network, const NativeSimulationConfig& config)
        : simulation_(new Simulation(
              network.layers,
              network.connections,
              network.learning_rules,
              network.outer_dynamics,
              network.outer_dynamic_connections,
              network.input_convs,
              config.steps,
              config.timestep,
              config.queues,
              ParseEventQueueType(config.event_queue),
              config.timing_wheel_size)),
          initialized_(false) {}

    void init() {
        EnsureSimulation();
        if (!initialized_) {
            simulation_->InitSimulation();
            initialized_ = true;
        }
    }

    void run(int steps) {
        if (steps <= 0) {
            throw py::value_error("Simulation.run steps must be positive");
        }
        this->init();
        py::gil_scoped_release release;
        simulation_->RunSimulationStep(steps);
    }

    void enable_realtime(int slot_steps,
                         double max_advance_seconds,
                         float first_section,
                         float second_section,
                         float third_section) {
        EnsureSimulation();
        simulation_->EnableRealtime(
            slot_steps,
            max_advance_seconds,
            first_section,
            second_section,
            third_section);
    }

    void disable_realtime() {
        EnsureSimulation();
        simulation_->DisableRealtime();
    }

    void run_realtime(int steps) {
        if (steps <= 0) {
            throw py::value_error("Simulation.run_realtime steps must be positive");
        }
        this->init();
        py::gil_scoped_release release;
        simulation_->RunSimulationRealtime(steps);
    }

    void reset_bench_profiling() {
        bench_profile::reset();
    }

    py::dict bench_profiling_snapshot() const {
        return BenchProfileSnapshotToDict(bench_profile::snapshot());
    }

    py::dict realtime_skip_counters() const {
        EnsureSimulation();
        const auto values = simulation_->GetRealtimeSkipCounters();
        py::dict result;
        result["input_spike"] = values[static_cast<std::size_t>(RealtimeSkipKind::InputSpike)];
        result["propagated_spike"] = values[static_cast<std::size_t>(RealtimeSkipKind::PropagatedSpike)];
        result["propagated_spike_group"] = values[static_cast<std::size_t>(RealtimeSkipKind::PropagatedSpikeGroup)];
        result["trigger_relay_spike"] = values[static_cast<std::size_t>(RealtimeSkipKind::TriggerRelaySpike)];
        result["time_driven_spike"] = values[static_cast<std::size_t>(RealtimeSkipKind::TimeDrivenSpike)];
        result["learning_update"] = values[static_cast<std::size_t>(RealtimeSkipKind::LearningUpdate)];
        result["synchronize_activity"] = values[static_cast<std::size_t>(RealtimeSkipKind::SynchronizeActivity)];
        result["input_conv"] = values[static_cast<std::size_t>(RealtimeSkipKind::InputConv)];
        result["outer_update"] = values[static_cast<std::size_t>(RealtimeSkipKind::OuterUpdate)];
        result["unessential_event"] = values[static_cast<std::size_t>(RealtimeSkipKind::UnessentialEvent)];
        return result;
    }

    void reset_realtime_skip_counters() {
        EnsureSimulation();
        simulation_->ResetRealtimeSkipCounters();
    }

    py::dict realtime_restriction_counts() const {
        EnsureSimulation();
        const auto values = simulation_->GetRealtimeRestrictionCounts();
        py::dict result;
        result["simulation_too_fast"] = values[0];
        result["all_events_enabled"] = values[1];
        result["learning_rules_disabled"] = values[2];
        result["spikes_disabled"] = values[3];
        result["all_unessential_events_disabled"] = values[4];
        return result;
    }

    void reset_realtime_restriction_counts() {
        EnsureSimulation();
        simulation_->ResetRealtimeRestrictionCounts();
    }

    void reset(bool preserve_weights) {
        EnsureSimulation();
        simulation_->ResetForNextRound(preserve_weights);
        initialized_ = true;
    }

    void add_external_spikes(const std::vector<int>& times, const std::vector<int>& neuron_ids) {
        if (times.size() != neuron_ids.size()) {
            throw py::value_error("external spike times and neuron_ids must have the same length");
        }
        EnsureSimulation();
        simulation_->AddExternalSpikeActivity(times, neuron_ids);
    }

    void add_external_currents(const std::vector<int>& times,
                               const std::vector<int>& neuron_ids,
                               const std::vector<float>& currents) {
        if (times.size() != neuron_ids.size() || times.size() != currents.size()) {
            throw py::value_error("external current times, neuron_ids, and currents must have the same length");
        }
        EnsureSimulation();
        simulation_->AddExternalCurrentActivity(times, neuron_ids, currents);
    }

    void add_zmq_async_input_output_spike_driver(const std::string& subscribe_address,
                                                unsigned short publish_port,
                                                unsigned short subscribe_port,
                                                const std::string& publish_topic,
                                                const std::string& subscribe_topic,
                                                int communication_interval) {
        EnsureSimulation();
        simulation_->AddZMQAsyncInputOutputSpikeDriver(
            subscribe_address,
            publish_port,
            subscribe_port,
            publish_topic,
            subscribe_topic,
            communication_interval);
    }

    void add_zmq_input_output_spike_driver(const std::string& server_address,
                                           unsigned short server_port,
                                           int communication_interval) {
        EnsureSimulation();
        simulation_->AddZMQInputOutputSpikeDriver(
            CLIENT,
            server_address,
            server_port,
            communication_interval);
    }

    void add_input_conv_frames(const std::string& source_name, const py::list& frames) {
        EnsureSimulation();
        std::vector<InputConvFrame> native_frames;
        native_frames.reserve(static_cast<std::size_t>(py::len(frames)));
        for (const py::handle& item : frames) {
            native_frames.push_back(InputConvFrameFromPython(item));
        }
        std::string reason;
        if (!simulation_->AddExternalInputConvFrames(source_name, native_frames, &reason)) {
            throw std::runtime_error("AddExternalInputConvFrames failed: " + reason);
        }
    }

    void clear_input_conv_frame_queue(const std::string& source_name) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->ClearInputConvFrameQueue(source_name, &reason)) {
            throw std::runtime_error("ClearInputConvFrameQueue failed: " + reason);
        }
    }

    void bind_input_conv_frame_source_index(int inputconv_index,
                                            const std::string& source_name,
                                            int source_camera_index) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->BindInputConvFrameSource(inputconv_index, source_name, source_camera_index, &reason)) {
            throw std::runtime_error("BindInputConvFrameSource failed: " + reason);
        }
    }

    void bind_input_conv_frame_source_name(const std::string& inputconv_name,
                                           const std::string& source_name,
                                           int source_camera_index) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->BindInputConvFrameSource(inputconv_name, source_name, source_camera_index, &reason)) {
            throw std::runtime_error("BindInputConvFrameSource failed: " + reason);
        }
    }

    bool has_input_conv_frame_source_binding(int inputconv_index) const {
        EnsureSimulation();
        return simulation_->HasInputConvFrameSourceBinding(inputconv_index);
    }

    int add_zmq_input_conv_frame_source(const std::string& source_name,
                                        const std::string& address,
                                        unsigned short port,
                                        int max_payload_bytes) {
        EnsureSimulation();
        const int source_index =
            simulation_->AddZMQInputConvFrameSource(source_name, address, port, max_payload_bytes);
        if (source_index < 0) {
            throw std::runtime_error("AddZMQInputConvFrameSource failed");
        }
        return source_index;
    }

    int add_zmq_async_input_conv_frame_source(const std::string& source_name,
                                              const std::string& subscribe_address,
                                              unsigned short subscribe_port,
                                              const std::string& topic,
                                              int max_payload_bytes,
                                              int max_buffered_frames_per_camera) {
        EnsureSimulation();
        const int source_index =
            simulation_->AddZMQAsyncInputConvFrameSource(source_name,
                                                        subscribe_address,
                                                        subscribe_port,
                                                        topic,
                                                        max_payload_bytes,
                                                        max_buffered_frames_per_camera);
        if (source_index < 0) {
            throw std::runtime_error("AddZMQAsyncInputConvFrameSource failed");
        }
        return source_index;
    }

    py::dict input_conv_frame_source_status_name(const std::string& source_name) const {
        EnsureSimulation();
        InputConvFrameSourceStats stats;
        std::string reason;
        if (!simulation_->GetInputConvFrameSourceStats(source_name, &stats, &reason)) {
            throw std::runtime_error("GetInputConvFrameSourceStats failed: " + reason);
        }
        return InputConvFrameSourceStatsToDict(stats);
    }

    py::dict input_conv_frame_source_status_index(int source_index) const {
        EnsureSimulation();
        InputConvFrameSourceStats stats;
        std::string reason;
        if (!simulation_->GetInputConvFrameSourceStats(source_index, &stats, &reason)) {
            throw std::runtime_error("GetInputConvFrameSourceStats failed: " + reason);
        }
        return InputConvFrameSourceStatsToDict(stats);
    }

    py::dict neuron_state(int original_neuron_id) const {
        EnsureSimulation();
        return MainNeuronStateToDict(original_neuron_id);
    }

    py::list neuron_states(const std::vector<int>& original_neuron_ids) const {
        EnsureSimulation();
        py::list states;
        for (int original_neuron_id : original_neuron_ids) {
            states.append(MainNeuronStateToDict(original_neuron_id));
        }
        return states;
    }

    py::list output_spikes() {
        EnsureSimulation();
        py::list spikes;
        if (simulation_->output_spike_driver == nullptr) {
            return spikes;
        }

        int* times = nullptr;
        int* cells = nullptr;
        const int count = simulation_->output_spike_driver->GetBufferedSpikes(times, cells);
        for (int index = 0; index < count; ++index) {
            py::dict spike;
            spike["time"] = times[index];
            spike["neuron_id"] = cells[index];
            spikes.append(spike);
        }
        delete[] times;
        delete[] cells;
        return spikes;
    }

    void enable_debug_monitor(const npgr::DebugMonitorConfig& config) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->EnableDebugMonitor(config, &reason)) {
            throw std::runtime_error("EnableDebugMonitor failed: " + reason);
        }
    }

    void disable_debug_monitor() {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->DisableDebugMonitor(&reason)) {
            throw std::runtime_error("DisableDebugMonitor failed: " + reason);
        }
    }

    void flush_debug_monitor() {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->FlushDebugMonitor(&reason)) {
            throw std::runtime_error("FlushDebugMonitor failed: " + reason);
        }
    }

    void publish_output() {
        EnsureSimulation();
        simulation_->PublishOutput(nullptr);
    }

    float get_connection_weight(int original_connection_index) const {
        EnsureSimulation();
        float weight = 0.0f;
        std::string reason;
        if (!simulation_->GetConnectionWeight(original_connection_index, &weight, &reason)) {
            throw std::runtime_error("GetConnectionWeight failed: " + reason);
        }
        return weight;
    }

    void set_connection_weight(int original_connection_index, float weight) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->SetConnectionWeight(original_connection_index, weight, &reason)) {
            throw std::runtime_error("SetConnectionWeight failed: " + reason);
        }
    }

    void save_weights(const std::string& path) {
        EnsureSimulation();
        simulation_->SaveWeightToFile(path.c_str());
    }

    void load_weights(const std::string& path) {
        EnsureSimulation();
        simulation_->LoadWeight(path.c_str());
    }

    int dense_subnetwork_count() const {
        EnsureSimulation();
        return simulation_->GetDenseSubnetworkCount();
    }

    std::string dense_subnetwork_name(int index) const {
        EnsureSimulation();
        std::string name;
        if (!simulation_->GetDenseSubnetworkName(index, name)) {
            throw py::index_error("dense subnetwork index is out of range");
        }
        return name;
    }

    std::vector<float> dense_subnetwork_weights(int index) const {
        EnsureSimulation();
        std::vector<float> weights;
        if (!simulation_->GetDenseSubnetworkWeights(index, weights)) {
            throw py::index_error("dense subnetwork index is out of range or weights are unavailable");
        }
        return weights;
    }

    int find_dense_subnetwork(const std::string& name) const {
        EnsureSimulation();
        return simulation_->FindDenseSubnetworkByName(name);
    }

    py::dict dense_subnetwork_snapshot(int index) const {
        EnsureSimulation();
        npgr::DenseSubnetworkModel* model = DenseSubnetworkModelAt(index);
        npgr::DenseSubnetworkDebugSnapshot snapshot;
        if (!simulation_->GetDenseSubnetworkDebugSnapshot(index, snapshot)) {
            throw py::index_error("dense subnetwork index is out of range or snapshot is unavailable");
        }
        py::dict data = DenseSubnetworkSnapshotToDict(snapshot);
        std::vector<int> local_to_original;
        local_to_original.reserve(snapshot.membrane_v.size());
        for (int local_id = 0; local_id < static_cast<int>(snapshot.membrane_v.size()); ++local_id) {
            local_to_original.push_back(model->OriginalGlobalNeuronIdForLocal(local_id));
        }
        data["local_to_original_neuron_ids"] = local_to_original;
        return data;
    }

    py::dict dense_subnetwork_snapshot_by_name(const std::string& name) const {
        const int index = find_dense_subnetwork(name);
        if (index < 0) {
            throw py::key_error("dense subnetwork was not found: " + name);
        }
        return dense_subnetwork_snapshot(index);
    }

    void reset_dense_subnetwork(int index) {
        EnsureSimulation();
        if (!simulation_->ResetDenseSubnetwork(index)) {
            throw py::index_error("dense subnetwork index is out of range or reset failed");
        }
    }

    void set_dense_subnetwork_full_firing_export_enabled(int index, bool enabled) {
        EnsureSimulation();
        DenseSubnetworkModelAt(index)->SetFullFiringExportEnabled(enabled);
    }

    int input_conv_count() const {
        EnsureSimulation();
        return static_cast<int>(simulation_->InputConvModelList.size());
    }

    int input_conv_output_count(int index) const {
        EnsureSimulation();
        InputConvModel* model = InputConvModelAt(index);
        return model->GetOutputCount();
    }

    std::vector<float> input_conv_output(int index) const {
        EnsureSimulation();
        InputConvModel* model = InputConvModelAt(index);
        std::vector<float> output;
        std::string reason;
        if (!model->ExportMonitorOutput(&output, &reason)) {
            throw std::runtime_error("InputConv output export failed: " + reason);
        }
        return output;
    }

    std::vector<float> input_conv_input(int index) const {
        EnsureSimulation();
        InputConvModel* model = InputConvModelAt(index);
        std::vector<float> input;
        std::string reason;
        if (!model->ExportMonitorInput(&input, &reason)) {
            throw std::runtime_error("InputConv input export failed: " + reason);
        }
        return input;
    }

    void enable_input_conv_monitor_index(int index) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->EnableInputConvMonitor(index, &reason)) {
            throw std::runtime_error("EnableInputConvMonitor failed: " + reason);
        }
    }

    void enable_input_conv_monitor_name(const std::string& name) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->EnableInputConvMonitor(name, &reason)) {
            throw std::runtime_error("EnableInputConvMonitor failed: " + reason);
        }
    }

    void disable_input_conv_monitor_index(int index) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->DisableInputConvMonitor(index, &reason)) {
            throw std::runtime_error("DisableInputConvMonitor failed: " + reason);
        }
    }

    void disable_input_conv_monitor_name(const std::string& name) {
        EnsureSimulation();
        std::string reason;
        if (!simulation_->DisableInputConvMonitor(name, &reason)) {
            throw std::runtime_error("DisableInputConvMonitor failed: " + reason);
        }
    }

    py::dict outer_dynamic_spike_counter_snapshot(const std::string& name) const {
        const OuterDynamicSpikeCounter* counter = FindSpikeCounter(name);
        py::dict data;
        data["slot_count"] = counter->GetSlotCount();
        data["spike_counts"] = counter->GetSpikeCounts();
        data["weighted_sums"] = counter->GetWeightedSums();
        data["last_spike_times"] = counter->GetLastSpikeTimes();
        py::list by_type_counts;
        py::list by_type_weights;
        for (int type = 0; type < 64; ++type) {
            const std::vector<int> counts = counter->GetSpikeCountsByType(type);
            const std::vector<float> weights = counter->GetWeightedSumsByType(type);
            if (counts.empty() && weights.empty()) {
                break;
            }
            by_type_counts.append(counts);
            by_type_weights.append(weights);
        }
        data["spike_counts_by_type"] = by_type_counts;
        data["weighted_sums_by_type"] = by_type_weights;
        return data;
    }

    void clear_outer_dynamic_spike_counter(const std::string& name) {
        OuterDynamicSpikeCounter* counter = FindMutableSpikeCounter(name);
        counter->ClearAllSlots();
    }

    void clear_outer_dynamic_spike_counter_slot(const std::string& name, int slot_id) {
        OuterDynamicSpikeCounter* counter = FindMutableSpikeCounter(name);
        if (!counter->ClearSlot(slot_id)) {
            throw py::index_error("OuterDynamicSpikeCounter slot is out of range");
        }
    }

    py::dict outer_dynamic_state() const {
        EnsureSimulation();
        OuterDynamicJointState state;
        int time_step = 0;
        if (!simulation_->GetLatestOuterDynamicState(state, time_step)) {
            throw std::runtime_error("No outer dynamic state is available");
        }
        py::dict data;
        data["time_step"] = time_step;
        data["q"] = std::vector<double>(state.q.begin(), state.q.end());
        data["qv"] = std::vector<double>(state.qv.begin(), state.qv.end());
        data["qdd"] = std::vector<double>(state.qdd.begin(), state.qdd.end());
        data["q_des"] = std::vector<double>(state.q_des.begin(), state.q_des.end());
        data["qv_des"] = std::vector<double>(state.qv_des.begin(), state.qv_des.end());
        data["tau_total"] = std::vector<double>(state.tau_total.begin(), state.tau_total.end());
        return data;
    }

    py::list outer_dynamic_states() const {
        EnsureSimulation();
        std::vector<OuterDynamicStateSnapshot> states;
        if (!simulation_->GetLatestOuterDynamicStates(states)) {
            throw std::runtime_error("No outer dynamic states are available");
        }
        py::list result;
        for (std::size_t index = 0; index < states.size(); ++index) {
            const OuterDynamicStateSnapshot& snapshot = states[index];
            py::dict data;
            data["component_index"] = snapshot.component_index;
            data["component_name"] = snapshot.component_name;
            data["time_step"] = snapshot.time_step;
            data["q"] = std::vector<double>(snapshot.state.q.begin(), snapshot.state.q.end());
            data["qv"] = std::vector<double>(snapshot.state.qv.begin(), snapshot.state.qv.end());
            data["qdd"] = std::vector<double>(snapshot.state.qdd.begin(), snapshot.state.qdd.end());
            data["q_des"] = std::vector<double>(snapshot.state.q_des.begin(), snapshot.state.q_des.end());
            data["qv_des"] = std::vector<double>(snapshot.state.qv_des.begin(), snapshot.state.qv_des.end());
            data["tau_total"] = std::vector<double>(snapshot.state.tau_total.begin(), snapshot.state.tau_total.end());
            result.append(data);
        }
        return result;
    }

    void reset_outer_dynamic_state(const std::string& name,
                                   const std::vector<double>& q,
                                   const std::vector<double>& qd) {
        EnsureSimulation();
        const std::array<double, 2> q_array = ToPairArray(q, "q");
        const std::array<double, 2> qd_array = ToPairArray(qd, "qd");
        OuterDynamicModel* model = FindMutableOuterDynamic(name);
        if (auto* arm = dynamic_cast<StrictMatlabPlanarArm2DOFOuterDynamic*>(model)) {
            arm->ResetState(q_array, qd_array);
            return;
        }
        if (auto* arm = dynamic_cast<PlanarArm2DOFPinocchio*>(model)) {
            arm->ResetState(q_array, qd_array);
            return;
        }
        throw py::type_error("OuterDynamic model does not support reset_state: " + name);
    }

    void set_outer_dynamic_desired_state(const std::string& name,
                                         const std::vector<double>& q_des,
                                         const std::vector<double>& qd_des) {
        EnsureSimulation();
        const std::array<double, 2> q_des_array = ToPairArray(q_des, "q_des");
        const std::array<double, 2> qd_des_array = ToPairArray(qd_des, "qd_des");
        OuterDynamicModel* model = FindMutableOuterDynamic(name);
        if (auto* arm = dynamic_cast<StrictMatlabPlanarArm2DOFOuterDynamic*>(model)) {
            arm->SetDesiredState(q_des_array, qd_des_array);
            return;
        }
        if (auto* arm = dynamic_cast<PlanarArm2DOFPinocchio*>(model)) {
            arm->SetDesiredState(q_des_array, qd_des_array);
            return;
        }
        throw py::type_error("OuterDynamic model does not support set_desired_state: " + name);
    }

    bool initialized() const {
        return initialized_;
    }

private:
    void EnsureSimulation() const {
        if (!simulation_) {
            throw std::runtime_error("native Simulation is not constructed");
        }
    }

    OuterDynamicSpikeCounter* FindMutableSpikeCounter(const std::string& name) {
        EnsureSimulation();
        OuterDynamicModel* model = simulation_->GetOuterDynamicByName(name);
        OuterDynamicSpikeCounter* counter = dynamic_cast<OuterDynamicSpikeCounter*>(model);
        if (counter == nullptr) {
            throw py::key_error("OuterDynamicSpikeCounter was not found: " + name);
        }
        return counter;
    }

    OuterDynamicModel* FindMutableOuterDynamic(const std::string& name) {
        OuterDynamicModel* model = simulation_->GetOuterDynamicByName(name);
        if (model == nullptr) {
            throw py::key_error("OuterDynamic model was not found: " + name);
        }
        return model;
    }

    static std::array<double, 2> ToPairArray(const std::vector<double>& values, const std::string& name) {
        if (values.size() != 2) {
            throw py::value_error(name + " must contain exactly 2 values");
        }
        return {values[0], values[1]};
    }

    static InputConvPixelFormat InputConvPixelFormatFromString(const std::string& value) {
        if (value == "uint8_gray" || value == "UInt8Gray") {
            return InputConvPixelFormat::UInt8Gray;
        }
        if (value == "uint8_rgb" || value == "UInt8RGB") {
            return InputConvPixelFormat::UInt8RGB;
        }
        if (value == "uint8_bgr" || value == "UInt8BGR") {
            return InputConvPixelFormat::UInt8BGR;
        }
        if (value == "float32_gray" || value == "Float32Gray") {
            return InputConvPixelFormat::Float32Gray;
        }
        if (value == "float32_hwc" || value == "Float32HWC") {
            return InputConvPixelFormat::Float32HWC;
        }
        if (value == "float32_chw" || value == "Float32CHW") {
            return InputConvPixelFormat::Float32CHW;
        }
        throw py::value_error("unsupported InputConv pixel_format: " + value);
    }

    static InputConvFrame InputConvFrameFromPython(const py::handle& value) {
        if (!py::isinstance<py::dict>(value)) {
            throw py::type_error("InputConvFrame must be provided as a dict");
        }
        py::dict item = py::reinterpret_borrow<py::dict>(value);
        InputConvFrame frame;
        frame.time_step = item["time_step"].cast<int>();
        frame.source_camera_index = item["source_camera_index"].cast<int>();
        frame.width = item["width"].cast<int>();
        frame.height = item["height"].cast<int>();
        frame.channels = item["channels"].cast<int>();
        frame.pixel_format = InputConvPixelFormatFromString(item["pixel_format"].cast<std::string>());
        const std::string payload = item["bytes"].cast<std::string>();
        frame.bytes.assign(payload.begin(), payload.end());
        if (frame.width <= 0 || frame.height <= 0 || frame.channels <= 0) {
            throw py::value_error("InputConvFrame width, height, and channels must be positive");
        }
        return frame;
    }

    static py::dict InputConvFrameSourceStatsToDict(const InputConvFrameSourceStats& stats) {
        py::dict data;
        data["requested_frames"] = stats.requested_frames;
        data["received_frames"] = stats.received_frames;
        data["consumed_frames"] = stats.consumed_frames;
        data["failed_requests"] = stats.failed_requests;
        data["dropped_frames"] = stats.dropped_frames;
        data["queued_frames"] = stats.queued_frames;
        data["last_received_time_step"] = stats.last_received_time_step;
        data["last_consumed_time_step"] = stats.last_consumed_time_step;
        data["last_source_camera_index"] = stats.last_source_camera_index;
        data["running"] = stats.running;
        data["last_error"] = stats.last_error;
        return data;
    }

    const OuterDynamicSpikeCounter* FindSpikeCounter(const std::string& name) const {
        EnsureSimulation();
        const OuterDynamicModel* model = simulation_->GetOuterDynamicByName(name);
        const OuterDynamicSpikeCounter* counter = dynamic_cast<const OuterDynamicSpikeCounter*>(model);
        if (counter == nullptr) {
            throw py::key_error("OuterDynamicSpikeCounter was not found: " + name);
        }
        return counter;
    }

    InputConvModel* InputConvModelAt(int index) const {
        if (index < 0 || index >= static_cast<int>(simulation_->InputConvModelList.size())) {
            throw py::index_error("InputConv index is out of range");
        }
        InputConvModel* model = simulation_->InputConvModelList[static_cast<std::size_t>(index)];
        if (model == nullptr) {
            throw std::runtime_error("InputConv model is null");
        }
        return model;
    }

    py::dict MainNeuronStateToDict(int original_neuron_id) const {
        if (simulation_->network == nullptr) {
            throw std::runtime_error("native Simulation network is not constructed");
        }
        const int main_neuron_id = simulation_->TranslateOriginalNeuronIdToMain(original_neuron_id);
        if (main_neuron_id < 0 || main_neuron_id >= simulation_->network->neuronsNum) {
            throw py::key_error("original neuron id is not present in the main network");
        }
        const Neuron& neuron = simulation_->network->neurons[main_neuron_id];
        const Neuron_State_Vector* state = neuron.neuron_state_vector;
        if (state == nullptr) {
            throw std::runtime_error("native neuron state vector is null");
        }
        const int local = neuron.index_in_NeuronModel;
        const int state_count = state->NumberofStateVariable;
        py::list values;
        for (int index = 0; index < state_count; ++index) {
            values.append(state->Vector_of_StateVariable[local * state_count + index]);
        }

        py::dict data;
        data["original_neuron_id"] = original_neuron_id;
        data["main_neuron_id"] = main_neuron_id;
        data["local_neuron_id"] = local;
        data["queue_index"] = neuron.Queue_index;
        data["is_output"] = neuron.IsOutput;
        data["is_monitored"] = neuron.IsMonitor;
        data["state_variables"] = values;
        data["last_update"] = state->LastUpdate != nullptr ? state->LastUpdate[local] : -1;
        data["last_spike"] = state->LastSpike != nullptr ? state->LastSpike[local] : -1;
        data["predict_spike"] = state->PredictSpike != nullptr ? state->PredictSpike[local] : -1;
        data["model_name"] =
            neuron.neuron_model != nullptr ? neuron.neuron_model->getModelName() : std::string();
        return data;
    }

    npgr::DenseSubnetworkModel* DenseSubnetworkModelAt(int index) const {
        if (index < 0 || index >= static_cast<int>(simulation_->dense_subnetworks.size())) {
            throw py::index_error("dense subnetwork index is out of range");
        }
        npgr::DenseSubnetworkModel* model = simulation_->dense_subnetworks[static_cast<std::size_t>(index)];
        if (model == nullptr) {
            throw std::runtime_error("dense subnetwork model is null");
        }
        return model;
    }

    static py::dict DenseOutputSpikeToDict(const npgr::DenseOutputSpike& spike) {
        py::dict data;
        data["output_index"] = spike.output_index;
        data["source_neuron"] = spike.source_neuron;
        data["external_target_id"] = spike.external_target_id;
        data["time_step"] = spike.time_step;
        data["delay"] = spike.delay;
        data["weight"] = spike.weight;
        data["synapse_type"] = spike.synapse_type;
        return data;
    }

    static py::dict DenseModelDebugSnapshotToDict(const npgr::DenseNeuronDebugSnapshot& snapshot) {
        py::dict data;
        data["factory_model_id"] = snapshot.factory_model_id;
        data["model_id"] = snapshot.model_id;
        data["legacy_model_name"] = snapshot.legacy_model_name;
        data["float_state_vectors"] = snapshot.float_state_vectors;
        data["byte_state_vectors"] = snapshot.byte_state_vectors;
        return data;
    }

    static py::dict DenseSubnetworkSnapshotToDict(const npgr::DenseSubnetworkDebugSnapshot& snapshot) {
        py::dict data;
        data["name"] = snapshot.name;
        data["time_step"] = snapshot.time_step;
        data["cuda_build_enabled"] = snapshot.cuda_build_enabled;
        data["gpu_backend_ready"] = snapshot.gpu_backend_ready;
        data["carlsim_like_gpu_active"] = snapshot.carlsim_like_gpu_active;
        data["visible_output_firing_ids"] = snapshot.visible_output_firing_ids;
        data["full_firing_ids"] = snapshot.full_firing_ids;
        data["membrane_v"] = snapshot.membrane_v;
        data["gexc"] = snapshot.gexc;
        data["ginh"] = snapshot.ginh;
        data["fired"] = snapshot.fired;
        data["pending_channels"] = snapshot.pending_channels;
        data["synaptic_weights"] = snapshot.synaptic_weights;
        data["synaptic_learning_pending_dwt"] = snapshot.synaptic_learning_pending_dwt;
        py::list model_debug_states;
        for (const npgr::DenseNeuronDebugSnapshot& model_snapshot : snapshot.model_debug_states) {
            model_debug_states.append(DenseModelDebugSnapshotToDict(model_snapshot));
        }
        data["model_debug_states"] = model_debug_states;
        py::list emitted_output_spikes;
        for (const npgr::DenseOutputSpike& spike : snapshot.emitted_output_spikes) {
            emitted_output_spikes.append(DenseOutputSpikeToDict(spike));
        }
        data["emitted_output_spikes"] = emitted_output_spikes;
        data["firing_history_ring_size"] = static_cast<int>(snapshot.firing_history_ring.size());
        return data;
    }

    std::unique_ptr<Simulation> simulation_;
    bool initialized_;
};

bool HasNativeParameterKind(const py::handle& value) {
    return py::hasattr(value, "__neuronbridge_param_kind__") && py::hasattr(value, "value");
}

boost::any NativeParameterToAny(const py::handle& value) {
    const std::string kind = py::getattr(value, "__neuronbridge_param_kind__").cast<std::string>();
    const py::object payload = py::getattr(value, "value");
    if (kind == "int32") {
        return boost::any(payload.cast<int>());
    }
    if (kind == "float32") {
        return boost::any(payload.cast<float>());
    }
    if (kind == "float64") {
        return boost::any(payload.cast<double>());
    }
    if (kind == "int32_list") {
        return boost::any(payload.cast<std::vector<int> >());
    }
    if (kind == "float32_list") {
        return boost::any(payload.cast<std::vector<float> >());
    }
    if (kind == "float64_list") {
        return boost::any(payload.cast<std::vector<double> >());
    }
    if (kind == "float32_array3") {
        const std::vector<float> values = payload.cast<std::vector<float> >();
        if (values.size() != 3) {
            throw py::value_error("float32_array3 requires exactly 3 values");
        }
        return boost::any(std::array<float, 3>{values[0], values[1], values[2]});
    }
    if (kind == "float32_array4") {
        const std::vector<float> values = payload.cast<std::vector<float> >();
        if (values.size() != 4) {
            throw py::value_error("float32_array4 requires exactly 4 values");
        }
        return boost::any(std::array<float, 4>{values[0], values[1], values[2], values[3]});
    }
    throw py::type_error("unsupported neuronbridge native parameter kind: " + kind);
}

boost::any PythonValueToAny(const py::handle& value) {
    if (HasNativeParameterKind(value)) {
        return NativeParameterToAny(value);
    }
    if (py::isinstance<py::bool_>(value)) {
        return boost::any(value.cast<bool>());
    }
    if (py::isinstance<py::int_>(value)) {
        return boost::any(value.cast<int>());
    }
    if (py::isinstance<py::float_>(value)) {
        return boost::any(value.cast<float>());
    }
    if (py::isinstance<py::str>(value)) {
        return boost::any(value.cast<std::string>());
    }
    if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
        py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
        if (sequence.size() == 0) {
            return boost::any(std::vector<int>());
        }
        py::handle first = sequence[0];
        if (py::isinstance<py::bool_>(first)) {
            return boost::any(sequence.cast<std::vector<bool> >());
        }
        if (py::isinstance<py::int_>(first)) {
            return boost::any(sequence.cast<std::vector<int> >());
        }
        if (py::isinstance<py::float_>(first)) {
            return boost::any(sequence.cast<std::vector<float> >());
        }
        if (py::isinstance<py::str>(first)) {
            return boost::any(sequence.cast<std::vector<std::string> >());
        }
    }
    throw py::type_error("parameters only support bool, int, float, str, or homogeneous lists of those types");
}

InputConvOutputTarget ParseInputConvOutputTarget(const std::string& value) {
    if (value == "main_network") {
        return InputConvOutputTarget::MainNetwork;
    }
    if (value == "dense_subnetwork") {
        return InputConvOutputTarget::DenseSubnetwork;
    }
    throw py::value_error("InputConv output_target must be 'main_network' or 'dense_subnetwork'");
}

std::string InputConvOutputTargetName(InputConvOutputTarget target) {
    switch (target) {
        case InputConvOutputTarget::MainNetwork:
            return "main_network";
        case InputConvOutputTarget::DenseSubnetwork:
            return "dense_subnetwork";
        default:
            return "unknown";
    }
}

OuterDynamicFeedbackProductEncoding ProductEncodingFromDict(const py::dict& data) {
    OuterDynamicFeedbackProductEncoding encoding;
    if (data.contains("neuron_indices_by_joint")) {
        encoding.neuron_indices_by_joint =
            data["neuron_indices_by_joint"].cast<std::vector<std::vector<int> > >();
    }
    if (data.contains("bins")) {
        encoding.bins = data["bins"].cast<std::vector<int> >();
    }
    return encoding;
}

OuterDynamicFeedbackSingleEncoding SingleEncodingFromDict(const py::dict& data) {
    OuterDynamicFeedbackSingleEncoding encoding;
    if (data.contains("neuron_indices_by_joint_variable")) {
        encoding.neuron_indices_by_joint_variable =
            data["neuron_indices_by_joint_variable"].cast<std::vector<std::vector<std::vector<int> > > >();
    }
    if (data.contains("bins")) {
        encoding.bins = data["bins"].cast<std::vector<int> >();
    }
    return encoding;
}

py::dict ProductEncodingToDict(const OuterDynamicFeedbackProductEncoding& encoding) {
    py::dict data;
    data["neuron_indices_by_joint"] = encoding.neuron_indices_by_joint;
    data["bins"] = encoding.bins;
    return data;
}

py::dict SingleEncodingToDict(const OuterDynamicFeedbackSingleEncoding& encoding) {
    py::dict data;
    data["neuron_indices_by_joint_variable"] = encoding.neuron_indices_by_joint_variable;
    data["bins"] = encoding.bins;
    return data;
}

std::map<std::string, boost::any> PythonDictToAnyMap(const py::dict& parameters) {
    std::map<std::string, boost::any> result;
    for (const auto& item : parameters) {
        if (!py::isinstance<py::str>(item.first)) {
            throw py::type_error("parameter names must be strings");
        }
        result[item.first.cast<std::string>()] = PythonValueToAny(item.second);
    }
    return result;
}

py::dict AnyMapToPythonDict(const std::map<std::string, boost::any>& parameters);

py::object AnyToPythonValue(const boost::any& value) {
    if (value.type() == typeid(bool)) {
        return py::bool_(boost::any_cast<bool>(value));
    }
    if (value.type() == typeid(int)) {
        return py::int_(boost::any_cast<int>(value));
    }
    if (value.type() == typeid(float)) {
        return py::float_(boost::any_cast<float>(value));
    }
    if (value.type() == typeid(double)) {
        return py::float_(boost::any_cast<double>(value));
    }
    if (value.type() == typeid(std::string)) {
        return py::str(boost::any_cast<std::string>(value));
    }
    if (value.type() == typeid(std::vector<bool>)) {
        return py::cast(boost::any_cast<std::vector<bool> >(value));
    }
    if (value.type() == typeid(std::vector<int>)) {
        return py::cast(boost::any_cast<std::vector<int> >(value));
    }
    if (value.type() == typeid(std::vector<float>)) {
        return py::cast(boost::any_cast<std::vector<float> >(value));
    }
    if (value.type() == typeid(std::vector<double>)) {
        return py::cast(boost::any_cast<std::vector<double> >(value));
    }
    if (value.type() == typeid(std::vector<std::string>)) {
        return py::cast(boost::any_cast<std::vector<std::string> >(value));
    }
    if (value.type() == typeid(std::array<float, 2>)) {
        const std::array<float, 2> values = boost::any_cast<std::array<float, 2> >(value);
        return py::cast(std::vector<float>(values.begin(), values.end()));
    }
    if (value.type() == typeid(std::array<float, 3>)) {
        const std::array<float, 3> values = boost::any_cast<std::array<float, 3> >(value);
        return py::cast(std::vector<float>(values.begin(), values.end()));
    }
    if (value.type() == typeid(std::array<float, 4>)) {
        const std::array<float, 4> values = boost::any_cast<std::array<float, 4> >(value);
        return py::cast(std::vector<float>(values.begin(), values.end()));
    }
    if (value.type() == typeid(std::array<float, 6>)) {
        const std::array<float, 6> values = boost::any_cast<std::array<float, 6> >(value);
        return py::cast(std::vector<float>(values.begin(), values.end()));
    }
    if (value.type() == typeid(ModelDescription)) {
        const ModelDescription description = boost::any_cast<ModelDescription>(value);
        py::dict result;
        result["name"] = description.ModelName;
        result["parameters"] = AnyMapToPythonDict(description.ModelParameter);
        return result;
    }
    return py::str("<unsupported boost::any value>");
}

py::dict AnyMapToPythonDict(const std::map<std::string, boost::any>& parameters) {
    py::dict result;
    for (const auto& item : parameters) {
        result[item.first.c_str()] = AnyToPythonValue(item.second);
    }
    return result;
}

const char* NeuronBackendName(npgr::NeuronBackend backend) {
    switch (backend) {
        case npgr::NeuronBackend::LegacyCpu:
            return "legacy_cpu";
        case npgr::NeuronBackend::LegacyGpu:
            return "legacy_gpu";
        case npgr::NeuronBackend::DenseGpu:
            return "dense_gpu";
    }
    throw std::runtime_error("unknown neuron backend");
}

npgr::NeuronBackend ParseNeuronBackend(const std::string& value) {
    if (value == "legacy_cpu") {
        return npgr::NeuronBackend::LegacyCpu;
    }
    if (value == "legacy_gpu") {
        return npgr::NeuronBackend::LegacyGpu;
    }
    if (value == "dense_gpu") {
        return npgr::NeuronBackend::DenseGpu;
    }
    throw py::value_error(
        "backend must be 'legacy_cpu', 'legacy_gpu', or 'dense_gpu'");
}

const npgr::NeuronModelBackendBinding* FindNeuronBackendBinding(
    const npgr::NeuronModelCatalogEntry& entry,
    npgr::NeuronBackend backend) {
    for (const npgr::NeuronModelBackendBinding& binding : entry.backends) {
        if (binding.backend == backend) {
            return &binding;
        }
    }
    return nullptr;
}

npgr::NeuronBackend SelectNeuronBackend(
    const npgr::NeuronModelCatalogEntry& entry,
    const std::string& requested_backend,
    const std::string& requested_name) {
    if (!requested_backend.empty()) {
        return ParseNeuronBackend(requested_backend);
    }
    if (requested_name.find("_GPU") != std::string::npos) {
        const npgr::NeuronModelBackendBinding* gpu =
            FindNeuronBackendBinding(entry, npgr::NeuronBackend::LegacyGpu);
        if (gpu != nullptr && gpu->supported) {
            return npgr::NeuronBackend::LegacyGpu;
        }
    }
    for (const npgr::NeuronModelBackendBinding& binding : entry.backends) {
        if (binding.supported && binding.preferred) {
            return binding.backend;
        }
    }
    for (const npgr::NeuronModelBackendBinding& binding : entry.backends) {
        if (binding.supported) {
            return binding.backend;
        }
    }
    throw std::runtime_error("neuron model has no supported backend: " + entry.canonical_name);
}

py::list ListNeuronModels(const std::string& backend, bool public_only) {
    const npgr::NeuronModelCatalog& catalog = npgr::NeuronModelCatalog::Instance();
    const bool filter_backend = !backend.empty();
    const npgr::NeuronBackend selected_backend =
        filter_backend ? ParseNeuronBackend(backend) : npgr::NeuronBackend::LegacyCpu;
    py::list result;
    for (const npgr::NeuronModelCatalogEntry& entry : catalog.Entries()) {
        if (public_only && !entry.public_user_model) {
            continue;
        }
        if (filter_backend && !catalog.IsSupported(entry.canonical_name, selected_backend)) {
            continue;
        }
        result.append(entry.canonical_name);
    }
    return result;
}

py::dict DescribeNeuronModel(const std::string& name,
                             const std::string& backend,
                             const py::dict& parameters,
                             int timestep_size,
                             float base_timestep) {
    const npgr::NeuronModelCatalog& catalog = npgr::NeuronModelCatalog::Instance();
    const npgr::NeuronModelCatalogEntry* entry = catalog.Resolve(name);
    if (entry == nullptr) {
        throw py::key_error("unknown neuron model: " + name);
    }
    const npgr::NeuronBackend selected_backend =
        SelectNeuronBackend(*entry, backend, name);
    const npgr::NeuronModelBackendBinding* binding =
        FindNeuronBackendBinding(*entry, selected_backend);
    if (binding == nullptr || !binding->supported) {
        throw py::value_error(
            "neuron model " + entry->canonical_name + " is not supported by " +
            NeuronBackendName(selected_backend));
    }

    py::list supported_backends;
    for (const npgr::NeuronModelBackendBinding& candidate : entry->backends) {
        if (!candidate.supported) {
            continue;
        }
        py::dict backend_info;
        backend_info["backend"] = NeuronBackendName(candidate.backend);
        backend_info["implementation"] = candidate.implementation_name;
        supported_backends.append(std::move(backend_info));
    }

    py::dict result;
    result["name"] = entry->canonical_name;
    result["backend"] = NeuronBackendName(selected_backend);
    result["implementation"] = binding->implementation_name;
    result["supported_backends"] = std::move(supported_backends);
    result["parameters"] = AnyMapToPythonDict(
        NeuronModelFactory::QueryParameters(
            entry->canonical_name,
            selected_backend,
            PythonDictToAnyMap(parameters),
            timestep_size,
            base_timestep));
    return result;
}

void AddLayer(NativeNetworkDescription& network,
              const std::string& model,
              int count,
              int update_timestep,
              bool monitored,
              bool output,
              bool communication_input,
              const py::dict& parameters) {
    if (count <= 0) {
        throw py::value_error("layer count must be positive");
    }
    if (update_timestep <= 0) {
        throw py::value_error("layer update_timestep must be positive");
    }
    NeuronLayerDescription layer;
    layer.ModelName = model;
    layer.numberofneuron = count;
    layer.update_timestep = update_timestep;
    layer.isMonitored = monitored;
    layer.isOutput = output;
    layer.isCommunicationInput = communication_input;
    layer.NeuronParameter = PythonDictToAnyMap(parameters);
    network.layers.push_back(layer);
}

void AddConnection(NativeNetworkDescription& network,
                   const std::vector<int>& source,
                   const std::vector<int>& target,
                   const std::vector<int>& synapse_type,
                   const std::vector<float>& weight,
                   const std::vector<float>& max_weight,
                   const std::vector<int>& delay,
                   const std::vector<int>& synapse_rule,
                   const std::vector<int>& trigger_rule) {
    const size_t count = source.size();
    if (count == 0 || target.size() != count || synapse_type.size() != count ||
        weight.size() != count || max_weight.size() != count || delay.size() != count ||
        synapse_rule.size() != count || trigger_rule.size() != count) {
        throw py::value_error("all connection arrays must be non-empty and have the same length");
    }
    ConnectionDescription connection;
    connection.SourceNeuron = source;
    connection.TargetNeuron = target;
    connection.Type = synapse_type;
    connection.Weight = weight;
    connection.MaxWeight = max_weight;
    connection.Delay = delay;
    connection.SynapseRule = synapse_rule;
    connection.TriggerRule = trigger_rule;
    network.connections.push_back(connection);
}

void AddLearningRule(NativeNetworkDescription& network,
                     const std::string& name,
                     const py::dict& parameters) {
    LearningRuleDescription rule;
    rule.RuleName = name;
    rule.RuleParameter = PythonDictToAnyMap(parameters);
    network.learning_rules.push_back(rule);
}

void AddOuterDynamic(NativeNetworkDescription& network,
                     const std::string& model,
                     const std::string& name,
                     const py::dict& parameters,
                     int update_timestep,
                     int communication_interval,
                     int queue_index,
                     const std::vector<std::vector<int> >& gc_neuron_indices_by_joint,
                     const std::vector<std::vector<int> >& mf_neuron_indices_by_joint,
                     const std::vector<std::vector<int> >& cf_positive_neuron_indices_by_joint,
                     const std::vector<std::vector<int> >& cf_negative_neuron_indices_by_joint,
                     const std::vector<std::vector<int> >& dcn_positive_neuron_indices_by_joint,
                     const std::vector<std::vector<int> >& dcn_negative_neuron_indices_by_joint,
                     const py::dict& state_feedback_product,
                     const py::dict& state_feedback_single,
                     const py::dict& error_feedback_product,
                     const py::dict& error_feedback_single) {
    if (update_timestep <= 0) {
        throw py::value_error("outer dynamic update_timestep must be positive");
    }
    if (communication_interval <= 0) {
        throw py::value_error("outer dynamic communication_interval must be positive");
    }
    if (queue_index < 0) {
        throw py::value_error("outer dynamic queue_index must be non-negative");
    }
    OuterDynamicDescription description;
    description.ModelName = model;
    description.name = name;
    description.ModelParameter = PythonDictToAnyMap(parameters);
    description.update_timestep = update_timestep;
    description.communication_interval = communication_interval;
    description.queue_index = queue_index;
    description.gc_neuron_indices_by_joint = gc_neuron_indices_by_joint;
    description.mf_neuron_indices_by_joint = mf_neuron_indices_by_joint;
    description.cf_positive_neuron_indices_by_joint = cf_positive_neuron_indices_by_joint;
    description.cf_negative_neuron_indices_by_joint = cf_negative_neuron_indices_by_joint;
    description.dcn_positive_neuron_indices_by_joint = dcn_positive_neuron_indices_by_joint;
    description.dcn_negative_neuron_indices_by_joint = dcn_negative_neuron_indices_by_joint;
    description.state_feedback_product = ProductEncodingFromDict(state_feedback_product);
    description.state_feedback_single = SingleEncodingFromDict(state_feedback_single);
    description.error_feedback_product = ProductEncodingFromDict(error_feedback_product);
    description.error_feedback_single = SingleEncodingFromDict(error_feedback_single);
    network.outer_dynamics.push_back(description);
}

void AddOuterDynamicConnection(NativeNetworkDescription& network,
                               const std::vector<int>& source,
                               const std::vector<int>& target_outer_dynamic,
                               const std::vector<int>& target_joint,
                               const std::vector<int>& synapse_type,
                               const std::vector<float>& weight,
                               const std::vector<int>& delay) {
    const size_t count = source.size();
    if (count == 0 || target_outer_dynamic.size() != count || target_joint.size() != count ||
        synapse_type.size() != count || weight.size() != count || delay.size() != count) {
        throw py::value_error("all outer dynamic connection arrays must be non-empty and have the same length");
    }
    OuterDynamicConnectionDescription connection;
    connection.SourceNeuron = source;
    connection.TargetOuterDynamic = target_outer_dynamic;
    connection.TargetJoint = target_joint;
    connection.Type = synapse_type;
    connection.Weight = weight;
    connection.Delay = delay;
    network.outer_dynamic_connections.push_back(connection);
}

void AddInputConv(NativeNetworkDescription& network,
                  const std::string& model,
                  const py::dict& parameters,
                  int update_timestep,
                  int queue_index,
                  const std::string& output_target,
                  const std::string& target_dense_subnetwork_name,
                  const std::vector<int>& output_source_indices,
                  const std::vector<int>& output_target_neuron_ids,
                  int output_pending_channel,
                  float output_scale,
                  const std::vector<float>& output_scales,
                  bool output_overwrite) {
    if (update_timestep <= 0) {
        throw py::value_error("InputConv update_timestep must be positive");
    }
    if (queue_index < 0) {
        throw py::value_error("InputConv queue_index must be non-negative");
    }
    if (output_source_indices.size() != output_target_neuron_ids.size()) {
        throw py::value_error("InputConv output source and target id lists must have the same length");
    }
    if (!output_scales.empty() && output_scales.size() != output_source_indices.size()) {
        throw py::value_error("InputConv output_scales must be empty or match output_source_indices length");
    }
    InputConvDescription description;
    description.ModelName = model;
    description.ModelParameter = PythonDictToAnyMap(parameters);
    description.update_timestep = update_timestep;
    description.queue_index = queue_index;
    description.output_target = ParseInputConvOutputTarget(output_target);
    description.target_dense_subnetwork_name = target_dense_subnetwork_name;
    description.output_source_indices = output_source_indices;
    description.output_target_neuron_ids = output_target_neuron_ids;
    description.output_pending_channel = output_pending_channel;
    description.output_scale = output_scale;
    description.output_scales = output_scales;
    description.output_overwrite = output_overwrite;
    network.input_convs.push_back(description);
}

py::dict NetworkDescriptionToDict(const NativeNetworkDescription& network) {
    py::dict data;
    py::list layers;
    for (const NeuronLayerDescription& layer : network.layers) {
        py::dict item;
        item["model"] = layer.ModelName;
        item["count"] = layer.numberofneuron;
        item["update_timestep"] = layer.update_timestep;
        item["monitored"] = layer.isMonitored;
        item["output"] = layer.isOutput;
        item["communication_input"] = layer.isCommunicationInput;
        item["parameters"] = AnyMapToPythonDict(layer.NeuronParameter);
        layers.append(item);
    }
    py::list connections;
    for (const ConnectionDescription& connection : network.connections) {
        py::dict item;
        item["source"] = connection.SourceNeuron;
        item["target"] = connection.TargetNeuron;
        item["synapse_type"] = connection.Type;
        item["weight"] = connection.Weight;
        item["max_weight"] = connection.MaxWeight;
        item["delay"] = connection.Delay;
        item["synapse_rule"] = connection.SynapseRule;
        item["trigger_rule"] = connection.TriggerRule;
        connections.append(item);
    }
    py::list learning_rules;
    for (const LearningRuleDescription& rule : network.learning_rules) {
        py::dict item;
        item["name"] = rule.RuleName;
        item["parameters"] = AnyMapToPythonDict(rule.RuleParameter);
        learning_rules.append(item);
    }
    py::list outer_dynamics;
    for (const OuterDynamicDescription& outer_dynamic : network.outer_dynamics) {
        py::dict item;
        item["model"] = outer_dynamic.ModelName;
        item["name"] = outer_dynamic.name;
        item["parameters"] = AnyMapToPythonDict(outer_dynamic.ModelParameter);
        item["update_timestep"] = outer_dynamic.update_timestep;
        item["communication_interval"] = outer_dynamic.communication_interval;
        item["queue_index"] = outer_dynamic.queue_index;
        item["gc_neuron_indices_by_joint"] = outer_dynamic.gc_neuron_indices_by_joint;
        item["mf_neuron_indices_by_joint"] = outer_dynamic.mf_neuron_indices_by_joint;
        item["cf_positive_neuron_indices_by_joint"] = outer_dynamic.cf_positive_neuron_indices_by_joint;
        item["cf_negative_neuron_indices_by_joint"] = outer_dynamic.cf_negative_neuron_indices_by_joint;
        item["dcn_positive_neuron_indices_by_joint"] = outer_dynamic.dcn_positive_neuron_indices_by_joint;
        item["dcn_negative_neuron_indices_by_joint"] = outer_dynamic.dcn_negative_neuron_indices_by_joint;
        item["state_feedback_product"] = ProductEncodingToDict(outer_dynamic.state_feedback_product);
        item["state_feedback_single"] = SingleEncodingToDict(outer_dynamic.state_feedback_single);
        item["error_feedback_product"] = ProductEncodingToDict(outer_dynamic.error_feedback_product);
        item["error_feedback_single"] = SingleEncodingToDict(outer_dynamic.error_feedback_single);
        outer_dynamics.append(item);
    }
    py::list outer_dynamic_connections;
    for (const OuterDynamicConnectionDescription& connection : network.outer_dynamic_connections) {
        py::dict item;
        item["source"] = connection.SourceNeuron;
        item["target_outer_dynamic"] = connection.TargetOuterDynamic;
        item["target_joint"] = connection.TargetJoint;
        item["synapse_type"] = connection.Type;
        item["weight"] = connection.Weight;
        item["delay"] = connection.Delay;
        outer_dynamic_connections.append(item);
    }
    py::list input_convs;
    for (const InputConvDescription& input_conv : network.input_convs) {
        py::dict item;
        item["model"] = input_conv.ModelName;
        item["parameters"] = AnyMapToPythonDict(input_conv.ModelParameter);
        item["update_timestep"] = input_conv.update_timestep;
        item["queue_index"] = input_conv.queue_index;
        item["output_target"] = InputConvOutputTargetName(input_conv.output_target);
        item["target_dense_subnetwork_name"] = input_conv.target_dense_subnetwork_name;
        item["output_source_indices"] = input_conv.output_source_indices;
        item["output_target_neuron_ids"] = input_conv.output_target_neuron_ids;
        item["output_pending_channel"] = input_conv.output_pending_channel;
        item["output_scale"] = input_conv.output_scale;
        item["output_scales"] = input_conv.output_scales;
        item["output_overwrite"] = input_conv.output_overwrite;
        input_convs.append(item);
    }
    data["layers"] = layers;
    data["connections"] = connections;
    data["learning_rules"] = learning_rules;
    data["outer_dynamics"] = outer_dynamics;
    data["outer_dynamic_connections"] = outer_dynamic_connections;
    data["input_convs"] = input_convs;
    data["neuron_count"] = network.neuron_count();
    return data;
}

py::dict GetBuildInfo() {
    py::dict info;
    info["package"] = "neuronbridge";
    info["backend"] = "NeuronBridge runtime";
#if NPGR_ENABLE_CUDA
    info["cuda_enabled"] = true;
    info["dense_runtime_enabled"] = true;
#else
    info["cuda_enabled"] = false;
    info["dense_runtime_enabled"] = false;
#endif
    info["cpp_namespace"] = "npgr";
    info["binding"] = "pybind11";
    info["api_stage"] = "python-api";
    return info;
}

py::dict DebugMonitorConfigToDict(const npgr::DebugMonitorConfig& config) {
    py::dict data;
    data["enabled"] = config.enabled;
    data["output_dir"] = config.output_dir;
    data["sample_interval_steps"] = config.sample_interval_steps;
    data["flush_interval_steps"] = config.flush_interval_steps;
    data["record_spikes"] = config.record_spikes;
    data["record_state"] = config.record_state;
    data["record_weights"] = config.record_weights;
    data["record_pending_channels"] = config.record_pending_channels;
    data["record_outer_dynamic_state"] = config.record_outer_dynamic_state;
    data["all_neurons"] = config.all_neurons;
    data["neuron_ids"] = config.neuron_ids;
    data["dense_local_neuron_ids"] = config.dense_local_neuron_ids;
    data["record_inputconv_outputs"] = config.record_inputconv_outputs;
    data["record_inputconv_inputs"] = config.record_inputconv_inputs;
    data["record_inputconv_internal_state"] = config.record_inputconv_internal_state;
    data["monitor_all_inputconv"] = config.monitor_all_inputconv;
    data["monitored_inputconv_indices"] = config.monitored_inputconv_indices;
    data["monitored_inputconv_names"] = config.monitored_inputconv_names;
    return data;
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "pybind11 bridge for the NeuronBridge simulation runtime.";

    module.def("list_neuron_models",
               &ListNeuronModels,
               py::arg("backend") = "",
               py::arg("public_only") = true);
    module.def("describe_neuron_model",
               &DescribeNeuronModel,
               py::arg("name"),
               py::arg("backend") = "",
               py::arg("parameters") = py::dict(),
               py::arg("timestep_size") = 1,
               py::arg("base_timestep") = 1.0f);

    py::enum_<npgr::DebugComponentKind>(module, "DebugComponentKind")
        .value("MAIN_NETWORK", npgr::DebugComponentKind::MainNetwork)
        .value("DENSE_SUBNETWORK", npgr::DebugComponentKind::DenseSubnetwork)
        .value("INPUT_CONV", npgr::DebugComponentKind::InputConv)
        .value("OUTER_DYNAMIC", npgr::DebugComponentKind::OuterDynamic)
        .export_values();

    py::class_<npgr::DebugMonitorConfig>(module, "DebugMonitorConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &npgr::DebugMonitorConfig::enabled)
        .def_readwrite("output_dir", &npgr::DebugMonitorConfig::output_dir)
        .def_readwrite("sample_interval_steps", &npgr::DebugMonitorConfig::sample_interval_steps)
        .def_readwrite("flush_interval_steps", &npgr::DebugMonitorConfig::flush_interval_steps)
        .def_readwrite("record_spikes", &npgr::DebugMonitorConfig::record_spikes)
        .def_readwrite("record_state", &npgr::DebugMonitorConfig::record_state)
        .def_readwrite("record_weights", &npgr::DebugMonitorConfig::record_weights)
        .def_readwrite("record_pending_channels", &npgr::DebugMonitorConfig::record_pending_channels)
        .def_readwrite("record_outer_dynamic_state", &npgr::DebugMonitorConfig::record_outer_dynamic_state)
        .def_readwrite("all_neurons", &npgr::DebugMonitorConfig::all_neurons)
        .def_readwrite("neuron_ids", &npgr::DebugMonitorConfig::neuron_ids)
        .def_readwrite("dense_local_neuron_ids", &npgr::DebugMonitorConfig::dense_local_neuron_ids)
        .def_readwrite("record_inputconv_outputs", &npgr::DebugMonitorConfig::record_inputconv_outputs)
        .def_readwrite("record_inputconv_inputs", &npgr::DebugMonitorConfig::record_inputconv_inputs)
        .def_readwrite("record_inputconv_internal_state", &npgr::DebugMonitorConfig::record_inputconv_internal_state)
        .def_readwrite("monitor_all_inputconv", &npgr::DebugMonitorConfig::monitor_all_inputconv)
        .def_readwrite("monitored_inputconv_indices", &npgr::DebugMonitorConfig::monitored_inputconv_indices)
        .def_readwrite("monitored_inputconv_names", &npgr::DebugMonitorConfig::monitored_inputconv_names)
        .def("to_dict", &DebugMonitorConfigToDict);

    py::class_<NativeSimulationConfig>(module, "SimulationConfig")
        .def(py::init<int, float, int, std::string, int>(),
             py::arg("steps"),
             py::arg("timestep"),
             py::arg("queues") = 1,
             py::arg("event_queue") = "heap",
             py::arg("timing_wheel_size") = 0)
        .def_readwrite("steps", &NativeSimulationConfig::steps)
        .def_readwrite("timestep", &NativeSimulationConfig::timestep)
        .def_readwrite("queues", &NativeSimulationConfig::queues)
        .def_readwrite("event_queue", &NativeSimulationConfig::event_queue)
        .def_readwrite("timing_wheel_size", &NativeSimulationConfig::timing_wheel_size)
        .def("to_dict", [](const NativeSimulationConfig& config) {
            py::dict data;
            data["steps"] = config.steps;
            data["timestep"] = config.timestep;
            data["queues"] = config.queues;
            data["event_queue"] = config.event_queue;
            data["timing_wheel_size"] = config.timing_wheel_size;
            return data;
        });

    py::class_<NativeNetworkDescription>(module, "NetworkDescription")
        .def(py::init<>())
        .def("add_layer",
             &AddLayer,
             py::arg("model"),
             py::arg("count"),
             py::arg("update_timestep") = 1,
             py::arg("monitored") = false,
             py::arg("output") = false,
             py::arg("communication_input") = false,
             py::arg("parameters") = py::dict())
        .def("add_connection",
             &AddConnection,
             py::arg("source"),
             py::arg("target"),
             py::arg("synapse_type"),
             py::arg("weight"),
             py::arg("max_weight"),
             py::arg("delay"),
             py::arg("synapse_rule"),
             py::arg("trigger_rule"))
        .def("add_learning_rule",
             &AddLearningRule,
             py::arg("name"),
             py::arg("parameters") = py::dict())
        .def("add_outer_dynamic",
             &AddOuterDynamic,
             py::arg("model"),
             py::arg("name") = "",
             py::arg("parameters") = py::dict(),
             py::arg("update_timestep") = 1,
             py::arg("communication_interval") = 1,
             py::arg("queue_index") = 0,
             py::arg("gc_neuron_indices_by_joint") = std::vector<std::vector<int> >(),
             py::arg("mf_neuron_indices_by_joint") = std::vector<std::vector<int> >(),
             py::arg("cf_positive_neuron_indices_by_joint") = std::vector<std::vector<int> >(),
             py::arg("cf_negative_neuron_indices_by_joint") = std::vector<std::vector<int> >(),
             py::arg("dcn_positive_neuron_indices_by_joint") = std::vector<std::vector<int> >(),
             py::arg("dcn_negative_neuron_indices_by_joint") = std::vector<std::vector<int> >(),
             py::arg("state_feedback_product") = py::dict(),
             py::arg("state_feedback_single") = py::dict(),
             py::arg("error_feedback_product") = py::dict(),
             py::arg("error_feedback_single") = py::dict())
        .def("add_outer_dynamic_connection",
             &AddOuterDynamicConnection,
             py::arg("source"),
             py::arg("target_outer_dynamic"),
             py::arg("target_joint"),
             py::arg("synapse_type"),
             py::arg("weight"),
             py::arg("delay"))
        .def("add_input_conv",
             &AddInputConv,
             py::arg("model"),
             py::arg("parameters") = py::dict(),
             py::arg("update_timestep") = 1,
             py::arg("queue_index") = 0,
             py::arg("output_target") = "main_network",
             py::arg("target_dense_subnetwork_name") = "",
             py::arg("output_source_indices") = std::vector<int>(),
             py::arg("output_target_neuron_ids") = std::vector<int>(),
             py::arg("output_pending_channel") = 0,
             py::arg("output_scale") = 1.0f,
             py::arg("output_scales") = std::vector<float>(),
             py::arg("output_overwrite") = true)
        .def_property_readonly("layer_count", &NativeNetworkDescription::layer_count)
        .def_property_readonly("connection_block_count", &NativeNetworkDescription::connection_block_count)
        .def_property_readonly("learning_rule_count", &NativeNetworkDescription::learning_rule_count)
        .def_property_readonly("outer_dynamic_count", &NativeNetworkDescription::outer_dynamic_count)
        .def_property_readonly("outer_dynamic_connection_block_count",
                               &NativeNetworkDescription::outer_dynamic_connection_block_count)
        .def_property_readonly("input_conv_count", &NativeNetworkDescription::input_conv_count)
        .def_property_readonly("neuron_count", &NativeNetworkDescription::neuron_count)
        .def("to_dict", &NetworkDescriptionToDict);

    py::class_<NativeSimulation>(module, "Simulation")
        .def(py::init<const NativeNetworkDescription&, const NativeSimulationConfig&>(),
             py::arg("network"),
             py::arg("config"))
        .def("init", &NativeSimulation::init)
        .def("run", &NativeSimulation::run, py::arg("steps"))
        .def("enable_realtime",
             &NativeSimulation::enable_realtime,
             py::arg("slot_steps"),
             py::arg("max_advance_seconds"),
             py::arg("first_section"),
             py::arg("second_section"),
             py::arg("third_section"))
        .def("disable_realtime", &NativeSimulation::disable_realtime)
        .def("run_realtime", &NativeSimulation::run_realtime, py::arg("steps"))
        .def("reset_bench_profiling", &NativeSimulation::reset_bench_profiling)
        .def("bench_profiling_snapshot", &NativeSimulation::bench_profiling_snapshot)
        .def("realtime_skip_counters", &NativeSimulation::realtime_skip_counters)
        .def("reset_realtime_skip_counters", &NativeSimulation::reset_realtime_skip_counters)
        .def("realtime_restriction_counts", &NativeSimulation::realtime_restriction_counts)
        .def("reset_realtime_restriction_counts", &NativeSimulation::reset_realtime_restriction_counts)
        .def("reset", &NativeSimulation::reset, py::arg("preserve_weights") = true)
        .def("add_external_spikes",
             &NativeSimulation::add_external_spikes,
             py::arg("times"),
             py::arg("neuron_ids"))
        .def("add_external_currents",
             &NativeSimulation::add_external_currents,
             py::arg("times"),
             py::arg("neuron_ids"),
             py::arg("currents"))
        .def("add_zmq_async_input_output_spike_driver",
             &NativeSimulation::add_zmq_async_input_output_spike_driver,
             py::arg("subscribe_address"),
             py::arg("publish_port"),
             py::arg("subscribe_port"),
             py::arg("publish_topic"),
             py::arg("subscribe_topic"),
             py::arg("communication_interval"))
        .def("add_zmq_input_output_spike_driver",
             &NativeSimulation::add_zmq_input_output_spike_driver,
             py::arg("server_address"),
             py::arg("server_port"),
             py::arg("communication_interval"))
        .def("add_input_conv_frames",
             &NativeSimulation::add_input_conv_frames,
             py::arg("source_name"),
             py::arg("frames"))
        .def("clear_input_conv_frame_queue",
             &NativeSimulation::clear_input_conv_frame_queue,
             py::arg("source_name"))
        .def("bind_input_conv_frame_source",
             py::overload_cast<int, const std::string&, int>(&NativeSimulation::bind_input_conv_frame_source_index),
             py::arg("inputconv_index"),
             py::arg("source_name"),
             py::arg("source_camera_index") = 0)
        .def("bind_input_conv_frame_source",
             py::overload_cast<const std::string&, const std::string&, int>(&NativeSimulation::bind_input_conv_frame_source_name),
             py::arg("inputconv_name"),
             py::arg("source_name"),
             py::arg("source_camera_index") = 0)
        .def("has_input_conv_frame_source_binding",
             &NativeSimulation::has_input_conv_frame_source_binding,
             py::arg("inputconv_index"))
        .def("add_zmq_input_conv_frame_source",
             &NativeSimulation::add_zmq_input_conv_frame_source,
             py::arg("source_name"),
             py::arg("address"),
             py::arg("port"),
             py::arg("max_payload_bytes") = 0)
        .def("add_zmq_async_input_conv_frame_source",
             &NativeSimulation::add_zmq_async_input_conv_frame_source,
             py::arg("source_name"),
             py::arg("subscribe_address"),
             py::arg("subscribe_port"),
             py::arg("topic"),
             py::arg("max_payload_bytes") = 0,
             py::arg("max_buffered_frames_per_camera") = 8)
        .def("input_conv_frame_source_status",
             py::overload_cast<const std::string&>(&NativeSimulation::input_conv_frame_source_status_name, py::const_),
             py::arg("source_name"))
        .def("input_conv_frame_source_status",
             py::overload_cast<int>(&NativeSimulation::input_conv_frame_source_status_index, py::const_),
             py::arg("source_index"))
        .def("neuron_state",
             &NativeSimulation::neuron_state,
             py::arg("original_neuron_id"))
        .def("neuron_states",
             &NativeSimulation::neuron_states,
             py::arg("original_neuron_ids"))
        .def("output_spikes", &NativeSimulation::output_spikes)
        .def("enable_debug_monitor",
             &NativeSimulation::enable_debug_monitor,
             py::arg("config"))
        .def("disable_debug_monitor", &NativeSimulation::disable_debug_monitor)
        .def("flush_debug_monitor", &NativeSimulation::flush_debug_monitor)
        .def("publish_output", &NativeSimulation::publish_output)
        .def("get_connection_weight",
             &NativeSimulation::get_connection_weight,
             py::arg("original_connection_index"))
        .def("set_connection_weight",
             &NativeSimulation::set_connection_weight,
             py::arg("original_connection_index"),
             py::arg("weight"))
        .def("save_weights", &NativeSimulation::save_weights, py::arg("path"))
        .def("load_weights", &NativeSimulation::load_weights, py::arg("path"))
        .def_property_readonly("dense_subnetwork_count", &NativeSimulation::dense_subnetwork_count)
        .def("dense_subnetwork_name",
             &NativeSimulation::dense_subnetwork_name,
             py::arg("index"))
        .def("dense_subnetwork_weights",
             &NativeSimulation::dense_subnetwork_weights,
             py::arg("index"))
        .def("find_dense_subnetwork",
             &NativeSimulation::find_dense_subnetwork,
             py::arg("name"))
        .def("dense_subnetwork_snapshot",
             &NativeSimulation::dense_subnetwork_snapshot,
             py::arg("index"))
        .def("dense_subnetwork_snapshot_by_name",
             &NativeSimulation::dense_subnetwork_snapshot_by_name,
             py::arg("name"))
        .def("reset_dense_subnetwork",
             &NativeSimulation::reset_dense_subnetwork,
             py::arg("index"))
        .def("set_dense_subnetwork_full_firing_export_enabled",
             &NativeSimulation::set_dense_subnetwork_full_firing_export_enabled,
             py::arg("index"),
             py::arg("enabled"))
        .def_property_readonly("input_conv_count", &NativeSimulation::input_conv_count)
        .def("input_conv_output_count",
             &NativeSimulation::input_conv_output_count,
             py::arg("index"))
        .def("input_conv_output",
             &NativeSimulation::input_conv_output,
             py::arg("index"))
        .def("input_conv_input",
             &NativeSimulation::input_conv_input,
             py::arg("index"))
        .def("enable_input_conv_monitor",
             py::overload_cast<int>(&NativeSimulation::enable_input_conv_monitor_index),
             py::arg("index"))
        .def("enable_input_conv_monitor",
             py::overload_cast<const std::string&>(&NativeSimulation::enable_input_conv_monitor_name),
             py::arg("name"))
        .def("disable_input_conv_monitor",
             py::overload_cast<int>(&NativeSimulation::disable_input_conv_monitor_index),
             py::arg("index"))
        .def("disable_input_conv_monitor",
             py::overload_cast<const std::string&>(&NativeSimulation::disable_input_conv_monitor_name),
             py::arg("name"))
        .def("outer_dynamic_spike_counter_snapshot",
             &NativeSimulation::outer_dynamic_spike_counter_snapshot,
             py::arg("name"))
        .def("clear_outer_dynamic_spike_counter",
             &NativeSimulation::clear_outer_dynamic_spike_counter,
             py::arg("name"))
        .def("clear_outer_dynamic_spike_counter_slot",
             &NativeSimulation::clear_outer_dynamic_spike_counter_slot,
             py::arg("name"),
             py::arg("slot_id"))
        .def("outer_dynamic_state", &NativeSimulation::outer_dynamic_state)
        .def("outer_dynamic_states", &NativeSimulation::outer_dynamic_states)
        .def("reset_outer_dynamic_state",
             &NativeSimulation::reset_outer_dynamic_state,
             py::arg("name"),
             py::arg("q"),
             py::arg("qd"))
        .def("set_outer_dynamic_desired_state",
             &NativeSimulation::set_outer_dynamic_desired_state,
             py::arg("name"),
             py::arg("q_des"),
             py::arg("qd_des"))
        .def_property_readonly("initialized", &NativeSimulation::initialized);

    module.def("get_build_info", &GetBuildInfo, "Return build metadata for the native bridge.");
    module.def(
        "component_kind_name",
        [](npgr::DebugComponentKind kind) {
            return std::string(npgr::DebugComponentKindName(kind));
        },
        "Return the native debug component kind name.");
}
