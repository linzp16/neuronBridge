#include "debug_monitor/DebugMonitorWriter.h"

#include <filesystem>
#include <sstream>

namespace npgr {

DebugMonitorWriter::DebugMonitorWriter()
    : initialized_(false),
      headers_written_(false) {}

DebugMonitorWriter::~DebugMonitorWriter() {
    this->Flush(nullptr);
}

bool DebugMonitorWriter::Initialize(const DebugMonitorConfig& config, std::string* reason) {
    this->CloseStreams();
    config_ = config;
    initialized_ = false;
    headers_written_ = false;
    if (config_.output_dir.empty()) {
        if (reason != nullptr) {
            *reason = "debug monitor output_dir must not be empty";
        }
        return false;
    }
    if (!this->EnsureOpen(reason)) {
        return false;
    }
    if (!this->WriteMeta(reason)) {
        return false;
    }
    initialized_ = true;
    return true;
}

bool DebugMonitorWriter::AppendFrame(const DebugMonitorFrame& frame, std::string* reason) {
    if (!initialized_ && !this->EnsureOpen(reason)) {
        return false;
    }
    if (!headers_written_ && !this->WriteHeaders(reason)) {
        return false;
    }

    for (std::size_t i = 0; i < frame.neuron_states.size(); ++i) {
        const DebugNeuronStateRecord& r = frame.neuron_states[i];
        neuron_state_out_ << r.time_step << ','
                          << DebugComponentKindName(r.component_kind) << ','
                          << r.component_index << ','
                          << r.component_name << ','
                          << r.global_neuron_id << ','
                          << r.local_neuron_id << ','
                          << r.field_name << ','
                          << r.value << '\n';
    }
    for (std::size_t i = 0; i < frame.spikes.size(); ++i) {
        const DebugSpikeRecord& r = frame.spikes[i];
        spikes_out_ << r.time_step << ','
                    << DebugComponentKindName(r.component_kind) << ','
                    << r.component_index << ','
                    << r.component_name << ','
                    << r.global_neuron_id << ','
                    << r.local_neuron_id << '\n';
    }
    for (std::size_t i = 0; i < frame.pending_channels.size(); ++i) {
        const DebugPendingChannelRecord& r = frame.pending_channels[i];
        pending_channels_out_ << r.time_step << ','
                              << r.subnetwork_index << ','
                              << r.subnetwork_name << ','
                              << r.channel << ','
                              << r.local_neuron_id << ','
                              << r.global_neuron_id << ','
                              << r.value << '\n';
    }
    for (std::size_t i = 0; i < frame.inputconv_outputs.size(); ++i) {
        const DebugInputConvOutputRecord& r = frame.inputconv_outputs[i];
        inputconv_outputs_out_ << r.time_step << ','
                               << r.inputconv_index << ','
                               << r.inputconv_name << ','
                               << r.output_index << ','
                               << r.value << '\n';
    }
    for (std::size_t i = 0; i < frame.inputconv_inputs.size(); ++i) {
        const DebugInputConvInputRecord& r = frame.inputconv_inputs[i];
        inputconv_inputs_out_ << r.time_step << ','
                              << r.inputconv_index << ','
                              << r.inputconv_name << ','
                              << r.input_index << ','
                              << r.value << '\n';
    }
    for (std::size_t i = 0; i < frame.inputconv_states.size(); ++i) {
        const DebugInputConvStateRecord& r = frame.inputconv_states[i];
        inputconv_state_out_ << r.time_step << ','
                             << r.inputconv_index << ','
                             << r.inputconv_name << ','
                             << r.field_name << ','
                             << r.state_index << ','
                             << r.value << '\n';
    }
    for (std::size_t i = 0; i < frame.outer_dynamic_states.size(); ++i) {
        const DebugOuterDynamicStateRecord& r = frame.outer_dynamic_states[i];
        outer_dynamic_state_out_ << r.time_step << ','
                                 << r.component_index << ','
                                 << r.component_name << ','
                                 << r.field_name << ','
                                 << r.index << ','
                                 << r.value << '\n';
    }
    for (std::size_t i = 0; i < frame.weights.size(); ++i) {
        const DebugWeightRecord& r = frame.weights[i];
        weights_out_ << r.time_step << ','
                     << DebugComponentKindName(r.component_kind) << ','
                     << r.component_index << ','
                     << r.component_name << ','
                     << r.synapse_id << ','
                     << r.value << '\n';
    }
    return true;
}

bool DebugMonitorWriter::Flush(std::string* reason) {
    (void)reason;
    if (neuron_state_out_.is_open()) {
        neuron_state_out_.flush();
    }
    if (spikes_out_.is_open()) {
        spikes_out_.flush();
    }
    if (pending_channels_out_.is_open()) {
        pending_channels_out_.flush();
    }
    if (inputconv_outputs_out_.is_open()) {
        inputconv_outputs_out_.flush();
    }
    if (inputconv_inputs_out_.is_open()) {
        inputconv_inputs_out_.flush();
    }
    if (inputconv_state_out_.is_open()) {
        inputconv_state_out_.flush();
    }
    if (outer_dynamic_state_out_.is_open()) {
        outer_dynamic_state_out_.flush();
    }
    if (weights_out_.is_open()) {
        weights_out_.flush();
    }
    return true;
}

bool DebugMonitorWriter::initialized() const {
    return initialized_;
}

void DebugMonitorWriter::CloseStreams() {
    this->Flush(nullptr);
    if (neuron_state_out_.is_open()) {
        neuron_state_out_.close();
    }
    if (spikes_out_.is_open()) {
        spikes_out_.close();
    }
    if (pending_channels_out_.is_open()) {
        pending_channels_out_.close();
    }
    if (inputconv_outputs_out_.is_open()) {
        inputconv_outputs_out_.close();
    }
    if (inputconv_inputs_out_.is_open()) {
        inputconv_inputs_out_.close();
    }
    if (inputconv_state_out_.is_open()) {
        inputconv_state_out_.close();
    }
    if (outer_dynamic_state_out_.is_open()) {
        outer_dynamic_state_out_.close();
    }
    if (weights_out_.is_open()) {
        weights_out_.close();
    }
}

bool DebugMonitorWriter::EnsureOpen(std::string* reason) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(fs::path(config_.output_dir), ec);
    if (ec) {
        if (reason != nullptr) {
            *reason = "failed to create debug monitor output directory: " + ec.message();
        }
        return false;
    }
    const fs::path root(config_.output_dir);
    neuron_state_out_.open((root / "neuron_state.csv").string().c_str(), std::ios::out | std::ios::trunc);
    spikes_out_.open((root / "spikes.csv").string().c_str(), std::ios::out | std::ios::trunc);
    pending_channels_out_.open((root / "pending_channels.csv").string().c_str(), std::ios::out | std::ios::trunc);
    inputconv_outputs_out_.open((root / "inputconv_outputs.csv").string().c_str(), std::ios::out | std::ios::trunc);
    inputconv_inputs_out_.open((root / "inputconv_inputs.csv").string().c_str(), std::ios::out | std::ios::trunc);
    inputconv_state_out_.open((root / "inputconv_state.csv").string().c_str(), std::ios::out | std::ios::trunc);
    outer_dynamic_state_out_.open((root / "outer_dynamic_state.csv").string().c_str(), std::ios::out | std::ios::trunc);
    weights_out_.open((root / "weights.csv").string().c_str(), std::ios::out | std::ios::trunc);
    if (!neuron_state_out_.is_open() ||
        !spikes_out_.is_open() ||
        !pending_channels_out_.is_open() ||
        !inputconv_outputs_out_.is_open() ||
        !inputconv_inputs_out_.is_open() ||
        !inputconv_state_out_.is_open() ||
        !outer_dynamic_state_out_.is_open() ||
        !weights_out_.is_open()) {
        if (reason != nullptr) {
            *reason = "failed to open one or more debug monitor output files";
        }
        return false;
    }
    return true;
}

bool DebugMonitorWriter::WriteMeta(std::string* reason) {
    namespace fs = std::filesystem;
    std::ofstream meta((fs::path(config_.output_dir) / "meta.json").string().c_str(),
                       std::ios::out | std::ios::trunc);
    if (!meta.is_open()) {
        if (reason != nullptr) {
            *reason = "failed to open debug monitor meta.json";
        }
        return false;
    }
    meta << "{\n";
    meta << "  \"sample_interval_steps\": " << config_.sample_interval_steps << ",\n";
    meta << "  \"flush_interval_steps\": " << config_.flush_interval_steps << ",\n";
    meta << "  \"record_state\": " << (config_.record_state ? "true" : "false") << ",\n";
    meta << "  \"record_spikes\": " << (config_.record_spikes ? "true" : "false") << ",\n";
    meta << "  \"record_weights\": " << (config_.record_weights ? "true" : "false") << ",\n";
    meta << "  \"record_pending_channels\": " << (config_.record_pending_channels ? "true" : "false") << ",\n";
    meta << "  \"record_inputconv_outputs\": " << (config_.record_inputconv_outputs ? "true" : "false") << ",\n";
    meta << "  \"record_outer_dynamic_state\": " << (config_.record_outer_dynamic_state ? "true" : "false") << "\n";
    meta << "}\n";
    return true;
}

bool DebugMonitorWriter::WriteHeaders(std::string* reason) {
    (void)reason;
    neuron_state_out_ << "time_step,component_kind,component_index,component_name,global_neuron_id,local_neuron_id,field_name,value\n";
    spikes_out_ << "time_step,component_kind,component_index,component_name,global_neuron_id,local_neuron_id\n";
    pending_channels_out_ << "time_step,subnetwork_index,subnetwork_name,channel,local_neuron_id,global_neuron_id,value\n";
    inputconv_outputs_out_ << "time_step,inputconv_index,inputconv_name,output_index,value\n";
    inputconv_inputs_out_ << "time_step,inputconv_index,inputconv_name,input_index,value\n";
    inputconv_state_out_ << "time_step,inputconv_index,inputconv_name,field_name,state_index,value\n";
    outer_dynamic_state_out_ << "time_step,component_index,component_name,field_name,index,value\n";
    weights_out_ << "time_step,component_kind,component_index,component_name,synapse_id,value\n";
    headers_written_ = true;
    return true;
}

}  // namespace npgr
