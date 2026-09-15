#include "debug_monitor/SimulationDebugMonitor.h"

#include "debug_monitor/DenseSubnetworkDebugSource.h"
#include "debug_monitor/InputConvDebugSource.h"
#include "debug_monitor/MainNetworkDebugSource.h"
#include "debug_monitor/OuterDynamicDebugSource.h"
#include "simulation_dense/DenseSubnetworkModel.h"
#include "source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"

#include <algorithm>
#include <sstream>

namespace npgr {

SimulationDebugMonitor::SimulationDebugMonitor()
    : simulation_(nullptr),
      enabled_(false),
      samples_since_flush_(0) {}

SimulationDebugMonitor::~SimulationDebugMonitor() {
    this->Flush(nullptr);
}

bool SimulationDebugMonitor::Initialize(Simulation* simulation,
                                        const DebugMonitorConfig& config,
                                        std::string* reason) {
    simulation_ = simulation;
    config_ = config;
    enabled_ = false;
    samples_since_flush_ = 0;
    sources_.clear();
    if (simulation_ == nullptr) {
        if (reason != nullptr) {
            *reason = "debug monitor requires a simulation";
        }
        return false;
    }
    if (config_.sample_interval_steps <= 0) {
        config_.sample_interval_steps = 1;
    }
    if (config_.flush_interval_steps <= 0) {
        config_.flush_interval_steps = 100;
    }
    this->RegisterSources(simulation_);
    if (!config_.enabled && !this->HasAnySourceTarget()) {
        return true;
    }
    config_.enabled = true;
    if (!writer_.Initialize(config_, reason)) {
        return false;
    }
    enabled_ = true;
    return true;
}

bool SimulationDebugMonitor::CaptureStep(int time_step, std::string* reason) {
    if (!enabled_) {
        return true;
    }
    if (config_.sample_interval_steps > 1 &&
        time_step % config_.sample_interval_steps != 0) {
        return true;
    }
    DebugMonitorFrame frame;
    for (std::size_t index = 0; index < sources_.size(); ++index) {
        IDebugMonitorSource* source = sources_[index].get();
        if (source == nullptr || !source->HasAnyTarget(config_)) {
            continue;
        }
        if (!source->Capture(time_step, config_, &frame, reason)) {
            return false;
        }
    }
    if (!frame.empty() && !writer_.AppendFrame(frame, reason)) {
        return false;
    }
    ++samples_since_flush_;
    if (samples_since_flush_ >= config_.flush_interval_steps) {
        samples_since_flush_ = 0;
        return writer_.Flush(reason);
    }
    return true;
}

bool SimulationDebugMonitor::Flush(std::string* reason) {
    return writer_.Flush(reason);
}

bool SimulationDebugMonitor::EnableInputConvMonitor(int inputconv_index, std::string* reason) {
    if (inputconv_index < 0) {
        if (reason != nullptr) {
            *reason = "InputConv monitor index must be non-negative";
        }
        return false;
    }
    if (std::find(config_.monitored_inputconv_indices.begin(),
                  config_.monitored_inputconv_indices.end(),
                  inputconv_index) == config_.monitored_inputconv_indices.end()) {
        config_.monitored_inputconv_indices.push_back(inputconv_index);
    }
    config_.enabled = true;
    config_.record_inputconv_outputs = true;
    enabled_ = writer_.initialized();
    return true;
}

bool SimulationDebugMonitor::DisableInputConvMonitor(int inputconv_index, std::string* reason) {
    (void)reason;
    config_.monitored_inputconv_indices.erase(
        std::remove(config_.monitored_inputconv_indices.begin(),
                    config_.monitored_inputconv_indices.end(),
                    inputconv_index),
        config_.monitored_inputconv_indices.end());
    return true;
}

bool SimulationDebugMonitor::enabled() const {
    return enabled_;
}

const DebugMonitorConfig& SimulationDebugMonitor::config() const {
    return config_;
}

void SimulationDebugMonitor::RegisterSources(Simulation* simulation) {
    if (simulation == nullptr) {
        return;
    }
    sources_.push_back(std::unique_ptr<IDebugMonitorSource>(
        new MainNetworkDebugSource(simulation->network, simulation->original_to_main_neuron_id)));
    for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
        if (simulation->dense_subnetworks[index] != nullptr) {
            // Dense spike monitoring needs the runtime to keep the current-step
            // firing list on the host; users only configure record_spikes.
            simulation->dense_subnetworks[index]->SetFullFiringExportEnabled(config_.record_spikes);
            sources_.push_back(std::unique_ptr<IDebugMonitorSource>(
                new DenseSubnetworkDebugSource(simulation->dense_subnetworks[index],
                                               static_cast<int>(index))));
        }
    }
    for (std::size_t index = 0; index < simulation->InputConvModelList.size(); ++index) {
        InputConvModel* model = simulation->InputConvModelList[index];
        if (model == nullptr) {
            continue;
        }
        std::ostringstream name;
        if (index < simulation->InputConvDescriptionList.size() &&
            !simulation->InputConvDescriptionList[index].ModelName.empty()) {
            name << simulation->InputConvDescriptionList[index].ModelName;
        } else {
            name << "InputConv";
        }
        name << "#" << index;
        sources_.push_back(std::unique_ptr<IDebugMonitorSource>(
            new InputConvDebugSource(model, static_cast<int>(index), name.str())));
    }
    sources_.push_back(std::unique_ptr<IDebugMonitorSource>(
        new OuterDynamicDebugSource(simulation)));
}

bool SimulationDebugMonitor::HasAnySourceTarget() const {
    for (std::size_t index = 0; index < sources_.size(); ++index) {
        if (sources_[index] != nullptr && sources_[index]->HasAnyTarget(config_)) {
            return true;
        }
    }
    return false;
}

}  // namespace npgr
