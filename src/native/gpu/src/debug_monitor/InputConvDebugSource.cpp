#include "debug_monitor/InputConvDebugSource.h"

#include "source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"

#include <algorithm>

namespace npgr {

InputConvDebugSource::InputConvDebugSource(InputConvModel* model,
                                           int inputconv_index,
                                           const std::string& name)
    : model_(model),
      inputconv_index_(inputconv_index),
      name_(name) {}

DebugComponentKind InputConvDebugSource::kind() const {
    return DebugComponentKind::InputConv;
}

int InputConvDebugSource::component_index() const {
    return inputconv_index_;
}

const std::string& InputConvDebugSource::component_name() const {
    return name_;
}

bool InputConvDebugSource::HasAnyTarget(const DebugMonitorConfig& config) const {
    return model_ != nullptr && this->IsSelected(config);
}

bool InputConvDebugSource::Capture(int time_step,
                                   const DebugMonitorConfig& config,
                                   DebugMonitorFrame* frame,
                                   std::string* reason) {
    if (frame == nullptr) {
        if (reason != nullptr) {
            *reason = "InputConv debug source requires an output frame";
        }
        return false;
    }
    if (model_ == nullptr || !this->IsSelected(config)) {
        return true;
    }
    if (config.record_inputconv_outputs) {
        std::vector<float> output;
        if (!model_->ExportMonitorOutput(&output, reason)) {
            return false;
        }
        for (std::size_t index = 0; index < output.size(); ++index) {
            DebugInputConvOutputRecord record;
            record.time_step = time_step;
            record.inputconv_index = inputconv_index_;
            record.inputconv_name = name_;
            record.output_index = static_cast<int>(index);
            record.value = output[index];
            frame->inputconv_outputs.push_back(record);
        }
    }
    if (config.record_inputconv_inputs) {
        std::vector<float> input;
        if (!model_->ExportMonitorInput(&input, reason)) {
            return false;
        }
        for (std::size_t index = 0; index < input.size(); ++index) {
            DebugInputConvInputRecord record;
            record.time_step = time_step;
            record.inputconv_index = inputconv_index_;
            record.inputconv_name = name_;
            record.input_index = static_cast<int>(index);
            record.value = input[index];
            frame->inputconv_inputs.push_back(record);
        }
    }
    if (config.record_inputconv_internal_state) {
        std::vector<DebugInputConvStateRecord> states;
        if (!model_->ExportMonitorState(&states, reason)) {
            return false;
        }
        for (std::size_t index = 0; index < states.size(); ++index) {
            DebugInputConvStateRecord record = states[index];
            record.time_step = time_step;
            record.inputconv_index = inputconv_index_;
            record.inputconv_name = name_;
            frame->inputconv_states.push_back(record);
        }
    }
    return true;
}

bool InputConvDebugSource::IsSelected(const DebugMonitorConfig& config) const {
    if (config.monitor_all_inputconv) {
        return true;
    }
    if (std::find(config.monitored_inputconv_indices.begin(),
                  config.monitored_inputconv_indices.end(),
                  inputconv_index_) != config.monitored_inputconv_indices.end()) {
        return true;
    }
    return std::find(config.monitored_inputconv_names.begin(),
                     config.monitored_inputconv_names.end(),
                     name_) != config.monitored_inputconv_names.end();
}

}  // namespace npgr
