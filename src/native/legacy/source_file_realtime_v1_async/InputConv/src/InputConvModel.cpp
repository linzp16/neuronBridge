#include "../source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"

InputConvModel::InputConvModel() : timestep_size_(1), queue_index_(0), inputconv_index_(-1) {}

InputConvModel::InputConvModel(int timestep_size)
    : timestep_size_(timestep_size), queue_index_(0), inputconv_index_(-1) {}

InputConvModel::~InputConvModel() {}

void InputConvModel::Reset(Simulation* simulation) {
    (void)simulation;
}

const float* InputConvModel::GetDeviceOutputBuffer() const {
    return 0;
}

int InputConvModel::GetOutputCount() const {
    return 0;
}

bool InputConvModel::HasDeviceOutputBuffer() const {
    return false;
}

bool InputConvModel::ExportMonitorOutput(std::vector<float>* output, std::string* reason) const {
    (void)reason;
    if (output != 0) {
        output->clear();
    }
    return true;
}

bool InputConvModel::ExportMonitorInput(std::vector<float>* input, std::string* reason) const {
    (void)reason;
    if (input != 0) {
        input->clear();
    }
    return true;
}

bool InputConvModel::ExportMonitorState(std::vector<npgr::DebugInputConvStateRecord>* state,
                                        std::string* reason) const {
    (void)reason;
    if (state != 0) {
        state->clear();
    }
    return true;
}

bool InputConvModel::UsesDynamicFrameInput() const {
    return false;
}

InputConvFrameRequest InputConvModel::GetFrameRequest(int time_step,
                                                      int inputconv_index,
                                                      int source_camera_index) const {
    InputConvFrameRequest request;
    request.time_step = time_step;
    request.inputconv_index = inputconv_index;
    request.source_camera_index = source_camera_index;
    return request;
}

bool InputConvModel::AcceptFrame(const InputConvFrame& frame, std::string* reason) {
    (void)frame;
    if (reason != 0) {
        *reason = "This InputConv model does not accept dynamic frames";
    }
    return false;
}

void InputConvModel::ApplyMissingFramePolicy(int time_step) {
    (void)time_step;
}

int InputConvModel::getTimestepSize() const {
    return this->timestep_size_;
}

void InputConvModel::setTimestepSize(int timestep_size) {
    this->timestep_size_ = timestep_size;
}

int InputConvModel::getQueueIndex() const {
    return this->queue_index_;
}

void InputConvModel::setQueueIndex(int queue_index) {
    this->queue_index_ = queue_index;
}

int InputConvModel::getInputConvIndex() const {
    return this->inputconv_index_;
}

void InputConvModel::setInputConvIndex(int inputconv_index) {
    this->inputconv_index_ = inputconv_index;
}
