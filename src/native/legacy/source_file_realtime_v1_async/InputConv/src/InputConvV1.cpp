#include "../source_file_realtime_v1_async/InputConv/inc/InputConvV1.h"

#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"

#if NPGR_ENABLE_CUDA
#include "input_conv/InputConvV1CudaHelpers.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <boost/any.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct StimulusFileHeader {
    int type = 0;
    int channels = 0;
    int width = 0;
    int height = 0;
    int length = 0;
};

template <typename T>
T GetAnyOrDefault(const std::map<std::string, boost::any>& values,
                  const char* key,
                  const T& fallback) {
    std::map<std::string, boost::any>::const_iterator found = values.find(key);
    if (found == values.end()) {
        return fallback;
    }
    try {
        return boost::any_cast<T>(found->second);
    } catch (const boost::bad_any_cast&) {
        return fallback;
    }
}

void AddPlaidFrame(std::vector<unsigned char>* frame,
                   int width,
                   int height,
                   int frame_index,
                   float temporal_speed,
                   float direction_a_deg,
                   float direction_b_deg,
                   float spatial_frequency) {
    if (frame == 0) {
        return;
    }
    const float phase_t = temporal_speed * static_cast<float>(frame_index);
    const float theta_a = direction_a_deg * kPi / 180.0f;
    const float theta_b = direction_b_deg * kPi / 180.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float xf = (static_cast<float>(x) - width * 0.5f) / static_cast<float>(width);
            const float yf = (static_cast<float>(y) - height * 0.5f) / static_cast<float>(height);
            const float proj_a = std::cos(theta_a) * xf + std::sin(theta_a) * yf;
            const float proj_b = std::cos(theta_b) * xf + std::sin(theta_b) * yf;
            const float g1 = std::sin(2.0f * kPi * (spatial_frequency * proj_a - phase_t));
            const float g2 = std::sin(2.0f * kPi * (spatial_frequency * proj_b - phase_t));
            float value = 127.5f + 63.75f * (g1 + g2);
            value = std::max(0.0f, std::min(255.0f, value));
            (*frame)[static_cast<std::size_t>(y * width + x)] = static_cast<unsigned char>(value);
        }
    }
}

void AddGratingFrame(std::vector<unsigned char>* frame,
                     int width,
                     int height,
                     int frame_index,
                     float temporal_speed,
                     float direction_deg,
                     float spatial_frequency) {
    if (frame == 0) {
        return;
    }
    const float phase_t = temporal_speed * static_cast<float>(frame_index);
    const float theta = direction_deg * kPi / 180.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float xf = (static_cast<float>(x) - width * 0.5f) / static_cast<float>(width);
            const float yf = (static_cast<float>(y) - height * 0.5f) / static_cast<float>(height);
            const float proj = std::cos(theta) * xf + std::sin(theta) * yf;
            float value = 127.5f + 127.5f * std::sin(2.0f * kPi * (spatial_frequency * proj - phase_t));
            value = std::max(0.0f, std::min(255.0f, value));
            (*frame)[static_cast<std::size_t>(y * width + x)] = static_cast<unsigned char>(value);
        }
    }
}

bool ReadStimulusHeader(std::ifstream* input, StimulusFileHeader* header) {
    if (input == 0 || header == 0) {
        return false;
    }
    int32_t signature = 0;
    float version = 0.0f;
    int32_t type = 0;
    char channels = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t length = 0;
    input->read(reinterpret_cast<char*>(&signature), sizeof(signature));
    input->read(reinterpret_cast<char*>(&version), sizeof(version));
    input->read(reinterpret_cast<char*>(&type), sizeof(type));
    input->read(reinterpret_cast<char*>(&channels), sizeof(channels));
    input->read(reinterpret_cast<char*>(&width), sizeof(width));
    input->read(reinterpret_cast<char*>(&height), sizeof(height));
    input->read(reinterpret_cast<char*>(&length), sizeof(length));
    if (!input->good()) {
        return false;
    }
    header->type = type;
    header->channels = static_cast<int>(channels);
    header->width = width;
    header->height = height;
    header->length = length;
    return signature == 293390619 && std::abs(version - 1.0f) < 1e-6f;
}

}  // namespace

InputConvV1::InputConvV1()
    : InputConvModel(),
      width_(0),
      height_(0),
      channels_(0),
      output_count_(0),
      stimulus_mode_("bar"),
      speed_(1.5f),
      stimulus_temporal_speed_(0.08f),
      grating_direction_deg_(0.0f),
      plaid_direction_a_deg_(0.0f),
      plaid_direction_b_deg_(120.0f),
      spatial_frequency_(2.2f),
      bar_width_(2),
      motion_period_steps_(32),
      frame_hold_steps_(1),
      stimulus_file_frame_count_(0),
      input_frame_missing_policy_("zero"),
      input_frame_max_lag_steps_(0),
      has_last_dynamic_frame_(false),
      cuda_runtime_handle_(0) {}

InputConvV1::InputConvV1(int timestep_size)
    : InputConvModel(timestep_size),
      width_(0),
      height_(0),
      channels_(0),
      output_count_(0),
      stimulus_mode_("bar"),
      speed_(1.5f),
      stimulus_temporal_speed_(0.08f),
      grating_direction_deg_(0.0f),
      plaid_direction_a_deg_(0.0f),
      plaid_direction_b_deg_(120.0f),
      spatial_frequency_(2.2f),
      bar_width_(2),
      motion_period_steps_(32),
      frame_hold_steps_(1),
      stimulus_file_frame_count_(0),
      input_frame_missing_policy_("zero"),
      input_frame_max_lag_steps_(0),
      has_last_dynamic_frame_(false),
      cuda_runtime_handle_(0) {}

InputConvV1::~InputConvV1() {
#if NPGR_ENABLE_CUDA
    if (cuda_runtime_handle_ != 0) {
        DestroyInputConvV1CudaRuntime(cuda_runtime_handle_);
        cuda_runtime_handle_ = 0;
    }
#endif
}

bool InputConvV1::LoadStimulusFile() {
    stimulus_file_frames_.clear();
    stimulus_file_frame_count_ = 0;
    if (stimulus_file_path_.empty()) {
        return false;
    }
    std::ifstream input(stimulus_file_path_.c_str(), std::ios::binary);
    if (!input.is_open()) {
        return false;
    }
    StimulusFileHeader header;
    if (!ReadStimulusHeader(&input, &header)) {
        return false;
    }
    if (header.width <= 0 || header.height <= 0 || header.channels <= 0 || header.length <= 0) {
        return false;
    }
    width_ = header.width;
    height_ = header.height;
    channels_ = header.channels;
    const std::size_t frame_bytes =
        static_cast<std::size_t>(header.width) *
        static_cast<std::size_t>(header.height) *
        static_cast<std::size_t>(header.channels);
    stimulus_file_frames_.resize(frame_bytes * static_cast<std::size_t>(header.length));
    input.read(reinterpret_cast<char*>(stimulus_file_frames_.data()),
               static_cast<std::streamsize>(stimulus_file_frames_.size()));
    if (!input.good()) {
        stimulus_file_frames_.clear();
        return false;
    }
    stimulus_file_frame_count_ = header.length;
    return true;
}

void InputConvV1::Initialize(const InputConvDescription& description, Simulation* simulation) {
    description_ = description;
    this->setTimestepSize(description.update_timestep > 0 ? description.update_timestep : 1);
    this->setQueueIndex(description.queue_index);

    width_ = GetAnyOrDefault<int>(description.ModelParameter, "width", 8);
    height_ = GetAnyOrDefault<int>(description.ModelParameter, "height", 8);
    channels_ = GetAnyOrDefault<int>(description.ModelParameter, "channels", 1);
    stimulus_mode_ = GetAnyOrDefault<std::string>(description.ModelParameter, "stimulus_mode", std::string("bar"));
    speed_ = GetAnyOrDefault<float>(description.ModelParameter, "speed", 1.5f);
    stimulus_temporal_speed_ = GetAnyOrDefault<float>(description.ModelParameter, "stimulus_temporal_speed", 0.08f);
    grating_direction_deg_ = GetAnyOrDefault<float>(description.ModelParameter, "grating_direction_deg", 0.0f);
    plaid_direction_a_deg_ = GetAnyOrDefault<float>(description.ModelParameter, "plaid_direction_a_deg", 0.0f);
    plaid_direction_b_deg_ = GetAnyOrDefault<float>(description.ModelParameter, "plaid_direction_b_deg", 120.0f);
    spatial_frequency_ = GetAnyOrDefault<float>(description.ModelParameter, "spatial_frequency", 2.2f);
    bar_width_ = GetAnyOrDefault<int>(description.ModelParameter, "bar_width", 2);
    motion_period_steps_ = GetAnyOrDefault<int>(description.ModelParameter, "motion_period_steps", 32);
    frame_hold_steps_ = GetAnyOrDefault<int>(description.ModelParameter, "frame_hold_steps", 1);
    stimulus_file_path_ = GetAnyOrDefault<std::string>(description.ModelParameter, "stimulus_file_path", std::string());
    input_frame_missing_policy_ =
        GetAnyOrDefault<std::string>(description.ModelParameter, "input_frame_missing_policy", std::string("zero"));
    input_frame_max_lag_steps_ = GetAnyOrDefault<int>(description.ModelParameter, "input_frame_max_lag_steps", 0);
    if (bar_width_ <= 0) {
        bar_width_ = 1;
    }
    if (motion_period_steps_ <= 0) {
        motion_period_steps_ = 1;
    }
    if (frame_hold_steps_ <= 0) {
        frame_hold_steps_ = 1;
    }
    if (input_frame_max_lag_steps_ < 0) {
        input_frame_max_lag_steps_ = 0;
    }
    if (width_ <= 0) {
        width_ = 1;
    }
    if (height_ <= 0) {
        height_ = 1;
    }
    if (channels_ <= 0) {
        channels_ = 1;
    }

    if (!stimulus_file_path_.empty()) {
        this->LoadStimulusFile();
    }
    host_stimulus_.assign(static_cast<std::size_t>(width_ * height_ * channels_), 0);
    last_dynamic_frame_.assign(host_stimulus_.size(), 0);
    has_last_dynamic_frame_ = false;
    (void)simulation;

#if NPGR_ENABLE_CUDA
    InputConvV1CudaConfig config;
    config.width = width_;
    config.height = height_;
    config.channels = channels_;
    config.speed = speed_;
    const float* device_rates = 0;
    CreateInputConvV1CudaRuntime(config, &cuda_runtime_handle_, &output_count_, &device_rates);
#else
#endif
}

bool InputConvV1::UsesDynamicFrameInput() const {
    return true;
}

InputConvFrameRequest InputConvV1::GetFrameRequest(int time_step,
                                                   int inputconv_index,
                                                   int source_camera_index) const {
    InputConvFrameRequest request;
    request.time_step = time_step;
    request.inputconv_index = inputconv_index;
    request.source_camera_index = source_camera_index;
    request.width = width_;
    request.height = height_;
    request.channels = channels_;
    request.max_lag_steps = input_frame_max_lag_steps_;
    request.preferred_pixel_format =
        channels_ == 1 ? InputConvPixelFormat::UInt8Gray : InputConvPixelFormat::UInt8RGB;
    return request;
}

bool InputConvV1::CopyUInt8FrameToStimulus(const InputConvFrame& frame, std::string* reason) {
    if (frame.width != width_ || frame.height != height_) {
        if (reason != 0) {
            *reason = "InputConvV1 dynamic frame size does not match the model";
        }
        return false;
    }
    if (frame.pixel_format != InputConvPixelFormat::UInt8Gray &&
        frame.pixel_format != InputConvPixelFormat::UInt8RGB &&
        frame.pixel_format != InputConvPixelFormat::UInt8BGR) {
        if (reason != 0) {
            *reason = "InputConvV1 currently accepts UInt8Gray, UInt8RGB, or UInt8BGR frames";
        }
        return false;
    }

    const int source_channels =
        frame.pixel_format == InputConvPixelFormat::UInt8Gray ? 1 : 3;
    const std::size_t expected_bytes =
        static_cast<std::size_t>(width_) *
        static_cast<std::size_t>(height_) *
        static_cast<std::size_t>(source_channels);
    if (frame.bytes.size() != expected_bytes) {
        if (reason != 0) {
            *reason = "InputConvV1 dynamic frame byte count does not match the pixel format";
        }
        return false;
    }
    if (channels_ != 1 && channels_ != 3) {
        if (reason != 0) {
            *reason = "InputConvV1 dynamic frame conversion supports one or three model channels";
        }
        return false;
    }

    if (host_stimulus_.size() !=
        static_cast<std::size_t>(width_ * height_ * channels_)) {
        host_stimulus_.assign(static_cast<std::size_t>(width_ * height_ * channels_), 0);
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    if (source_channels == 1 && channels_ == 1) {
        std::copy(frame.bytes.begin(), frame.bytes.end(), host_stimulus_.begin());
    } else if (source_channels == 3 && channels_ == 3) {
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const unsigned char c0 = frame.bytes[pixel * 3 + 0];
            const unsigned char c1 = frame.bytes[pixel * 3 + 1];
            const unsigned char c2 = frame.bytes[pixel * 3 + 2];
            host_stimulus_[pixel] =
                frame.pixel_format == InputConvPixelFormat::UInt8BGR ? c2 : c0;
            host_stimulus_[pixel_count + pixel] = c1;
            host_stimulus_[pixel_count * 2 + pixel] =
                frame.pixel_format == InputConvPixelFormat::UInt8BGR ? c0 : c2;
        }
    } else if (source_channels == 1 && channels_ == 3) {
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const unsigned char value = frame.bytes[pixel];
            host_stimulus_[pixel] = value;
            host_stimulus_[pixel_count + pixel] = value;
            host_stimulus_[pixel_count * 2 + pixel] = value;
        }
    } else {
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            const unsigned char c0 = frame.bytes[pixel * 3 + 0];
            const unsigned char c1 = frame.bytes[pixel * 3 + 1];
            const unsigned char c2 = frame.bytes[pixel * 3 + 2];
            const unsigned char red =
                frame.pixel_format == InputConvPixelFormat::UInt8BGR ? c2 : c0;
            const unsigned char green = c1;
            const unsigned char blue =
                frame.pixel_format == InputConvPixelFormat::UInt8BGR ? c0 : c2;
            host_stimulus_[pixel] = static_cast<unsigned char>(
                (static_cast<int>(red) + static_cast<int>(green) + static_cast<int>(blue)) / 3);
        }
    }

    last_dynamic_frame_ = host_stimulus_;
    has_last_dynamic_frame_ = true;
    return true;
}

bool InputConvV1::AcceptFrame(const InputConvFrame& frame, std::string* reason) {
    return this->CopyUInt8FrameToStimulus(frame, reason);
}

void InputConvV1::ApplyMissingFramePolicy(int time_step) {
    if (input_frame_missing_policy_ == "hold_last" && has_last_dynamic_frame_) {
        host_stimulus_ = last_dynamic_frame_;
        return;
    }
    if (input_frame_missing_policy_ == "fallback") {
        this->GenerateStimulusFrame(time_step);
        return;
    }
    std::fill(host_stimulus_.begin(), host_stimulus_.end(), static_cast<unsigned char>(0));
}

void InputConvV1::GenerateStimulusFrame(int time_step) {
    if (host_stimulus_.empty()) {
        return;
    }
    std::fill(host_stimulus_.begin(), host_stimulus_.end(), static_cast<unsigned char>(0));
    const int timestep_size = this->getTimestepSize() > 0 ? this->getTimestepSize() : 1;
    const int update_index = time_step / timestep_size;
    if (!stimulus_file_frames_.empty() && stimulus_file_frame_count_ > 0) {
        const int frame_index = (update_index / frame_hold_steps_) % stimulus_file_frame_count_;
        const std::size_t frame_bytes = static_cast<std::size_t>(width_ * height_ * channels_);
        const std::size_t offset = static_cast<std::size_t>(frame_index) * frame_bytes;
        std::copy(stimulus_file_frames_.begin() + offset,
                  stimulus_file_frames_.begin() + offset + frame_bytes,
                  host_stimulus_.begin());
        return;
    }
    const int frame_index = update_index / frame_hold_steps_;
    if (stimulus_mode_ == "grating") {
        AddGratingFrame(&host_stimulus_,
                        width_,
                        height_,
                        frame_index,
                        stimulus_temporal_speed_,
                        grating_direction_deg_,
                        spatial_frequency_);
        return;
    }
    if (stimulus_mode_ == "plaid") {
        AddPlaidFrame(&host_stimulus_,
                      width_,
                      height_,
                      frame_index,
                      stimulus_temporal_speed_,
                      plaid_direction_a_deg_,
                      plaid_direction_b_deg_,
                      spatial_frequency_);
        return;
    }
    const int phase = motion_period_steps_ > 0 ? (update_index % motion_period_steps_) : 0;
    const int x_center = (phase * width_) / motion_period_steps_;
    for (int channel = 0; channel < channels_; ++channel) {
        for (int y = 0; y < height_; ++y) {
            for (int dx = 0; dx < bar_width_; ++dx) {
                const int x = (x_center + dx) % width_;
                const int index = channel * width_ * height_ + y * width_ + x;
                host_stimulus_[static_cast<std::size_t>(index)] = static_cast<unsigned char>(255);
            }
        }
    }
}

void InputConvV1::Update(int time, Simulation* simulation) {
#if NPGR_ENABLE_CUDA
    if (cuda_runtime_handle_ == 0) {
        return;
    }
    if (simulation != 0 && simulation->HasInputConvFrameSourceBinding(this->getInputConvIndex())) {
        InputConvFrame frame;
        std::string reason;
        if (simulation->LoadInputConvFrame(this->getInputConvIndex(), time, &frame, &reason)) {
            if (!this->AcceptFrame(frame, &reason)) {
                this->ApplyMissingFramePolicy(time);
            }
        } else {
            this->ApplyMissingFramePolicy(time);
        }
    } else {
        this->GenerateStimulusFrame(time);
    }
    StepInputConvV1CudaRuntime(cuda_runtime_handle_, host_stimulus_.data(), speed_);
#else
    (void)simulation;
    (void)time;
#endif
}

void InputConvV1::Reset(Simulation* simulation) {
    (void)simulation;
    if (!host_stimulus_.empty()) {
        std::fill(host_stimulus_.begin(), host_stimulus_.end(), static_cast<unsigned char>(0));
    }
    if (!last_dynamic_frame_.empty()) {
        std::fill(last_dynamic_frame_.begin(), last_dynamic_frame_.end(), static_cast<unsigned char>(0));
    }
    has_last_dynamic_frame_ = false;
}

const float* InputConvV1::GetCudaDeviceRates() const {
#if NPGR_ENABLE_CUDA
    if (cuda_runtime_handle_ == 0) {
        return 0;
    }
    return GetInputConvV1CudaDeviceRates(cuda_runtime_handle_);
#else
    return 0;
#endif
}

int InputConvV1::GetOutputCount() const {
    return output_count_;
}

const float* InputConvV1::GetDeviceOutputBuffer() const {
    return this->GetCudaDeviceRates();
}

bool InputConvV1::HasDeviceOutputBuffer() const {
    return this->HasCudaRuntime() && this->GetDeviceOutputBuffer() != 0 && output_count_ > 0;
}

bool InputConvV1::HasCudaRuntime() const {
    return cuda_runtime_handle_ != 0;
}

bool InputConvV1::ExportMonitorOutput(std::vector<float>* output, std::string* reason) const {
    if (output == 0) {
        if (reason != 0) {
            *reason = "InputConvV1 monitor output destination must not be null";
        }
        return false;
    }
    output->clear();
#if NPGR_ENABLE_CUDA
    if (cuda_runtime_handle_ == 0 || output_count_ <= 0) {
        return true;
    }
    const float* device_rates = this->GetCudaDeviceRates();
    if (device_rates == 0) {
        return true;
    }
    output->resize(static_cast<std::size_t>(output_count_));
    const cudaError_t err = cudaMemcpy(output->data(),
                                       device_rates,
                                       sizeof(float) * static_cast<std::size_t>(output_count_),
                                       cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        output->clear();
        if (reason != 0) {
            *reason = cudaGetErrorString(err);
        }
        return false;
    }
    return true;
#else
    return true;
#endif
}

bool InputConvV1::ExportMonitorInput(std::vector<float>* input, std::string* reason) const {
    (void)reason;
    if (input == 0) {
        return false;
    }
    input->clear();
    input->reserve(host_stimulus_.size());
    for (std::size_t index = 0; index < host_stimulus_.size(); ++index) {
        input->push_back(static_cast<float>(host_stimulus_[index]));
    }
    return true;
}

bool InputConvV1::ExportMonitorState(std::vector<npgr::DebugInputConvStateRecord>* state,
                                     std::string* reason) const {
    (void)reason;
    if (state == 0) {
        return false;
    }
    state->clear();
    const struct {
        const char* name;
        float value;
    } values[] = {
        {"width", static_cast<float>(width_)},
        {"height", static_cast<float>(height_)},
        {"channels", static_cast<float>(channels_)},
        {"output_count", static_cast<float>(output_count_)},
        {"speed", speed_},
        {"has_last_dynamic_frame", has_last_dynamic_frame_ ? 1.0f : 0.0f},
    };
    for (int index = 0; index < 6; ++index) {
        npgr::DebugInputConvStateRecord record;
        record.field_name = values[index].name;
        record.state_index = index;
        record.value = values[index].value;
        state->push_back(record);
    }
    return true;
}
