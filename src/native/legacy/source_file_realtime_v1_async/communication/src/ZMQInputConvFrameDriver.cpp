#include "../source_file_realtime_v1_async/communication/inc/ZMQInputConvFrameDriver.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct ZMQInputConvFrameRequestHeader {
    uint32_t magic;
    int32_t time_step;
    int32_t inputconv_index;
    int32_t source_camera_index;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t preferred_pixel_format;
    int32_t max_lag_steps;
};

struct ZMQInputConvFrameHeader {
    uint32_t magic;
    int32_t time_step;
    int32_t source_camera_index;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t pixel_format;
    uint32_t payload_bytes;
};

static const uint32_t kInputConvFrameRequestMagic = 0x49434652u;
static const uint32_t kInputConvFrameResponseMagic = 0x49434650u;
static const int kDefaultMaxPayloadBytes = 64 * 1024 * 1024;

InputConvPixelFormat PixelFormatFromWire(int32_t value) {
    switch (value) {
    case 0:
        return InputConvPixelFormat::UInt8Gray;
    case 1:
        return InputConvPixelFormat::UInt8RGB;
    case 2:
        return InputConvPixelFormat::UInt8BGR;
    case 3:
        return InputConvPixelFormat::Float32Gray;
    case 4:
        return InputConvPixelFormat::Float32HWC;
    case 5:
        return InputConvPixelFormat::Float32CHW;
    default:
        return InputConvPixelFormat::UInt8Gray;
    }
}

int32_t PixelFormatToWire(InputConvPixelFormat format) {
    switch (format) {
    case InputConvPixelFormat::UInt8Gray:
        return 0;
    case InputConvPixelFormat::UInt8RGB:
        return 1;
    case InputConvPixelFormat::UInt8BGR:
        return 2;
    case InputConvPixelFormat::Float32Gray:
        return 3;
    case InputConvPixelFormat::Float32HWC:
        return 4;
    case InputConvPixelFormat::Float32CHW:
        return 5;
    default:
        return 0;
    }
}

}  // namespace

ZMQInputConvFrameDriver::ZMQInputConvFrameDriver(const std::string& server_address,
                                                 unsigned short tcp_port,
                                                 int max_payload_bytes)
    : socket_(new ZmqSocket(ZmqSocket::Mode::REQUEST, server_address, static_cast<int>(tcp_port))),
      max_payload_bytes_(max_payload_bytes > 0 ? max_payload_bytes : kDefaultMaxPayloadBytes),
      requested_frames_(0),
      received_frames_(0),
      failed_requests_(0),
      last_received_time_step_(-1),
      last_source_camera_index_(-1) {}

ZMQInputConvFrameDriver::~ZMQInputConvFrameDriver() {
    delete socket_;
}

bool ZMQInputConvFrameDriver::LoadFrame(const InputConvFrameRequest& request,
                                        InputConvFrame* frame,
                                        std::string* reason) {
    if (frame == 0) {
        if (reason != 0) {
            *reason = "InputConv frame destination must not be null";
        }
        return false;
    }

    ZMQInputConvFrameRequestHeader request_header;
    std::memset(&request_header, 0, sizeof(request_header));
    request_header.magic = kInputConvFrameRequestMagic;
    request_header.time_step = request.time_step;
    request_header.inputconv_index = request.inputconv_index;
    request_header.source_camera_index = request.source_camera_index;
    request_header.width = request.width;
    request_header.height = request.height;
    request_header.channels = request.channels;
    request_header.preferred_pixel_format = PixelFormatToWire(request.preferred_pixel_format);
    request_header.max_lag_steps = request.max_lag_steps;

    requested_frames_.fetch_add(1);
    if (!socket_->sendBuffer(&request_header, sizeof(request_header))) {
        failed_requests_.fetch_add(1);
        SetLastError("Failed to send InputConv frame request");
        if (reason != 0) {
            *reason = "Failed to send InputConv frame request";
        }
        return false;
    }

    ZMQInputConvFrameHeader response_header;
    const int header_bytes = socket_->receiveBuffer(&response_header, sizeof(response_header));
    if (header_bytes != static_cast<int>(sizeof(response_header))) {
        failed_requests_.fetch_add(1);
        SetLastError("Failed to receive InputConv frame response header");
        if (reason != 0) {
            *reason = "Failed to receive InputConv frame response header";
        }
        return false;
    }
    if (response_header.magic != kInputConvFrameResponseMagic) {
        if (response_header.payload_bytes > 0 &&
            response_header.payload_bytes <= static_cast<uint32_t>(max_payload_bytes_)) {
            std::vector<unsigned char> discard(response_header.payload_bytes);
            socket_->receiveBuffer(discard.data(), static_cast<int>(discard.size()));
        }
        failed_requests_.fetch_add(1);
        SetLastError("Invalid InputConv frame response magic");
        if (reason != 0) {
            *reason = "Invalid InputConv frame response magic";
        }
        return false;
    }
    if (response_header.payload_bytes > static_cast<uint32_t>(max_payload_bytes_)) {
        failed_requests_.fetch_add(1);
        SetLastError("InputConv frame payload is too large");
        if (reason != 0) {
            *reason = "InputConv frame payload is too large";
        }
        return false;
    }

    frame->time_step = response_header.time_step;
    frame->source_camera_index = response_header.source_camera_index;
    frame->width = response_header.width;
    frame->height = response_header.height;
    frame->channels = response_header.channels;
    frame->pixel_format = PixelFormatFromWire(response_header.pixel_format);
    frame->bytes.assign(response_header.payload_bytes, 0);
    if (response_header.payload_bytes == 0) {
        received_frames_.fetch_add(1);
        last_received_time_step_.store(frame->time_step);
        last_source_camera_index_.store(frame->source_camera_index);
        SetLastError(std::string());
        return true;
    }
    const int payload_bytes =
        socket_->receiveBuffer(frame->bytes.data(), static_cast<int>(frame->bytes.size()));
    if (payload_bytes != static_cast<int>(frame->bytes.size())) {
        frame->bytes.clear();
        failed_requests_.fetch_add(1);
        SetLastError("Incomplete InputConv frame payload");
        if (reason != 0) {
            *reason = "Incomplete InputConv frame payload";
        }
        return false;
    }
    received_frames_.fetch_add(1);
    last_received_time_step_.store(frame->time_step);
    last_source_camera_index_.store(frame->source_camera_index);
    SetLastError(std::string());
    return true;
}

InputConvFrameSourceStats ZMQInputConvFrameDriver::GetStats() const {
    InputConvFrameSourceStats stats;
    stats.requested_frames = requested_frames_.load();
    stats.received_frames = received_frames_.load();
    stats.failed_requests = failed_requests_.load();
    stats.last_received_time_step = last_received_time_step_.load();
    stats.last_source_camera_index = last_source_camera_index_.load();
    stats.last_error = LastErrorString();
    return stats;
}

void ZMQInputConvFrameDriver::SetLastError(const std::string& reason) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = reason;
}

std::string ZMQInputConvFrameDriver::LastErrorString() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}
