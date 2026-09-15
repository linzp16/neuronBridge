#include "../source_file_realtime_v1_async/communication/inc/ZMQAsyncInputConvFrameDriver.h"

#include <cstdint>
#include <cstring>

namespace {

struct ZMQAsyncInputConvFrameHeader {
    uint32_t magic;
    uint32_t version;
    int32_t time_step;
    int32_t source_camera_index;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t pixel_format;
    uint32_t payload_bytes;
};

static const uint32_t kAsyncInputConvFrameMagic = 0x41494346u;
static const uint32_t kAsyncInputConvFrameVersion = 1u;
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

}  // namespace

ZMQAsyncInputConvFrameDriver::ZMQAsyncInputConvFrameDriver(const std::string& subscribe_address,
                                                           unsigned short subscribe_port,
                                                           const std::string& topic,
                                                           int max_payload_bytes,
                                                           int max_buffered_frames_per_camera)
    : AsyncInputConvFrameDriver(max_buffered_frames_per_camera),
      subscriber_socket_(new ZmqSocket(ZmqSocket::Mode::SUBSCRIBE,
                                       subscribe_address,
                                       static_cast<int>(subscribe_port))),
      topic_(topic),
      max_payload_bytes_(max_payload_bytes > 0 ? max_payload_bytes : kDefaultMaxPayloadBytes) {
    subscriber_socket_->setLinger(0);
    subscriber_socket_->subscribe(topic_);
}

ZMQAsyncInputConvFrameDriver::~ZMQAsyncInputConvFrameDriver() {
    Stop();
    delete subscriber_socket_;
    subscriber_socket_ = 0;
}

bool ZMQAsyncInputConvFrameDriver::PollFrame(InputConvFrame* frame, std::string* reason) {
    if (frame == 0) {
        if (reason != 0) {
            *reason = "InputConv frame destination must not be null";
        }
        return false;
    }

    std::string topic = subscriber_socket_->receiveString(true);
    if (topic.empty()) {
        return false;
    }

    ZMQAsyncInputConvFrameHeader header;
    std::memset(&header, 0, sizeof(header));
    const int header_bytes = subscriber_socket_->receiveBuffer(&header, sizeof(header));
    if (header_bytes != static_cast<int>(sizeof(header))) {
        if (reason != 0) {
            *reason = "Failed to receive async InputConv frame header";
        }
        return false;
    }

    if (topic != topic_ ||
        header.magic != kAsyncInputConvFrameMagic ||
        header.version != kAsyncInputConvFrameVersion) {
        if (header.payload_bytes > 0 &&
            header.payload_bytes <= static_cast<uint32_t>(max_payload_bytes_)) {
            std::vector<unsigned char> discard(header.payload_bytes);
            subscriber_socket_->receiveBuffer(discard.data(),
                                              static_cast<int>(discard.size()));
        }
        return false;
    }
    if (header.payload_bytes > static_cast<uint32_t>(max_payload_bytes_)) {
        if (reason != 0) {
            *reason = "Async InputConv frame payload is too large";
        }
        return false;
    }

    frame->time_step = header.time_step;
    frame->source_camera_index = header.source_camera_index;
    frame->width = header.width;
    frame->height = header.height;
    frame->channels = header.channels;
    frame->pixel_format = PixelFormatFromWire(header.pixel_format);
    frame->bytes.assign(header.payload_bytes, 0);
    if (header.payload_bytes == 0) {
        return true;
    }

    const int payload_bytes =
        subscriber_socket_->receiveBuffer(frame->bytes.data(),
                                          static_cast<int>(frame->bytes.size()));
    if (payload_bytes != static_cast<int>(frame->bytes.size())) {
        frame->bytes.clear();
        if (reason != 0) {
            *reason = "Incomplete async InputConv frame payload";
        }
        return false;
    }
    return true;
}
