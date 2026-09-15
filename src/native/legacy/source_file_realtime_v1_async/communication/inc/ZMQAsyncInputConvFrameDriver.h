#ifndef ZMQ_ASYNC_INPUT_CONV_FRAME_DRIVER_H
#define ZMQ_ASYNC_INPUT_CONV_FRAME_DRIVER_H

#include "../source_file_realtime_v1_async/InputConv/inc/AsyncInputConvFrameDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/ZmqSocket.h"

#include <string>

class ZMQAsyncInputConvFrameDriver : public AsyncInputConvFrameDriver {
public:
    ZMQAsyncInputConvFrameDriver(const std::string& subscribe_address,
                                 unsigned short subscribe_port,
                                 const std::string& topic,
                                 int max_payload_bytes,
                                 int max_buffered_frames_per_camera = 8);
    ~ZMQAsyncInputConvFrameDriver() override;

protected:
    bool PollFrame(InputConvFrame* frame, std::string* reason) override;

private:
    ZmqSocket* subscriber_socket_;
    std::string topic_;
    int max_payload_bytes_;
};

#endif
