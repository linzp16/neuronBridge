#ifndef ZMQ_INPUT_CONV_FRAME_DRIVER_H
#define ZMQ_INPUT_CONV_FRAME_DRIVER_H

#include "../source_file_realtime_v1_async/InputConv/inc/InputConvFrameDriver.h"
#include "../source_file_realtime_v1_async/communication/inc/ZmqSocket.h"

#include <atomic>
#include <mutex>
#include <string>

class ZMQInputConvFrameDriver : public InputConvFrameDriver {
public:
    ZMQInputConvFrameDriver(const std::string& server_address,
                            unsigned short tcp_port,
                            int max_payload_bytes);
    ~ZMQInputConvFrameDriver() override;

    bool LoadFrame(const InputConvFrameRequest& request,
                   InputConvFrame* frame,
                   std::string* reason) override;
    InputConvFrameSourceStats GetStats() const override;

private:
    void SetLastError(const std::string& reason);
    std::string LastErrorString() const;

    ZmqSocket* socket_;
    int max_payload_bytes_;
    std::atomic<int> requested_frames_;
    std::atomic<int> received_frames_;
    std::atomic<int> failed_requests_;
    std::atomic<int> last_received_time_step_;
    std::atomic<int> last_source_camera_index_;
    mutable std::mutex error_mutex_;
    std::string last_error_;
};

#endif
