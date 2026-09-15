#ifndef ASYNC_INPUT_CONV_FRAME_DRIVER_H
#define ASYNC_INPUT_CONV_FRAME_DRIVER_H

#include "../source_file_realtime_v1_async/InputConv/inc/InputConvFrameQueue.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

class AsyncInputConvFrameDriver : public InputConvFrameDriver {
public:
    explicit AsyncInputConvFrameDriver(int max_buffered_frames_per_camera = 8);
    ~AsyncInputConvFrameDriver() override;

    bool LoadFrame(const InputConvFrameRequest& request,
                   InputConvFrame* frame,
                   std::string* reason) override;
    void Clear() override;
    InputConvFrameSourceStats GetStats() const override;

    void Start();
    void Stop();
    bool running() const;
    bool GetLastError(std::string* reason) const;

protected:
    // PollFrame is owned by the concrete source. It should return quickly or
    // use non-blocking IO so Stop() can join the worker thread promptly.
    virtual bool PollFrame(InputConvFrame* frame, std::string* reason) = 0;

private:
    void WorkerLoop();
    void SetLastError(const std::string& reason);
    std::string LastErrorString() const;

    InputConvFrameQueue queue_;
    std::thread worker_;
    std::atomic<bool> running_;
    std::atomic<int> received_frames_;
    std::atomic<int> consumed_frames_;
    std::atomic<int> failed_requests_;
    std::atomic<int> dropped_frames_;
    std::atomic<int> last_received_time_step_;
    std::atomic<int> last_consumed_time_step_;
    std::atomic<int> last_source_camera_index_;
    mutable std::mutex error_mutex_;
    std::string last_error_;
};

#endif
