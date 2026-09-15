#ifndef INPUT_CONV_FRAME_QUEUE_H
#define INPUT_CONV_FRAME_QUEUE_H

#include "../source_file_realtime_v1_async/InputConv/inc/InputConvFrameDriver.h"

#include <deque>
#include <mutex>
#include <vector>

class InputConvFrameQueue {
public:
    explicit InputConvFrameQueue(int max_buffered_frames_per_camera = 8);
    int PushFrame(const InputConvFrame& frame);
    int PushFrames(const std::vector<InputConvFrame>& frames);
    bool PopLatestForTime(const InputConvFrameRequest& request,
                          InputConvFrame* frame,
                          std::string* reason);
    void Clear();
    void setMaxBufferedFramesPerCamera(int max_buffered_frames_per_camera);
    int maxBufferedFramesPerCamera() const;
    int Size() const;

private:
    int TrimCameraLocked(int source_camera_index);

    std::deque<InputConvFrame> frames_;
    int max_buffered_frames_per_camera_;
    mutable std::mutex mutex_;
};

class QueuedInputConvFrameDriver : public InputConvFrameDriver {
public:
    explicit QueuedInputConvFrameDriver(int max_buffered_frames_per_camera = 8);
    bool LoadFrame(const InputConvFrameRequest& request,
                   InputConvFrame* frame,
                   std::string* reason) override;
    void Clear() override;
    InputConvFrameSourceStats GetStats() const override;

    void PushFrame(const InputConvFrame& frame);
    void PushFrames(const std::vector<InputConvFrame>& frames);

private:
    InputConvFrameQueue queue_;
    int received_frames_;
    int consumed_frames_;
    int dropped_frames_;
    int last_received_time_step_;
    int last_consumed_time_step_;
    int last_source_camera_index_;
};

#endif
