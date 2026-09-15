#include "../source_file_realtime_v1_async/InputConv/inc/InputConvFrameQueue.h"

#include <algorithm>

InputConvFrameQueue::InputConvFrameQueue(int max_buffered_frames_per_camera)
    : max_buffered_frames_per_camera_(max_buffered_frames_per_camera > 0
                                          ? max_buffered_frames_per_camera
                                          : 8) {}

int InputConvFrameQueue::TrimCameraLocked(int source_camera_index) {
    if (max_buffered_frames_per_camera_ <= 0) {
        return 0;
    }
    int camera_frame_count = 0;
    for (std::deque<InputConvFrame>::reverse_iterator it = frames_.rbegin();
         it != frames_.rend();
         ++it) {
        if (it->source_camera_index != source_camera_index) {
            continue;
        }
        ++camera_frame_count;
        if (camera_frame_count <= max_buffered_frames_per_camera_) {
            continue;
        }
        const int keep_after_time_step = it->time_step;
        std::deque<InputConvFrame> kept;
        int dropped = 0;
        for (std::size_t index = 0; index < frames_.size(); ++index) {
            const InputConvFrame& frame = frames_[index];
            if (frame.source_camera_index == source_camera_index &&
                frame.time_step <= keep_after_time_step) {
                ++dropped;
                continue;
            }
            kept.push_back(frame);
        }
        frames_.swap(kept);
        return dropped;
    }
    return 0;
}

int InputConvFrameQueue::PushFrame(const InputConvFrame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    frames_.push_back(frame);
    return TrimCameraLocked(frame.source_camera_index);
}

int InputConvFrameQueue::PushFrames(const std::vector<InputConvFrame>& frames) {
    std::lock_guard<std::mutex> lock(mutex_);
    int dropped = 0;
    for (std::size_t index = 0; index < frames.size(); ++index) {
        frames_.push_back(frames[index]);
        dropped += TrimCameraLocked(frames[index].source_camera_index);
    }
    return dropped;
}

bool InputConvFrameQueue::PopLatestForTime(const InputConvFrameRequest& request,
                                           InputConvFrame* frame,
                                           std::string* reason) {
    if (frame == 0) {
        if (reason != 0) {
            *reason = "InputConv frame destination must not be null";
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const bool has_lag_limit = request.max_lag_steps > 0;
    const int min_time_step = request.time_step - request.max_lag_steps;
    int selected_index = -1;
    int selected_time_step = 0;

    for (std::size_t index = 0; index < frames_.size(); ++index) {
        const InputConvFrame& candidate = frames_[index];
        if (candidate.source_camera_index != request.source_camera_index) {
            continue;
        }
        if (candidate.time_step > request.time_step) {
            continue;
        }
        if (has_lag_limit && candidate.time_step < min_time_step) {
            continue;
        }
        if (selected_index < 0 || candidate.time_step >= selected_time_step) {
            selected_index = static_cast<int>(index);
            selected_time_step = candidate.time_step;
        }
    }

    if (selected_index < 0) {
        if (reason != 0) {
            *reason = "No InputConv frame is available for the requested time";
        }
        return false;
    }

    *frame = frames_[static_cast<std::size_t>(selected_index)];

    // Consume the chosen frame and older frames from the same camera. Future
    // frames, and frames from other cameras, remain available for later calls.
    std::deque<InputConvFrame> kept;
    for (std::size_t index = 0; index < frames_.size(); ++index) {
        const InputConvFrame& candidate = frames_[index];
        if (candidate.source_camera_index == request.source_camera_index &&
            candidate.time_step <= selected_time_step) {
            continue;
        }
        kept.push_back(candidate);
    }
    frames_.swap(kept);
    return true;
}

void InputConvFrameQueue::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    frames_.clear();
}

void InputConvFrameQueue::setMaxBufferedFramesPerCamera(int max_buffered_frames_per_camera) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_buffered_frames_per_camera_ =
        max_buffered_frames_per_camera > 0 ? max_buffered_frames_per_camera : 8;
}

int InputConvFrameQueue::maxBufferedFramesPerCamera() const {
    return max_buffered_frames_per_camera_;
}

int InputConvFrameQueue::Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(frames_.size());
}

QueuedInputConvFrameDriver::QueuedInputConvFrameDriver(int max_buffered_frames_per_camera)
    : queue_(max_buffered_frames_per_camera),
      received_frames_(0),
      consumed_frames_(0),
      dropped_frames_(0),
      last_received_time_step_(-1),
      last_consumed_time_step_(-1),
      last_source_camera_index_(-1) {}

bool QueuedInputConvFrameDriver::LoadFrame(const InputConvFrameRequest& request,
                                           InputConvFrame* frame,
                                           std::string* reason) {
    const bool ok = queue_.PopLatestForTime(request, frame, reason);
    if (ok && frame != 0) {
        ++consumed_frames_;
        last_consumed_time_step_ = frame->time_step;
        last_source_camera_index_ = frame->source_camera_index;
    }
    return ok;
}

void QueuedInputConvFrameDriver::Clear() {
    queue_.Clear();
}

InputConvFrameSourceStats QueuedInputConvFrameDriver::GetStats() const {
    InputConvFrameSourceStats stats;
    stats.received_frames = received_frames_;
    stats.consumed_frames = consumed_frames_;
    stats.dropped_frames = dropped_frames_;
    stats.queued_frames = queue_.Size();
    stats.last_received_time_step = last_received_time_step_;
    stats.last_consumed_time_step = last_consumed_time_step_;
    stats.last_source_camera_index = last_source_camera_index_;
    return stats;
}

void QueuedInputConvFrameDriver::PushFrame(const InputConvFrame& frame) {
    ++received_frames_;
    last_received_time_step_ = frame.time_step;
    last_source_camera_index_ = frame.source_camera_index;
    dropped_frames_ += queue_.PushFrame(frame);
}

void QueuedInputConvFrameDriver::PushFrames(const std::vector<InputConvFrame>& frames) {
    for (std::size_t index = 0; index < frames.size(); ++index) {
        ++received_frames_;
        last_received_time_step_ = frames[index].time_step;
        last_source_camera_index_ = frames[index].source_camera_index;
    }
    dropped_frames_ += queue_.PushFrames(frames);
}
