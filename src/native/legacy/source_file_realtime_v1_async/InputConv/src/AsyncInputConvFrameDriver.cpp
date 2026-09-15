#include "../source_file_realtime_v1_async/InputConv/inc/AsyncInputConvFrameDriver.h"

#include <chrono>

AsyncInputConvFrameDriver::AsyncInputConvFrameDriver(int max_buffered_frames_per_camera)
    : queue_(max_buffered_frames_per_camera),
      running_(false),
      received_frames_(0),
      consumed_frames_(0),
      failed_requests_(0),
      dropped_frames_(0),
      last_received_time_step_(-1),
      last_consumed_time_step_(-1),
      last_source_camera_index_(-1) {}

AsyncInputConvFrameDriver::~AsyncInputConvFrameDriver() {
    Stop();
}

bool AsyncInputConvFrameDriver::LoadFrame(const InputConvFrameRequest& request,
                                          InputConvFrame* frame,
                                          std::string* reason) {
    std::string queue_reason;
    if (queue_.PopLatestForTime(request, frame, &queue_reason)) {
        ++consumed_frames_;
        if (frame != 0) {
            last_consumed_time_step_.store(frame->time_step);
            last_source_camera_index_.store(frame->source_camera_index);
        }
        return true;
    }
    const std::string last_error = LastErrorString();
    if (!last_error.empty()) {
        if (reason != 0) {
            *reason = "Async InputConv frame driver failed: " + last_error;
        }
        return false;
    }
    if (reason != 0) {
        *reason = queue_reason;
    }
    return false;
}

void AsyncInputConvFrameDriver::Clear() {
    queue_.Clear();
}

void AsyncInputConvFrameDriver::Start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;
    }
    worker_ = std::thread(&AsyncInputConvFrameDriver::WorkerLoop, this);
}

void AsyncInputConvFrameDriver::Stop() {
    const bool was_running = running_.exchange(false);
    if (worker_.joinable()) {
        worker_.join();
    }
    (void)was_running;
}

bool AsyncInputConvFrameDriver::running() const {
    return running_.load();
}

bool AsyncInputConvFrameDriver::GetLastError(std::string* reason) const {
    const std::string last_error = LastErrorString();
    if (reason != 0) {
        *reason = last_error;
    }
    return !last_error.empty();
}

InputConvFrameSourceStats AsyncInputConvFrameDriver::GetStats() const {
    InputConvFrameSourceStats stats;
    stats.received_frames = received_frames_.load();
    stats.consumed_frames = consumed_frames_.load();
    stats.failed_requests = failed_requests_.load();
    stats.dropped_frames = dropped_frames_.load();
    stats.queued_frames = queue_.Size();
    stats.last_received_time_step = last_received_time_step_.load();
    stats.last_consumed_time_step = last_consumed_time_step_.load();
    stats.last_source_camera_index = last_source_camera_index_.load();
    stats.running = running_.load();
    stats.last_error = LastErrorString();
    return stats;
}

void AsyncInputConvFrameDriver::WorkerLoop() {
    while (running_.load()) {
        InputConvFrame frame;
        std::string reason;
        if (PollFrame(&frame, &reason)) {
            ++received_frames_;
            last_received_time_step_.store(frame.time_step);
            last_source_camera_index_.store(frame.source_camera_index);
            dropped_frames_.fetch_add(queue_.PushFrame(frame));
            SetLastError(std::string());
            continue;
        }
        if (!reason.empty()) {
            failed_requests_.fetch_add(1);
            SetLastError(reason);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AsyncInputConvFrameDriver::SetLastError(const std::string& reason) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = reason;
}

std::string AsyncInputConvFrameDriver::LastErrorString() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}
