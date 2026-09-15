#ifndef INPUT_CONV_FRAME_DRIVER_H
#define INPUT_CONV_FRAME_DRIVER_H

#include <string>
#include <vector>

enum class InputConvPixelFormat {
    UInt8Gray,
    UInt8RGB,
    UInt8BGR,
    Float32Gray,
    Float32HWC,
    Float32CHW,
};

struct InputConvFrame {
    int time_step = 0;
    int source_camera_index = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    InputConvPixelFormat pixel_format = InputConvPixelFormat::UInt8Gray;
    std::vector<unsigned char> bytes;
};

struct InputConvFrameRequest {
    int time_step = 0;
    int inputconv_index = -1;
    int source_camera_index = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    int max_lag_steps = 0;
    InputConvPixelFormat preferred_pixel_format = InputConvPixelFormat::UInt8Gray;
};

struct InputConvFrameSourceBinding {
    int source_index = -1;
    int source_camera_index = 0;
};

struct InputConvFrameSourceStats {
    int requested_frames = 0;
    int received_frames = 0;
    int consumed_frames = 0;
    int failed_requests = 0;
    int dropped_frames = 0;
    int queued_frames = 0;
    int last_received_time_step = -1;
    int last_consumed_time_step = -1;
    int last_source_camera_index = -1;
    bool running = false;
    std::string last_error;
};

class InputConvFrameDriver {
public:
    virtual ~InputConvFrameDriver() {}

    // Returns one frame that satisfies the request. Implementations may drop
    // stale frames internally, but should keep future frames for later steps.
    virtual bool LoadFrame(const InputConvFrameRequest& request,
                           InputConvFrame* frame,
                           std::string* reason) = 0;

    // Clears transient buffered input, when the driver owns such a buffer.
    virtual void Clear() {}

    virtual InputConvFrameSourceStats GetStats() const {
        return InputConvFrameSourceStats();
    }
};

#endif
