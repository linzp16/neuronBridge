#ifndef INPUT_CONV_V1_H
#define INPUT_CONV_V1_H

#include "../source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"

#include <string>
#include <vector>

class InputConvV1 : public InputConvModel {
public:
    InputConvV1();
    explicit InputConvV1(int timestep_size);
    virtual ~InputConvV1();

    virtual void Initialize(const InputConvDescription& description, Simulation* simulation);
    virtual void Update(int time, Simulation* simulation);
    virtual void Reset(Simulation* simulation);
    virtual const float* GetDeviceOutputBuffer() const;
    virtual int GetOutputCount() const;
    virtual bool HasDeviceOutputBuffer() const;
    virtual bool ExportMonitorOutput(std::vector<float>* output, std::string* reason) const;
    virtual bool ExportMonitorInput(std::vector<float>* input, std::string* reason) const;
    virtual bool ExportMonitorState(std::vector<npgr::DebugInputConvStateRecord>* state,
                                    std::string* reason) const;
    virtual bool UsesDynamicFrameInput() const;
    virtual InputConvFrameRequest GetFrameRequest(int time_step,
                                                  int inputconv_index,
                                                  int source_camera_index) const;
    virtual bool AcceptFrame(const InputConvFrame& frame, std::string* reason);
    virtual void ApplyMissingFramePolicy(int time_step);
    const float* GetCudaDeviceRates() const;
    bool HasCudaRuntime() const;

private:
    void GenerateStimulusFrame(int time_step);
    bool LoadStimulusFile();
    bool CopyUInt8FrameToStimulus(const InputConvFrame& frame, std::string* reason);

    InputConvDescription description_;
    int width_;
    int height_;
    int channels_;
    int output_count_;
    std::string stimulus_mode_;
    float speed_;
    float stimulus_temporal_speed_;
    float grating_direction_deg_;
    float plaid_direction_a_deg_;
    float plaid_direction_b_deg_;
    float spatial_frequency_;
    int bar_width_;
    int motion_period_steps_;
    int frame_hold_steps_;
    std::string stimulus_file_path_;
    int stimulus_file_frame_count_;
    std::string input_frame_missing_policy_;
    int input_frame_max_lag_steps_;
    bool has_last_dynamic_frame_;
    std::vector<unsigned char> host_stimulus_;
    std::vector<unsigned char> last_dynamic_frame_;
    std::vector<unsigned char> stimulus_file_frames_;
    void* cuda_runtime_handle_;
};

#endif
