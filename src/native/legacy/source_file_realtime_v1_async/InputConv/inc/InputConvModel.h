#ifndef INPUT_CONV_MODEL_H
#define INPUT_CONV_MODEL_H

#include "../source_file_realtime_v1_async/Simulation/inc/InputConvDescription.h"
#include "../source_file_realtime_v1_async/InputConv/inc/InputConvFrameDriver.h"

#include "debug_monitor/DebugMonitorTypes.h"

#include <string>
#include <vector>

class Simulation;

class InputConvModel {
public:
    InputConvModel();
    explicit InputConvModel(int timestep_size);
    virtual ~InputConvModel();

    virtual void Initialize(const InputConvDescription& description, Simulation* simulation) = 0;
    virtual void Update(int time, Simulation* simulation) = 0;
    virtual void Reset(Simulation* simulation);

    // Returns a device-resident output buffer when the model can publish one.
    // Simulation-level bindings use this instead of downcasting to a concrete
    // InputConv implementation.
    virtual const float* GetDeviceOutputBuffer() const;
    virtual int GetOutputCount() const;
    virtual bool HasDeviceOutputBuffer() const;
    // Exports a full monitor snapshot of the latest InputConv output.
    virtual bool ExportMonitorOutput(std::vector<float>* output, std::string* reason) const;
    // Exports a full monitor snapshot of the latest InputConv input, when available.
    virtual bool ExportMonitorInput(std::vector<float>* input, std::string* reason) const;
    // Exports implementation-specific InputConv state records.
    virtual bool ExportMonitorState(std::vector<npgr::DebugInputConvStateRecord>* state,
                                    std::string* reason) const;
    // Dynamic frame input is optional. A model opts in by overriding these
    // methods and accepting frames requested through Simulation.
    virtual bool UsesDynamicFrameInput() const;
    virtual InputConvFrameRequest GetFrameRequest(int time_step,
                                                  int inputconv_index,
                                                  int source_camera_index) const;
    virtual bool AcceptFrame(const InputConvFrame& frame, std::string* reason);
    virtual void ApplyMissingFramePolicy(int time_step);

    int getTimestepSize() const;
    void setTimestepSize(int timestep_size);

    int getQueueIndex() const;
    void setQueueIndex(int queue_index);

    int getInputConvIndex() const;
    void setInputConvIndex(int inputconv_index);

protected:
    int timestep_size_;
    int queue_index_;
    int inputconv_index_;
};

#endif
