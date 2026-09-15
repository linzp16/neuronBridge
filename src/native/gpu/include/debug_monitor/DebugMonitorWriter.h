#ifndef NPGR_DEBUG_MONITOR_WRITER_H
#define NPGR_DEBUG_MONITOR_WRITER_H

#include "debug_monitor/DebugMonitorTypes.h"

#include <fstream>
#include <string>

namespace npgr {

class DebugMonitorWriter {
public:
    DebugMonitorWriter();
    ~DebugMonitorWriter();

    bool Initialize(const DebugMonitorConfig& config, std::string* reason = nullptr);
    bool AppendFrame(const DebugMonitorFrame& frame, std::string* reason = nullptr);
    bool Flush(std::string* reason = nullptr);
    bool initialized() const;

private:
    // Flushes and closes every CSV stream before the writer is reused for a new output directory.
    void CloseStreams();
    bool EnsureOpen(std::string* reason);
    bool WriteMeta(std::string* reason);
    bool WriteHeaders(std::string* reason);

    DebugMonitorConfig config_;
    bool initialized_;
    bool headers_written_;
    std::ofstream neuron_state_out_;
    std::ofstream spikes_out_;
    std::ofstream pending_channels_out_;
    std::ofstream inputconv_outputs_out_;
    std::ofstream inputconv_inputs_out_;
    std::ofstream inputconv_state_out_;
    std::ofstream outer_dynamic_state_out_;
    std::ofstream weights_out_;
};

}  // namespace npgr

#endif
