/*
 * FileOuterDynamicStateDriver.h
 *
 * File-backed OuterDynamicStateDriver implementation. It writes outer-dynamics
 * joint state samples to a text file for later plotting or validation.
 */
#ifndef FILEOUTERDYNAMICSTATEDRIVER_H
#define FILEOUTERDYNAMICSTATEDRIVER_H

#include "../source_file_realtime_v1_async/communication/inc/OuterDynamicStateDriver.h"

#include <cstdio>
#include <string>

class FileOuterDynamicStateDriver : public OuterDynamicStateDriver {
public:
    // Opens a file-backed state writer.
    explicit FileOuterDynamicStateDriver(const char* file_name);
    // Flushes and closes the backing file.
    virtual ~FileOuterDynamicStateDriver();

    // Writes one timestamped joint-state sample.
    virtual void WriteJointState(
        int time_step,
        float base_timestep_ms,
        const OuterDynamicJointState& state) override;

    // Flushes pending file output.
    virtual void FlushBuffers() override;

private:
    // Output file path.
    std::string filename_;
    // C file handle used by the writer.
    FILE* handler_;
    bool header_written_;
};

#endif
