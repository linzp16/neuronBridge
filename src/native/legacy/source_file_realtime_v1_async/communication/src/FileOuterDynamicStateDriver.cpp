#include "../source_file_realtime_v1_async/communication/inc/FileOuterDynamicStateDriver.h"

#include <filesystem>
#include <stdexcept>

FileOuterDynamicStateDriver::FileOuterDynamicStateDriver(const char* file_name)
    : filename_(file_name), handler_(NULL), header_written_(false) {
    std::filesystem::path file_path(file_name);
    std::filesystem::path parent_path = file_path.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }

    this->handler_ = fopen(file_name, "wt");
    if (this->handler_ == NULL) {
        throw std::runtime_error(std::string("Failed to open outer dynamic state file: ") + file_name);
    }

}

FileOuterDynamicStateDriver::~FileOuterDynamicStateDriver() {
    if (this->handler_ != NULL) {
        fclose(this->handler_);
        this->handler_ = NULL;
    }
}

void FileOuterDynamicStateDriver::WriteJointState(
    int time_step,
    float base_timestep_ms,
    const OuterDynamicJointState& state) {
#pragma omp critical (FileOuterDynamicStateDriver)
    {
        if (this->handler_ == NULL) {
            throw std::runtime_error(std::string("Outer dynamic state file handle is null: ") + this->filename_);
        }
        const std::size_t joint_count = state.q.size();
        if (state.qv.size() != joint_count || state.qdd.size() != joint_count ||
            state.q_des.size() != joint_count || state.qv_des.size() != joint_count ||
            state.tau_total.size() != joint_count) {
            throw std::runtime_error("Outer dynamic state fields have inconsistent joint counts.");
        }
        if (!this->header_written_) {
            fprintf(this->handler_, "time_s");
            const char* prefixes[] = {"q", "qv", "qdd", "qdes", "qvdes", "tau"};
            for (const char* prefix : prefixes) {
                for (std::size_t joint = 0; joint < joint_count; ++joint) {
                    fprintf(this->handler_, "\t%s%zu", prefix, joint + 1);
                }
            }
            fprintf(this->handler_, "\n");
            this->header_written_ = true;
        }
        const double time_seconds = static_cast<double>(time_step) * static_cast<double>(base_timestep_ms) * 0.001;
        fprintf(this->handler_, "%.9f", time_seconds);
        const std::vector<double>* fields[] = {
            &state.q, &state.qv, &state.qdd, &state.q_des, &state.qv_des, &state.tau_total};
        for (const std::vector<double>* field : fields) {
            for (double value : *field) {
                fprintf(this->handler_, "\t%.12f", value);
            }
        }
        fprintf(this->handler_, "\n");
    }
}

void FileOuterDynamicStateDriver::FlushBuffers() {
    if (this->handler_ != NULL) {
        fflush(this->handler_);
    }
}
