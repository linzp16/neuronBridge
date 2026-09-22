#ifndef NPGR_STREAMING_BUILD_NBNET_READER_H
#define NPGR_STREAMING_BUILD_NBNET_READER_H

#include "source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "source_file_realtime_v1_async/Simulation/inc/InputConvDescription.h"
#include "streaming_build/ConnectionRecordSource.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <vector>

namespace npgr {
namespace streaming {

struct StreamingBuildOptions {
    std::uint64_t memory_budget_bytes = 128ULL * 1024ULL * 1024ULL;
    bool use_mmap = true;
    bool verify_checksum = true;
};

struct StreamingBuildStats {
    std::string mode = "streaming_file_cpp";
    std::string path;
    std::uint64_t file_bytes = 0;
    std::uint64_t metadata_bytes = 0;
    std::uint64_t neuron_count = 0;
    std::uint64_t connection_count = 0;
    std::uint64_t batch_records = 0;
    std::uint64_t memory_budget_bytes = 0;
    bool used_mmap = false;
    bool checksum_verified = false;
    double parse_seconds = 0.0;
    double total_build_seconds = 0.0;
    std::string runtime_build_path = "streaming_direct";
};

struct NbnetMetadata {
    std::list<NeuronLayerDescription> layers;
    std::list<LearningRuleDescription> learning_rules;
    std::list<OuterDynamicDescription> outer_dynamics;
    std::list<OuterDynamicConnectionDescription> outer_dynamic_connections;
    std::list<InputConvDescription> input_convs;
    std::uint64_t neuron_count = 0;
    bool has_dense_layers = false;
};

class NbnetReader : public ConnectionRecordSource {
public:
    NbnetReader(const std::string& path, const StreamingBuildOptions& options);
    ~NbnetReader();

    NbnetReader(NbnetReader&&) noexcept;
    NbnetReader& operator=(NbnetReader&&) noexcept;
    NbnetReader(const NbnetReader&) = delete;
    NbnetReader& operator=(const NbnetReader&) = delete;

    const NbnetMetadata& metadata() const;
    std::uint64_t connection_count() const;
    std::uint64_t file_size() const;
    std::uint64_t metadata_size() const;
    std::size_t batch_records() const;
    bool used_mmap() const;
    bool checksum_verified() const;
    const std::string& path() const;

    void ForEachConnectionBatch(const BatchCallback& callback) const override;
    std::vector<ConnectionRecordV1> ReadConnectionBatch(std::uint64_t first_record,
                                                        std::size_t max_records) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace streaming
}  // namespace npgr

#endif
