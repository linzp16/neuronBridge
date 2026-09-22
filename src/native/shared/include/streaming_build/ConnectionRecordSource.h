#ifndef NPGR_STREAMING_BUILD_CONNECTION_RECORD_SOURCE_H
#define NPGR_STREAMING_BUILD_CONNECTION_RECORD_SOURCE_H

#include <cstdint>
#include <functional>
#include <vector>

namespace npgr {
namespace streaming {

struct ConnectionRecordV1 {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    std::int32_t synapse_type = 0;
    float weight = 0.0f;
    float max_weight = 0.0f;
    std::uint32_t delay = 0;
    std::int32_t synapse_rule = -1;
    std::int32_t trigger_rule = -1;
};

static_assert(sizeof(ConnectionRecordV1) == 32, "nbnet v1 connection record must be 32 bytes");

class ConnectionRecordSource {
public:
    using BatchCallback = std::function<void(const std::vector<ConnectionRecordV1>&)>;
    virtual ~ConnectionRecordSource() = default;
    virtual void ForEachConnectionBatch(const BatchCallback& callback) const = 0;
};

}  // namespace streaming
}  // namespace npgr

#endif
