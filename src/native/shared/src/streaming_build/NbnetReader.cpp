#include "streaming_build/NbnetReader.h"

#include <boost/any.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace npgr {
namespace streaming {
namespace {

constexpr std::size_t kHeaderSize = 68;
constexpr std::size_t kRecordSize = 32;
constexpr std::uint16_t kMajorVersion = 1;
constexpr std::uint32_t kSha256Flag = 1;
constexpr std::uint64_t kMaxMetadataSize = 64ULL * 1024ULL * 1024ULL;
constexpr char kMagic[8] = {'N', 'B', 'N', 'E', 'T', '0', '1', '\0'};

std::uint16_t ReadU16(const unsigned char* data) {
    return static_cast<std::uint16_t>(data[0]) |
           (static_cast<std::uint16_t>(data[1]) << 8U);
}

std::uint32_t ReadU32(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8U) |
           (static_cast<std::uint32_t>(data[2]) << 16U) |
           (static_cast<std::uint32_t>(data[3]) << 24U);
}

std::uint64_t ReadU64(const unsigned char* data) {
    std::uint64_t value = 0;
    for (int index = 7; index >= 0; --index) {
        value = (value << 8U) | data[index];
    }
    return value;
}

float ReadFloat32(const unsigned char* data) {
    const std::uint32_t bits = ReadU32(data);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

class Sha256 {
public:
    Sha256() { Reset(); }

    void Update(const unsigned char* data, std::size_t size) {
        for (std::size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == 64) {
                Transform(buffer_.data());
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    std::array<unsigned char, 32> Final() {
        std::array<unsigned char, 32> digest{};
        std::uint64_t total_bits = bit_count_ + static_cast<std::uint64_t>(buffer_size_) * 8ULL;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) {
                buffer_[buffer_size_++] = 0;
            }
            Transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) {
            buffer_[buffer_size_++] = 0;
        }
        for (int shift = 7; shift >= 0; --shift) {
            buffer_[buffer_size_++] = static_cast<unsigned char>((total_bits >> (shift * 8)) & 0xffU);
        }
        Transform(buffer_.data());
        for (std::size_t index = 0; index < state_.size(); ++index) {
            digest[index * 4] = static_cast<unsigned char>((state_[index] >> 24U) & 0xffU);
            digest[index * 4 + 1] = static_cast<unsigned char>((state_[index] >> 16U) & 0xffU);
            digest[index * 4 + 2] = static_cast<unsigned char>((state_[index] >> 8U) & 0xffU);
            digest[index * 4 + 3] = static_cast<unsigned char>(state_[index] & 0xffU);
        }
        return digest;
    }

private:
    static std::uint32_t RotateRight(std::uint32_t value, std::uint32_t bits) {
        return (value >> bits) | (value << (32U - bits));
    }

    void Reset() {
        state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                  0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
        buffer_.fill(0);
        buffer_size_ = 0;
        bit_count_ = 0;
    }

    void Transform(const unsigned char* block) {
        static constexpr std::array<std::uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const unsigned char* value = block + index * 4;
            words[index] = (static_cast<std::uint32_t>(value[0]) << 24U) |
                           (static_cast<std::uint32_t>(value[1]) << 16U) |
                           (static_cast<std::uint32_t>(value[2]) << 8U) |
                           static_cast<std::uint32_t>(value[3]);
        }
        for (std::size_t index = 16; index < 64; ++index) {
            const std::uint32_t s0 = RotateRight(words[index - 15], 7) ^ RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3U);
            const std::uint32_t s1 = RotateRight(words[index - 2], 17) ^ RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10U);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < 64; ++index) {
            const std::uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + sum1 + choice + constants[index] + words[index];
            const std::uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<unsigned char, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t bit_count_ = 0;
};

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { Close(); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    void Open(const std::string& path) {
#if defined(_WIN32)
        file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("cannot open nbnet file: " + path);
        }
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart < 0) {
            throw std::runtime_error("cannot determine nbnet file size: " + path);
        }
        size_ = static_cast<std::uint64_t>(size.QuadPart);
        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            throw std::runtime_error("cannot create nbnet file mapping: " + path);
        }
        data_ = static_cast<const unsigned char*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (data_ == nullptr) {
            throw std::runtime_error("cannot map nbnet file: " + path);
        }
#else
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open nbnet file: " + path);
        }
        struct stat status {};
        if (fstat(fd_, &status) != 0 || status.st_size < 0) {
            throw std::runtime_error("cannot determine nbnet file size: " + path);
        }
        size_ = static_cast<std::uint64_t>(status.st_size);
        void* address = mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
        if (address == MAP_FAILED) {
            throw std::runtime_error("cannot map nbnet file: " + path);
        }
        data_ = static_cast<const unsigned char*>(address);
#endif
    }

    const unsigned char* data() const { return data_; }
    std::uint64_t size() const { return size_; }

private:
    void Close() noexcept {
#if defined(_WIN32)
        if (data_ != nullptr) {
            UnmapViewOfFile(data_);
        }
        if (mapping_ != nullptr) {
            CloseHandle(mapping_);
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_);
        }
        file_ = INVALID_HANDLE_VALUE;
        mapping_ = nullptr;
#else
        if (data_ != nullptr) {
            munmap(const_cast<unsigned char*>(data_), static_cast<std::size_t>(size_));
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        fd_ = -1;
#endif
        data_ = nullptr;
        size_ = 0;
    }

    const unsigned char* data_ = nullptr;
    std::uint64_t size_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

using boost::property_tree::ptree;

template <typename T>
std::vector<T> ReadVector(const ptree& node) {
    std::vector<T> values;
    values.reserve(node.size());
    for (const auto& item : node) {
        values.push_back(item.second.get_value<T>());
    }
    return values;
}

std::vector<std::vector<int> > ReadIntMatrix(const ptree& node) {
    std::vector<std::vector<int> > result;
    result.reserve(node.size());
    for (const auto& row : node) {
        result.push_back(ReadVector<int>(row.second));
    }
    return result;
}

std::vector<std::vector<std::vector<int> > > ReadIntTensor3(const ptree& node) {
    std::vector<std::vector<std::vector<int> > > result;
    result.reserve(node.size());
    for (const auto& matrix : node) {
        result.push_back(ReadIntMatrix(matrix.second));
    }
    return result;
}

template <typename T>
T GetOr(const ptree& node, const std::string& key, const T& fallback) {
    return node.get<T>(key, fallback);
}

boost::any ParseTaggedParameter(const ptree& node) {
    const std::string kind = node.get<std::string>("__neuronbridge_parameter__");
    const ptree& value = node.get_child("value");
    if (kind == "bool") return boost::any(value.get_value<bool>());
    if (kind == "int" || kind == "int32") return boost::any(value.get_value<int>());
    if (kind == "float" || kind == "float32") return boost::any(value.get_value<float>());
    if (kind == "float64") return boost::any(value.get_value<double>());
    if (kind == "str") return boost::any(value.get_value<std::string>());
    if (kind == "bool_list") return boost::any(ReadVector<bool>(value));
    if (kind == "int_list" || kind == "int32_list") return boost::any(ReadVector<int>(value));
    if (kind == "float_list" || kind == "float32_list") return boost::any(ReadVector<float>(value));
    if (kind == "float64_list") return boost::any(ReadVector<double>(value));
    if (kind == "str_list") return boost::any(ReadVector<std::string>(value));
    if (kind == "float32_array3") {
        const std::vector<float> values = ReadVector<float>(value);
        if (values.size() != 3) throw std::runtime_error("float32_array3 requires 3 values");
        return boost::any(std::array<float, 3>{values[0], values[1], values[2]});
    }
    if (kind == "float32_array4") {
        const std::vector<float> values = ReadVector<float>(value);
        if (values.size() != 4) throw std::runtime_error("float32_array4 requires 4 values");
        return boost::any(std::array<float, 4>{values[0], values[1], values[2], values[3]});
    }
    if (kind == "float32_array5") {
        const std::vector<float> values = ReadVector<float>(value);
        if (values.size() != 5) throw std::runtime_error("float32_array5 requires 5 values");
        return boost::any(std::array<float, 5>{values[0], values[1], values[2], values[3], values[4]});
    }
    throw std::runtime_error("unsupported nbnet parameter kind: " + kind);
}

boost::any ParseLegacyParameter(const ptree& node) {
    if (!node.empty()) {
        std::vector<std::string> values = ReadVector<std::string>(node);
        return boost::any(values);
    }
    const std::string text = node.get_value<std::string>();
    if (text == "true") return boost::any(true);
    if (text == "false") return boost::any(false);
    try {
        std::size_t consumed = 0;
        const int integer = std::stoi(text, &consumed);
        if (consumed == text.size()) return boost::any(integer);
    } catch (...) {
    }
    try {
        std::size_t consumed = 0;
        const float number = std::stof(text, &consumed);
        if (consumed == text.size()) return boost::any(number);
    } catch (...) {
    }
    return boost::any(text);
}

std::map<std::string, boost::any> ParseParameters(const ptree& node) {
    std::map<std::string, boost::any> result;
    for (const auto& item : node) {
        if (item.second.get_optional<std::string>("__neuronbridge_parameter__")) {
            result[item.first] = ParseTaggedParameter(item.second);
        } else {
            result[item.first] = ParseLegacyParameter(item.second);
        }
    }
    return result;
}

const ptree* OptionalChild(const ptree& node, const std::string& key) {
    const auto child = node.get_child_optional(key);
    return child ? &child.get() : nullptr;
}

OuterDynamicFeedbackProductEncoding ParseProductEncoding(const ptree* node) {
    OuterDynamicFeedbackProductEncoding result;
    if (node == nullptr) return result;
    if (const ptree* values = OptionalChild(*node, "neuron_indices_by_joint")) {
        result.neuron_indices_by_joint = ReadIntMatrix(*values);
    }
    if (const ptree* bins = OptionalChild(*node, "bins")) {
        result.bins = ReadVector<int>(*bins);
    }
    return result;
}

OuterDynamicFeedbackSingleEncoding ParseSingleEncoding(const ptree* node) {
    OuterDynamicFeedbackSingleEncoding result;
    if (node == nullptr) return result;
    if (const ptree* values = OptionalChild(*node, "neuron_indices_by_joint_variable")) {
        result.neuron_indices_by_joint_variable = ReadIntTensor3(*values);
    }
    if (const ptree* bins = OptionalChild(*node, "bins")) {
        result.bins = ReadVector<int>(*bins);
    }
    return result;
}

NbnetMetadata ParseMetadata(const std::string& json) {
    ptree root;
    std::istringstream input(json);
    boost::property_tree::read_json(input, root);
    NbnetMetadata metadata;
    const ptree& layers = root.get_child("layers");
    for (const auto& item : layers) {
        const ptree& data = item.second;
        NeuronLayerDescription layer;
        layer.ModelName = data.get<std::string>("model");
        layer.numberofneuron = data.get<int>("count");
        layer.update_timestep = data.get<int>("update_timestep", 1);
        layer.isMonitored = data.get<bool>("monitored", false);
        layer.isOutput = data.get<bool>("output", false);
        layer.isCommunicationInput = data.get<bool>("communication_input", false);
        if (const ptree* parameters = OptionalChild(data, "parameters")) {
            layer.NeuronParameter = ParseParameters(*parameters);
            const auto dense = layer.NeuronParameter.find("dense_subnetwork_name");
            if (dense != layer.NeuronParameter.end() && dense->second.type() == typeid(std::string) &&
                !boost::any_cast<std::string>(dense->second).empty()) {
                metadata.has_dense_layers = true;
            }
        }
        if (layer.numberofneuron <= 0 || layer.update_timestep <= 0) {
            throw std::runtime_error("invalid nbnet layer count or timestep");
        }
        metadata.neuron_count += static_cast<std::uint64_t>(layer.numberofneuron);
        metadata.layers.push_back(std::move(layer));
    }

    if (const ptree* rules = OptionalChild(root, "learning_rules")) {
        for (const auto& item : *rules) {
            LearningRuleDescription rule;
            rule.RuleName = item.second.get<std::string>("name");
            if (const ptree* parameters = OptionalChild(item.second, "parameters")) {
                rule.RuleParameter = ParseParameters(*parameters);
            }
            metadata.learning_rules.push_back(std::move(rule));
        }
    }

    if (const ptree* dynamics = OptionalChild(root, "outer_dynamics")) {
        for (const auto& item : *dynamics) {
            const ptree& data = item.second;
            OuterDynamicDescription outer;
            outer.ModelName = data.get<std::string>("model");
            outer.name = data.get<std::string>("name", "");
            outer.update_timestep = data.get<int>("update_timestep", 1);
            outer.communication_interval = data.get<int>("communication_interval", 1);
            outer.queue_index = data.get<int>("queue_index", 0);
            if (const ptree* parameters = OptionalChild(data, "parameters")) outer.ModelParameter = ParseParameters(*parameters);
            if (const ptree* value = OptionalChild(data, "gc_neuron_indices_by_joint")) outer.gc_neuron_indices_by_joint = ReadIntMatrix(*value);
            if (const ptree* value = OptionalChild(data, "mf_neuron_indices_by_joint")) outer.mf_neuron_indices_by_joint = ReadIntMatrix(*value);
            if (const ptree* value = OptionalChild(data, "cf_positive_neuron_indices_by_joint")) outer.cf_positive_neuron_indices_by_joint = ReadIntMatrix(*value);
            if (const ptree* value = OptionalChild(data, "cf_negative_neuron_indices_by_joint")) outer.cf_negative_neuron_indices_by_joint = ReadIntMatrix(*value);
            if (const ptree* value = OptionalChild(data, "dcn_positive_neuron_indices_by_joint")) outer.dcn_positive_neuron_indices_by_joint = ReadIntMatrix(*value);
            if (const ptree* value = OptionalChild(data, "dcn_negative_neuron_indices_by_joint")) outer.dcn_negative_neuron_indices_by_joint = ReadIntMatrix(*value);
            outer.state_feedback_product = ParseProductEncoding(OptionalChild(data, "state_feedback_product"));
            outer.state_feedback_single = ParseSingleEncoding(OptionalChild(data, "state_feedback_single"));
            outer.error_feedback_product = ParseProductEncoding(OptionalChild(data, "error_feedback_product"));
            outer.error_feedback_single = ParseSingleEncoding(OptionalChild(data, "error_feedback_single"));
            metadata.outer_dynamics.push_back(std::move(outer));
        }
    }

    if (const ptree* routes = OptionalChild(root, "outer_dynamic_connections")) {
        for (const auto& item : *routes) {
            const ptree& data = item.second;
            OuterDynamicConnectionDescription route;
            route.SourceNeuron = ReadVector<int>(data.get_child("source"));
            route.TargetOuterDynamic = ReadVector<int>(data.get_child("target_outer_dynamic"));
            route.TargetJoint = ReadVector<int>(data.get_child("target_joint"));
            route.Type = ReadVector<int>(data.get_child("synapse_type"));
            route.Weight = ReadVector<float>(data.get_child("weight"));
            route.Delay = ReadVector<int>(data.get_child("delay"));
            metadata.outer_dynamic_connections.push_back(std::move(route));
        }
    }

    if (const ptree* input_convs = OptionalChild(root, "input_convs")) {
        for (const auto& item : *input_convs) {
            const ptree& data = item.second;
            InputConvDescription input_conv;
            input_conv.ModelName = data.get<std::string>("model");
            if (const ptree* parameters = OptionalChild(data, "parameters")) input_conv.ModelParameter = ParseParameters(*parameters);
            input_conv.update_timestep = data.get<int>("update_timestep", 1);
            input_conv.queue_index = data.get<int>("queue_index", 0);
            const std::string target = data.get<std::string>("output_target", "main_network");
            if (target == "main_network") input_conv.output_target = InputConvOutputTarget::MainNetwork;
            else if (target == "dense_subnetwork") input_conv.output_target = InputConvOutputTarget::DenseSubnetwork;
            else throw std::runtime_error("invalid InputConv output_target in nbnet metadata");
            input_conv.target_dense_subnetwork_name = data.get<std::string>("target_dense_subnetwork_name", "");
            if (const ptree* value = OptionalChild(data, "output_source_indices")) input_conv.output_source_indices = ReadVector<int>(*value);
            if (const ptree* value = OptionalChild(data, "output_target_neuron_ids")) input_conv.output_target_neuron_ids = ReadVector<int>(*value);
            input_conv.output_pending_channel = data.get<int>("output_pending_channel", 0);
            input_conv.output_scale = data.get<float>("output_scale", 1.0f);
            if (const ptree* value = OptionalChild(data, "output_scales")) input_conv.output_scales = ReadVector<float>(*value);
            input_conv.output_overwrite = data.get<bool>("output_overwrite", true);
            metadata.input_convs.push_back(std::move(input_conv));
        }
    }
    return metadata;
}

ConnectionRecordV1 DecodeRecord(const unsigned char* data) {
    ConnectionRecordV1 record;
    record.source = ReadU32(data);
    record.target = ReadU32(data + 4);
    record.synapse_type = static_cast<std::int32_t>(ReadU32(data + 8));
    record.weight = ReadFloat32(data + 12);
    record.max_weight = ReadFloat32(data + 16);
    record.delay = ReadU32(data + 20);
    record.synapse_rule = static_cast<std::int32_t>(ReadU32(data + 24));
    record.trigger_rule = static_cast<std::int32_t>(ReadU32(data + 28));
    return record;
}

}  // namespace

struct NbnetReader::Impl {
    std::string path;
    StreamingBuildOptions options;
    std::uint64_t file_size = 0;
    std::uint64_t metadata_size = 0;
    std::uint64_t connection_count = 0;
    std::uint64_t records_offset = 0;
    std::size_t batch_records = 1;
    bool checksum_verified = false;
    NbnetMetadata metadata;
    std::unique_ptr<MappedFile> mapping;
};

NbnetReader::NbnetReader(const std::string& path, const StreamingBuildOptions& options)
    : impl_(new Impl()) {
    impl_->path = path;
    impl_->options = options;
    if (options.memory_budget_bytes < kRecordSize) {
        throw std::invalid_argument("streaming memory budget must be at least one connection record");
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open nbnet file: " + path);
    const std::streamoff end = input.tellg();
    if (end < static_cast<std::streamoff>(kHeaderSize)) throw std::runtime_error("truncated nbnet header");
    impl_->file_size = static_cast<std::uint64_t>(end);
    input.seekg(0);
    std::array<unsigned char, kHeaderSize> header{};
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (!input) throw std::runtime_error("truncated nbnet header");
    if (!std::equal(header.begin(), header.begin() + 8, reinterpret_cast<const unsigned char*>(kMagic))) {
        throw std::runtime_error("invalid nbnet magic");
    }
    const std::uint16_t major = ReadU16(header.data() + 8);
    const std::uint32_t flags = ReadU32(header.data() + 12);
    const std::uint32_t record_size = ReadU32(header.data() + 16);
    impl_->metadata_size = ReadU64(header.data() + 20);
    impl_->connection_count = ReadU64(header.data() + 28);
    if (major != kMajorVersion) throw std::runtime_error("unsupported nbnet major version");
    if (record_size != kRecordSize) throw std::runtime_error("unsupported nbnet connection record size");
    if (impl_->metadata_size > kMaxMetadataSize) throw std::runtime_error("nbnet metadata exceeds 64 MiB");
    if (impl_->connection_count > (std::numeric_limits<std::uint64_t>::max() - kHeaderSize - impl_->metadata_size) / kRecordSize) {
        throw std::runtime_error("nbnet connection table size overflows uint64");
    }
    const std::uint64_t expected_size = kHeaderSize + impl_->metadata_size + impl_->connection_count * kRecordSize;
    if (expected_size != impl_->file_size) throw std::runtime_error("nbnet file size mismatch");
    impl_->records_offset = kHeaderSize + impl_->metadata_size;
    std::string metadata_json(static_cast<std::size_t>(impl_->metadata_size), '\0');
    input.read(metadata_json.data(), static_cast<std::streamsize>(metadata_json.size()));
    if (!input) throw std::runtime_error("truncated nbnet metadata");

    if (options.verify_checksum) {
        if ((flags & kSha256Flag) == 0) throw std::runtime_error("nbnet SHA-256 checksum is missing");
        Sha256 sha;
        sha.Update(reinterpret_cast<const unsigned char*>(metadata_json.data()), metadata_json.size());
        std::array<unsigned char, 1024 * 1024> buffer{};
        while (input) {
            input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = input.gcount();
            if (count > 0) sha.Update(buffer.data(), static_cast<std::size_t>(count));
        }
        const std::array<unsigned char, 32> digest = sha.Final();
        if (!std::equal(digest.begin(), digest.end(), header.begin() + 36)) {
            throw std::runtime_error("nbnet checksum mismatch");
        }
        impl_->checksum_verified = true;
    }
    impl_->metadata = ParseMetadata(metadata_json);
    const std::uint64_t conservative_record_bytes = 64;
    impl_->batch_records = static_cast<std::size_t>(std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(1000000, options.memory_budget_bytes / conservative_record_bytes)));
    if (options.use_mmap) {
        impl_->mapping.reset(new MappedFile());
        impl_->mapping->Open(path);
    }
}

NbnetReader::~NbnetReader() = default;
NbnetReader::NbnetReader(NbnetReader&&) noexcept = default;
NbnetReader& NbnetReader::operator=(NbnetReader&&) noexcept = default;

const NbnetMetadata& NbnetReader::metadata() const { return impl_->metadata; }
std::uint64_t NbnetReader::connection_count() const { return impl_->connection_count; }
std::uint64_t NbnetReader::file_size() const { return impl_->file_size; }
std::uint64_t NbnetReader::metadata_size() const { return impl_->metadata_size; }
std::size_t NbnetReader::batch_records() const { return impl_->batch_records; }
bool NbnetReader::used_mmap() const { return impl_->mapping != nullptr; }
bool NbnetReader::checksum_verified() const { return impl_->checksum_verified; }
const std::string& NbnetReader::path() const { return impl_->path; }

std::vector<ConnectionRecordV1> NbnetReader::ReadConnectionBatch(
    std::uint64_t first_record,
    std::size_t max_records) const {
    if (first_record > impl_->connection_count) {
        throw std::out_of_range("nbnet connection batch offset is out of range");
    }
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(
        impl_->connection_count - first_record, max_records));
    std::vector<ConnectionRecordV1> records;
    records.reserve(count);
    if (count == 0) return records;
    const std::uint64_t offset = impl_->records_offset + first_record * kRecordSize;
    const std::size_t byte_count = count * kRecordSize;
    std::vector<unsigned char> raw;
    const unsigned char* bytes = nullptr;
    if (impl_->mapping) {
        bytes = impl_->mapping->data() + offset;
    } else {
        raw.resize(byte_count);
        std::ifstream input(impl_->path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot reopen nbnet file: " + impl_->path);
        input.seekg(static_cast<std::streamoff>(offset));
        input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_count));
        if (!input) throw std::runtime_error("truncated nbnet connection table");
        bytes = raw.data();
    }
    for (std::size_t index = 0; index < count; ++index) {
        ConnectionRecordV1 record = DecodeRecord(bytes + index * kRecordSize);
        if (record.source >= impl_->metadata.neuron_count || record.target >= impl_->metadata.neuron_count) {
            throw std::runtime_error("nbnet connection neuron id is out of range");
        }
        if (!std::isfinite(record.weight) || !std::isfinite(record.max_weight)) {
            throw std::runtime_error("nbnet connection weight is not finite");
        }
        if (record.delay > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("nbnet connection delay exceeds native int range");
        }
        records.push_back(record);
    }
    return records;
}

void NbnetReader::ForEachConnectionBatch(const BatchCallback& callback) const {
    std::ifstream input;
    if (!impl_->mapping) {
        input.open(impl_->path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot reopen nbnet file: " + impl_->path);
        input.seekg(static_cast<std::streamoff>(impl_->records_offset));
    }
    std::vector<unsigned char> raw;
    std::vector<ConnectionRecordV1> records;
    std::uint64_t remaining = impl_->connection_count;
    std::uint64_t offset = impl_->records_offset;
    while (remaining > 0) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, impl_->batch_records));
        const std::size_t byte_count = count * kRecordSize;
        const unsigned char* bytes = nullptr;
        if (impl_->mapping) {
            bytes = impl_->mapping->data() + offset;
        } else {
            raw.resize(byte_count);
            input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_count));
            if (!input) throw std::runtime_error("truncated nbnet connection table");
            bytes = raw.data();
        }
        records.clear();
        records.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            ConnectionRecordV1 record = DecodeRecord(bytes + index * kRecordSize);
            if (record.source >= impl_->metadata.neuron_count || record.target >= impl_->metadata.neuron_count) {
                throw std::runtime_error("nbnet connection neuron id is out of range");
            }
            if (!std::isfinite(record.weight) || !std::isfinite(record.max_weight)) {
                throw std::runtime_error("nbnet connection weight is not finite");
            }
            if (record.delay > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("nbnet connection delay exceeds native int range");
            }
            records.push_back(record);
        }
        callback(records);
        remaining -= count;
        offset += byte_count;
    }
}

}  // namespace streaming
}  // namespace npgr
