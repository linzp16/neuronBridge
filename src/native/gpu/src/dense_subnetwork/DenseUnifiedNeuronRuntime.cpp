#include "dense_subnetwork/DenseUnifiedNeuronRuntime.h"

#include "dense_subnetwork/model/DenseNeuronFieldAccess.h"
#include "dense_subnetwork/model/DenseNeuronFieldTableBuilder.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "dense_subnetwork/model/IDenseNeuronModel.h"

#include <algorithm>

namespace npgr {

namespace {

std::vector<float> CopyFloatField(const DenseNeuronHostFieldTable& table,
                                  const DenseFieldSpan& span) {
    if (span.storage != DenseFieldStorage::Float32) {
        return std::vector<float>();
    }
    const float* begin = table.float_pool.data() + span.offset;
    return std::vector<float>(begin, begin + span.count);
}

std::vector<unsigned char> CopyByteField(const DenseNeuronHostFieldTable& table,
                                         const DenseFieldSpan& span) {
    if (span.storage != DenseFieldStorage::UInt8) {
        return std::vector<unsigned char>();
    }
    const unsigned char* begin = table.byte_pool.data() + span.offset;
    return std::vector<unsigned char>(begin, begin + span.count);
}

}  // namespace

DenseUnifiedNeuronRuntime::DenseUnifiedNeuronRuntime()
    : float_pool_size_(0),
      int_pool_size_(0),
      byte_pool_size_(0),
      neuron_count_(0),
      dt_ms_(1.0f),
      initialized_(false),
      device_ready_(false),
      host_state_dirty_(false),
      d_model_id_by_neuron_(nullptr),
      d_active_mask_(nullptr),
      d_output_neuron_mask_(nullptr),
      d_field_spans_(nullptr),
      d_float_field_pool_(nullptr),
      d_int_field_pool_(nullptr),
      d_byte_field_pool_(nullptr),
      d_reset_float_field_pool_(nullptr),
      d_reset_int_field_pool_(nullptr),
      d_reset_byte_field_pool_(nullptr),
      d_model_field_spans_(nullptr),
      d_model_field_indices_(nullptr),
      d_model_field_span_by_neuron_(nullptr),
      d_current_firing_ids_(nullptr),
      d_current_firing_count_(nullptr),
      d_current_did_fire_(nullptr),
      d_output_firing_ids_(nullptr),
      d_output_firing_count_(nullptr),
      device_firing_capacity_(0),
      device_output_firing_capacity_(0) {
    device_fields_ = DenseNeuronDeviceFieldTable{};
    device_model_field_indices_ = DenseDeviceModelFieldIndexTable{};
}

DenseUnifiedNeuronRuntime::~DenseUnifiedNeuronRuntime() {
    DestroyDeviceBuffers();
}

bool DenseUnifiedNeuronRuntime::Initialize(const std::vector<DenseNeuronModelSpec>& specs,
                                           int neuron_count,
                                           float dt_ms,
                                           std::string* reason) {
    if (!DenseNeuronModelFactory::Instance().ValidateRegisteredModels(reason)) {
        return false;
    }
    if (neuron_count <= 0) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime requires a positive neuron_count";
        }
        return false;
    }
    neuron_count_ = neuron_count;
    dt_ms_ = dt_ms;
    this->ClearHostBuildCaches();
    host_field_spans_.clear();
    debug_model_infos_.clear();
    float_pool_size_ = 0;
    int_pool_size_ = 0;
    byte_pool_size_ = 0;
    // model_id_by_neuron_ and active_mask_ are initialized to all -1 and 0
    model_id_by_neuron_.assign(static_cast<std::size_t>(neuron_count_), -1);
    active_mask_.assign(static_cast<std::size_t>(neuron_count_), 0);
    output_neuron_mask_.assign(static_cast<std::size_t>(neuron_count_), 0);
    // fill the model_id_by_neuron_ and active_mask_
    // scan all the specs of the models
    for (std::size_t spec_index = 0; spec_index < specs.size(); ++spec_index) {
        
        const DenseNeuronModelSpec& spec = specs[spec_index];
        const DenseNeuronRange& range = spec.range;
        // fill the neuron model according to the range
        // scan all the neurons in the ranges
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            if (neuron_id >= 0 && neuron_id < neuron_count_) {
                const std::size_t idx = static_cast<std::size_t>(neuron_id);
                model_id_by_neuron_[idx] = spec.model_id;
                active_mask_[idx] = 1;
            }
        }
    }

    DenseNeuronHostFieldTable host_fields;
    // build the parameter table
    if (!BuildDenseNeuronHostFieldTable(specs, neuron_count_, &host_fields, reason)) {
        initialized_ = false;
        return false;
    }
    // scan all the spec of the models
    for (std::size_t spec_index = 0; spec_index < specs.size(); ++spec_index) {
        const DenseNeuronModelSpec& spec = specs[spec_index];
        // get the model expression by the modelID
        const IDenseNeuronModel* model =
            DenseNeuronModelFactory::Instance().FindByModelId(spec.factory_model_id);
        // fill the parameter and caculate the decay factor
        if (model == nullptr ||
            !model->FillInitialFieldValues(&host_fields, spec, reason) ||
            !model->BuildDerivedFields(&host_fields, spec, dt_ms_, reason)) {
            initialized_ = false;
            return false;
        }
    }
    if (!BuildDenseModelFieldIndexTable(specs, host_fields, neuron_count_, &model_field_indices_, reason)) {
        initialized_ = false;
        return false;
    }
    host_field_spans_ = host_fields.fields;
    float_pool_size_ = host_fields.float_pool.size();
    int_pool_size_ = host_fields.int_pool.size();
    byte_pool_size_ = host_fields.byte_pool.size();
    debug_model_infos_.reserve(specs.size());
    for (std::size_t spec_index = 0; spec_index < specs.size(); ++spec_index) {
        DenseRuntimeModelDebugInfo info;
        info.factory_model_id = specs[spec_index].factory_model_id;
        info.model_id = specs[spec_index].model_id;
        info.legacy_model_name = specs[spec_index].legacy_model_name;
        debug_model_infos_.push_back(info);
    }
    initial_host_fields_ = host_fields;

    initialized_ = true;
    host_state_dirty_ = false;
    return true;
}

bool DenseUnifiedNeuronRuntime::ResetState(std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime is not initialized";
        }
        return false;
    }
    host_state_dirty_ = false;
    if (device_ready_) {
        return ResetStateFieldsGpu(reason);
    }
    return true;
}

bool DenseUnifiedNeuronRuntime::BuildDebugSnapshots(std::vector<DenseNeuronDebugSnapshot>* out,
                                                    std::string* reason) const {
    if (out == nullptr) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime debug output must not be null";
        }
        return false;
    }
    DenseNeuronHostFieldTable debug_fields;
    if (!DownloadFieldsToHost(&debug_fields, reason)) {
        return false;
    }

    out->clear();
    for (std::size_t spec_index = 0; spec_index < debug_model_infos_.size(); ++spec_index) {
        const DenseRuntimeModelDebugInfo& info = debug_model_infos_[spec_index];
        const IDenseNeuronModel* model =
            DenseNeuronModelFactory::Instance().FindByModelId(info.factory_model_id);
        if (model == nullptr) {
            continue;
        }

        DenseNeuronDebugSnapshot snapshot;
        snapshot.factory_model_id = info.factory_model_id;
        snapshot.model_id = info.model_id;
        snapshot.legacy_model_name = info.legacy_model_name;

        std::vector<unsigned char> model_mask(static_cast<std::size_t>(neuron_count_), 0);
        for (int neuron_index = 0; neuron_index < neuron_count_; ++neuron_index) {
            if (model_id_by_neuron_[static_cast<std::size_t>(neuron_index)] == info.model_id &&
                active_mask_[static_cast<std::size_t>(neuron_index)] != 0) {
                model_mask[static_cast<std::size_t>(neuron_index)] = 1;
            }
        }
        snapshot.byte_state_vectors["active_mask"] = model_mask;

        const std::vector<DenseFieldSchema> fields = model->Fields();
        for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
            const DenseFieldSchema& schema = fields[field_index];
            if (schema.role != DenseFieldRole::State && schema.role != DenseFieldRole::Debug) {
                continue;
            }
            const DenseFieldSpan* span = FindField(debug_fields, schema.name);
            if (span == nullptr) {
                continue;
            }
            if (span->storage == DenseFieldStorage::Float32) {
                snapshot.float_state_vectors[schema.name] = CopyFloatField(debug_fields, *span);
            } else if (span->storage == DenseFieldStorage::UInt8) {
                snapshot.byte_state_vectors[schema.name] = CopyByteField(debug_fields, *span);
            }
        }
        out->push_back(snapshot);
    }
    return true;
}

void DenseUnifiedNeuronRuntime::ClearHostBuildCaches() {
    initial_host_fields_ = DenseNeuronHostFieldTable{};
    model_field_indices_ = DenseModelFieldIndexTable{};
}

}  // namespace npgr
