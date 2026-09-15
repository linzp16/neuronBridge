#ifndef NPGR_DENSE_BUILTIN_LEARNING_RULE_MODEL_COMMON_H
#define NPGR_DENSE_BUILTIN_LEARNING_RULE_MODEL_COMMON_H

#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"
#include "dense_subnetwork/learning/IDenseLearningRuleModel.h"

#include <boost/any.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace npgr {

namespace {

bool TryGetFloatParameter(const std::map<std::string, boost::any>& parameters,
                          const char* key,
                          float* value) {
    if (value == nullptr) {
        return false;
    }
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end()) {
        return false;
    }
    try {
        *value = boost::any_cast<float>(found->second);
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<float>(boost::any_cast<double>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<float>(boost::any_cast<int>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    return false;
}

float ReadLegacyFloat(const std::map<std::string, boost::any>& parameters,
                      const char* key,
                      float fallback) {
    float value = fallback;
    TryGetFloatParameter(parameters, key, &value);
    return value;
}

float ReadLegacyFloatAny(const std::map<std::string, boost::any>& parameters,
                         const std::vector<const char*>& keys,
                         float fallback) {
    float value = fallback;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (TryGetFloatParameter(parameters, keys[index], &value)) {
            return value;
        }
    }
    return value;
}

bool TryGetIntParameter(const std::map<std::string, boost::any>& parameters,
                        const char* key,
                        int* value) {
    if (value == nullptr) {
        return false;
    }
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end()) {
        return false;
    }
    try {
        *value = boost::any_cast<int>(found->second);
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<int>(boost::any_cast<unsigned int>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<int>(boost::any_cast<float>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<int>(boost::any_cast<double>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    return false;
}

int ReadLegacyInt(const std::map<std::string, boost::any>& parameters,
                  const char* key,
                  int fallback) {
    int value = fallback;
    TryGetIntParameter(parameters, key, &value);
    return value;
}

bool TryGetBoolParameter(const std::map<std::string, boost::any>& parameters,
                         const char* key,
                         bool* value) {
    if (value == nullptr) {
        return false;
    }
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end()) {
        return false;
    }
    try {
        *value = boost::any_cast<bool>(found->second);
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = boost::any_cast<int>(found->second) != 0;
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    return false;
}

bool ReadLegacyBool(const std::map<std::string, boost::any>& parameters,
                    const char* key,
                    bool fallback) {
    bool value = fallback;
    TryGetBoolParameter(parameters, key, &value);
    return value;
}

float ClampPositive(float value) {
    return std::max(1.0e-6f, value);
}

// Matches the legacy Cerebellar kernel support window computation. The result
// is used only to size dense learning spike buffers during finalization.
float CerebellarMaxTimeMeasured(float initpos,
                                float maxpos,
                                float kernel_step_size) {
    maxpos = std::max(initpos + 1.0e-6f, maxpos);
    kernel_step_size = ClampPositive(kernel_step_size);
    const float kernel_width = maxpos - initpos;
    float max_time_measured = maxpos;
    while (true) {
        const float value = (1.0f / kernel_width) * max_time_measured *
            std::exp(-(max_time_measured / kernel_width) + 1.0f);
        if (value < 1.0e-2f) {
            break;
        }
        max_time_measured += kernel_step_size;
    }
    return max_time_measured + initpos;
}

// Returns metadata for a legacy rule id while preserving invalid ids as an
// unsupported empty record. Parameters are not stored in the returned info.
DenseLearningRuleInfo RuleOrDefault(int rule_id,
                                    const std::vector<DenseLearningRuleInfo>& rule_table) {
    if (rule_id < 0 || rule_id >= static_cast<int>(rule_table.size())) {
        return DenseLearningRuleInfo{};
    }
    return rule_table[static_cast<std::size_t>(rule_id)];
}

// Field helpers below are used by both rule-indexed and synapse-indexed tables.
// The caller controls the index meaning: rule_id for DenseLearningRuleFieldTable
// and synapse_id for DenseLearningHostFieldTable.
DenseFieldSpan* FindField(DenseLearningHostFieldTable* fields, const char* name) {
    if (fields == nullptr || name == nullptr) {
        return nullptr;
    }
    std::unordered_map<std::string, int>::iterator found = fields->field_id_by_name.find(name);
    if (found == fields->field_id_by_name.end()) {
        return nullptr;
    }
    return &fields->fields[static_cast<std::size_t>(found->second)];
}

void WriteFloat(DenseLearningHostFieldTable* fields,
                const char* name,
                int synapse_id,
                float value) {
    DenseFieldSpan* span = FindField(fields, name);
    if (span != nullptr) {
        fields->float_pool[static_cast<std::size_t>(span->offset + synapse_id)] = value;
    }
}

void WriteByte(DenseLearningHostFieldTable* fields,
               const char* name,
               int synapse_id,
               unsigned char value) {
    DenseFieldSpan* span = FindField(fields, name);
    if (span != nullptr) {
        fields->byte_pool[static_cast<std::size_t>(span->offset + synapse_id)] = value;
    }
}

void WriteInt(DenseLearningHostFieldTable* fields,
              const char* name,
              int synapse_id,
              int value) {
    DenseFieldSpan* span = FindField(fields, name);
    if (span != nullptr) {
        fields->int_pool[static_cast<std::size_t>(span->offset + synapse_id)] = value;
    }
}

float ReadFloat(const DenseLearningHostFieldTable& fields,
                const char* name,
                int index,
                float fallback) {
    std::unordered_map<std::string, int>::const_iterator found = fields.field_id_by_name.find(name);
    if (found == fields.field_id_by_name.end()) {
        return fallback;
    }
    const DenseFieldSpan& span = fields.fields[static_cast<std::size_t>(found->second)];
    return fields.float_pool[static_cast<std::size_t>(span.offset + index)];
}

int ReadInt(const DenseLearningHostFieldTable& fields,
            const char* name,
            int index,
            int fallback) {
    std::unordered_map<std::string, int>::const_iterator found = fields.field_id_by_name.find(name);
    if (found == fields.field_id_by_name.end()) {
        return fallback;
    }
    const DenseFieldSpan& span = fields.fields[static_cast<std::size_t>(found->second)];
    return fields.int_pool[static_cast<std::size_t>(span.offset + index)];
}

unsigned char ReadByte(const DenseLearningHostFieldTable& fields,
                       const char* name,
                       int index,
                       unsigned char fallback) {
    std::unordered_map<std::string, int>::const_iterator found = fields.field_id_by_name.find(name);
    if (found == fields.field_id_by_name.end()) {
        return fallback;
    }
    const DenseFieldSpan& span = fields.fields[static_cast<std::size_t>(found->second)];
    return fields.byte_pool[static_cast<std::size_t>(span.offset + index)];
}

// Builds a generic field-major table from model-declared schemas. Duplicate
// names are ignored so multiple models can share compatible fields if needed.
bool BuildFlatFieldsFromSchemas(const std::vector<DenseFieldSchema>& schemas,
                                int item_count,
                                DenseLearningHostFieldTable* fields,
                                std::string* reason) {
    if (fields == nullptr) {
        if (reason != nullptr) {
            *reason = "dense learning field table output must not be null";
        }
        return false;
    }
    fields->fields.clear();
    fields->field_id_by_name.clear();
    fields->float_pool.clear();
    fields->int_pool.clear();
    fields->byte_pool.clear();
    for (std::size_t schema_index = 0; schema_index < schemas.size(); ++schema_index) {
        const DenseFieldSchema& schema = schemas[schema_index];
        if (fields->field_id_by_name.find(schema.name) != fields->field_id_by_name.end()) {
            continue;
        }
        DenseFieldSpan span;
        span.field_id = static_cast<int>(fields->fields.size());
        span.name = schema.name;
        span.storage = schema.storage;
        span.role = schema.role;
        span.count = item_count;
        if (schema.storage == DenseFieldStorage::Float32) {
            span.offset = static_cast<int>(fields->float_pool.size());
            fields->float_pool.insert(fields->float_pool.end(),
                                      static_cast<std::size_t>(item_count),
                                      schema.default_float);
        } else if (schema.storage == DenseFieldStorage::Int32) {
            span.offset = static_cast<int>(fields->int_pool.size());
            fields->int_pool.insert(fields->int_pool.end(),
                                    static_cast<std::size_t>(item_count),
                                    schema.default_int);
        } else {
            span.offset = static_cast<int>(fields->byte_pool.size());
            fields->byte_pool.insert(fields->byte_pool.end(),
                                     static_cast<std::size_t>(item_count),
                                     schema.default_byte);
        }
        fields->field_id_by_name[span.name] = span.field_id;
        fields->fields.push_back(span);
    }
    return true;
}

class DenseLearningRuleModelBase : public IDenseLearningRuleModel {
public:
    // Default metadata is enough for direct-runtime fallbacks where a compiled
    // model id exists but no legacy rule table was provided. Concrete parameter
    // defaults still come from RuleFields/SynapseFields.
    DenseLearningRuleInfo DefaultRuleInfo() const override {
        DenseLearningRuleInfo info;
        info.supported = true;
        info.factory_model_id = FactoryModelId();
        info.rule_name = CanonicalName();
        info.flags = Flags();
        return info;
    }
};


}  // namespace
}  // namespace npgr

#endif
