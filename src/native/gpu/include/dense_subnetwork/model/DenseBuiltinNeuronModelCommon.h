#ifndef NPGR_DENSE_BUILTIN_NEURON_MODEL_COMMON_H
#define NPGR_DENSE_BUILTIN_NEURON_MODEL_COMMON_H

#include "dense_subnetwork/model/DenseNeuronFieldAccess.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "dense_subnetwork/model/IDenseNeuronModel.h"

#include <boost/any.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace npgr {

namespace {

bool TryGetFloatParameter(const std::map<std::string, boost::any>& parameters,
                          const std::string& key,
                          float* value) {
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end() || value == nullptr) {
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

void FillRangeParam(DenseNeuronModelSpec* spec,
                    const DenseNeuronRange& range,
                    int neuron_count,
                    const char* key,
                    float value) {
    if (spec == nullptr || key == nullptr || neuron_count <= 0 || range.count <= 0) {
        return;
    }
    std::vector<float>& target = spec->per_neuron_params[key];
    if (target.empty()) {
        target.assign(static_cast<std::size_t>(neuron_count), value);
    }
    for (int offset = 0; offset < range.count; ++offset) {
        const int neuron_id = range.begin + offset;
        if (neuron_id >= 0 && neuron_id < neuron_count) {
            target[static_cast<std::size_t>(neuron_id)] = value;
        }
    }
}

bool TryGetSpecFloat(const DenseNeuronModelSpec& spec,
                     const std::vector<const char*>& keys,
                     int neuron_id,
                     float fallback,
                     float* value) {
    if (value == nullptr) {
        return false;
    }
    for (std::size_t key_index = 0; key_index < keys.size(); ++key_index) {
        const std::map<std::string, std::vector<float> >::const_iterator found =
            spec.per_neuron_params.find(keys[key_index]);
        if (found == spec.per_neuron_params.end() || found->second.empty()) {
            continue;
        }
        if (found->second.size() == 1) {
            if (!std::isnan(found->second.front())) {
                *value = found->second.front();
                return true;
            }
            continue;
        }
        if (neuron_id >= 0 && neuron_id < static_cast<int>(found->second.size())) {
            const float candidate = found->second[static_cast<std::size_t>(neuron_id)];
            if (!std::isnan(candidate)) {
                *value = candidate;
                return true;
            }
        }
    }
    *value = fallback;
    return false;
}

void FillFloatField(DenseNeuronHostFieldTable* table,
                    const DenseNeuronModelSpec& spec,
                    const char* field_name,
                    const std::vector<const char*>& aliases,
                    float fallback) {
    float* field = FloatField(table, field_name);
    if (field == nullptr) {
        return;
    }
    const DenseNeuronRange& range = spec.range;
    for (int offset = 0; offset < range.count; ++offset) {
        const int neuron_id = range.begin + offset;
        float value = fallback;
        TryGetSpecFloat(spec, aliases, neuron_id, fallback, &value);
        field[neuron_id] = value;
    }
}

void FillFloatConstant(DenseNeuronHostFieldTable* table,
                       const DenseNeuronModelSpec& spec,
                       const char* field_name,
                       float value) {
    float* field = FloatField(table, field_name);
    if (field == nullptr) {
        return;
    }
    const DenseNeuronRange& range = spec.range;
    for (int offset = 0; offset < range.count; ++offset) {
        const int neuron_id = range.begin + offset;
        field[neuron_id] = value;
    }
}

void FillByteConstant(DenseNeuronHostFieldTable* table,
                      const DenseNeuronModelSpec& spec,
                      const char* field_name,
                      unsigned char value) {
    unsigned char* field = ByteField(table, field_name);
    if (field == nullptr) {
        return;
    }
    const DenseNeuronRange& range = spec.range;
    for (int offset = 0; offset < range.count; ++offset) {
        const int neuron_id = range.begin + offset;
        field[neuron_id] = value;
    }
}

void FillIntConstant(DenseNeuronHostFieldTable* table,
                     const DenseNeuronModelSpec& spec,
                     const char* field_name,
                     int value) {
    int* field = IntField(table, field_name);
    if (field == nullptr) {
        return;
    }
    const DenseNeuronRange& range = spec.range;
    for (int offset = 0; offset < range.count; ++offset) {
        const int neuron_id = range.begin + offset;
        field[neuron_id] = value;
    }
}

class DenseNeuronModelBase : public IDenseNeuronModel {
public:
    bool FillLayerParams(DenseNeuronModelSpec* spec,
                         const NeuronLayerDescription& layer,
                         const DenseNeuronRange& range,
                         int neuron_count,
                         std::string*) const override {
        float value = 0.0f;
        const std::vector<const char*> keys = ParameterKeys();
        for (std::size_t index = 0; index < keys.size(); ++index) {
            if (TryGetFloatParameter(layer.NeuronParameter, keys[index], &value)) {
                FillRangeParam(spec, range, neuron_count, keys[index], value);
            }
        }
        return true;
    }

protected:
    virtual std::vector<const char*> ParameterKeys() const = 0;
};


}  // namespace

}  // namespace npgr

#endif
