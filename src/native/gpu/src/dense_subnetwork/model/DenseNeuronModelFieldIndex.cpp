#include "dense_subnetwork/model/DenseNeuronModelFieldIndex.h"

#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "dense_subnetwork/model/IDenseNeuronModel.h"

namespace npgr {

bool BuildDenseModelFieldIndexTable(const std::vector<DenseNeuronModelSpec>& specs,
                                    const DenseNeuronHostFieldTable& fields,
                                    int neuron_count,
                                    DenseModelFieldIndexTable* table,
                                    std::string* reason) {
    if (table == nullptr) {
        if (reason != nullptr) {
            *reason = "DenseModelFieldIndexTable output must not be null";
        }
        return false;
    }
    table->spans.clear();
    table->field_indices.clear();
    table->span_index_by_neuron.assign(static_cast<std::size_t>(neuron_count), -1);
    // scan all the merged models
    for (std::size_t spec_index = 0; spec_index < specs.size(); ++spec_index) {
        const DenseNeuronModelSpec& spec = specs[spec_index];
        const IDenseNeuronModel* model =
            DenseNeuronModelFactory::Instance().FindByModelId(spec.factory_model_id);
        if (model == nullptr) {
            if (reason != nullptr) {
                *reason = "dense neuron factory model id is not registered";
            }
            return false;
        }

        DenseModelFieldIndexSpan span;
        span.model_id = spec.model_id;
        span.factory_model_id = spec.factory_model_id;
        // define the base offset
        span.field_index_offset = static_cast<int>(table->field_indices.size());
        // get the number of parameters
        span.field_index_count = model->FieldSlotCount();
        if (span.field_index_count <= 0) {
            if (reason != nullptr) {
                *reason = "dense model field slot count must be positive";
            }
            return false;
        }
        // define the default insert
        table->field_indices.insert(table->field_indices.end(),
                                    static_cast<std::size_t>(span.field_index_count),
                                    -1);
        // get all the needed parameter slots of the model
        const std::vector<DenseFieldSlotBinding> slots = model->FieldSlots();
        // scan all the slot
        for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
            // check the slot id
            const DenseFieldSlotBinding& binding = slots[slot_index];
            if (binding.slot_id < 0 || binding.slot_id >= span.field_index_count) {
                if (reason != nullptr) {
                    *reason = "dense model field slot id is out of range";
                }
                return false;
            }
            // check the slot binding
            int& destination =
                table->field_indices[static_cast<std::size_t>(span.field_index_offset + binding.slot_id)];
            if (destination >= 0) {
                if (reason != nullptr) {
                    *reason = "dense model field slot id is duplicated";
                }
                return false;
            }
            // find the name in the field table name
            std::unordered_map<std::string, int>::const_iterator found =
                fields.field_id_by_name.find(binding.field_name);
            if (found == fields.field_id_by_name.end()) {
                if (reason != nullptr) {
                    *reason = "dense model field slot is not present in the field table";
                }
                return false;
            }
            destination = found->second;
        }
        for (int slot_id = 0; slot_id < span.field_index_count; ++slot_id) {
            if (table->field_indices[static_cast<std::size_t>(span.field_index_offset + slot_id)] < 0) {
                if (reason != nullptr) {
                    *reason = "dense model field slot binding is incomplete";
                }
                return false;
            }
        }
        const int span_index = static_cast<int>(table->spans.size());
        // push back the span
        table->spans.push_back(span);

        const DenseNeuronRange& range = spec.range;
        // map the neuron id to the span
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            if (neuron_id >= 0 && neuron_id < neuron_count) {
                // generate each neuron's span_index
                table->span_index_by_neuron[static_cast<std::size_t>(neuron_id)] = span_index;
            }
        }
    }

    return true;
}

}  // namespace npgr
