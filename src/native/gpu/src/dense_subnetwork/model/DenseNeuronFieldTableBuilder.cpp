#include "dense_subnetwork/model/DenseNeuronFieldTableBuilder.h"

#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "dense_subnetwork/model/IDenseNeuronModel.h"

namespace npgr {

bool BuildDenseNeuronHostFieldTable(const std::vector<DenseNeuronModelSpec>& specs,
                                    int neuron_count,
                                    DenseNeuronHostFieldTable* table,
                                    std::string* reason) {
    if (table == nullptr) {
        if (reason != nullptr) {
            *reason = "DenseNeuronHostFieldTable output must not be null";
        }
        return false;
    }
    if (neuron_count <= 0) {
        if (reason != nullptr) {
            *reason = "neuron_count must be positive when building dense field table";
        }
        return false;
    }

    table->fields.clear();
    table->field_id_by_name.clear();
    table->float_pool.clear();
    table->int_pool.clear();
    table->byte_pool.clear();
    // scan all the merged neuron models
    for (std::size_t spec_index = 0; spec_index < specs.size(); ++spec_index) {
        // get the model spec
        const DenseNeuronModelSpec& spec = specs[spec_index];
        // get the model by the factory model id
        const IDenseNeuronModel* model =
            DenseNeuronModelFactory::Instance().FindByModelId(spec.factory_model_id);
        if (model == nullptr) {
            if (reason != nullptr) {
                *reason = "dense neuron factory model id is not registered";
            }
            return false;
        }
        // get the field schema of the model registered in the factory
        const std::vector<DenseFieldSchema> fields = model->Fields();
        // scan all the field members of the model
        for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
            const DenseFieldSchema& schema = fields[field_index];
            // check if the name if empty or it is already in the table
            if (schema.name.empty() || table->field_id_by_name.find(schema.name) != table->field_id_by_name.end()) {
                continue;
            }

            DenseFieldSpan span;
            // define the field id
            span.field_id = static_cast<int>(table->fields.size());
            // define the field name
            span.name = schema.name;
            // define the storage type(float, int, byte)
            span.storage = schema.storage;
            // define the role(parameter, state, derived, debug)
            span.role = schema.role;
            span.count = neuron_count;
            // add the  parameters to the pool
            if (schema.storage == DenseFieldStorage::Float32) {
                span.offset = static_cast<int>(table->float_pool.size());
                table->float_pool.insert(table->float_pool.end(),
                                         static_cast<std::size_t>(neuron_count),
                                         schema.default_float);
            } else if (schema.storage == DenseFieldStorage::Int32) {
                span.offset = static_cast<int>(table->int_pool.size());
                table->int_pool.insert(table->int_pool.end(),
                                       static_cast<std::size_t>(neuron_count),
                                       schema.default_int);
            } else {
                span.offset = static_cast<int>(table->byte_pool.size());
                table->byte_pool.insert(table->byte_pool.end(),
                                        static_cast<std::size_t>(neuron_count),
                                        schema.default_byte);
            }

            table->field_id_by_name[span.name] = span.field_id;
            table->fields.push_back(span);
        }
    }

    return true;
}

}  // namespace npgr
