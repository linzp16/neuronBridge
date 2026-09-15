#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"

#include "dense_subnetwork/learning/DenseBuiltinLearningRuleModels.h"
#include "dense_subnetwork/learning/IDenseLearningRuleModel.h"
#include "learning_rule/LearningRuleCatalog.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace npgr {

bool IDenseLearningRuleModel::Validate(std::string* reason) const {
    if (FactoryModelId() < 0) {
        if (reason != nullptr) {
            *reason = "dense learning model has an invalid factory model id";
        }
        return false;
    }
    if (CanonicalName() == nullptr || CanonicalName()[0] == '\0') {
        if (reason != nullptr) {
            *reason = "dense learning model has an empty canonical name";
        }
        return false;
    }
    const int slot_count = FieldSlotCount();
    if (slot_count < 0) {
        if (reason != nullptr) {
            *reason = std::string(CanonicalName()) + " has a negative field slot count";
        }
        return false;
    }
    auto collect_names = [](const std::vector<DenseFieldSchema>& schemas,
                            std::unordered_set<std::string>* names,
                            std::string* duplicate) {
        for (std::size_t index = 0; index < schemas.size(); ++index) {
            if (schemas[index].name.empty()) {
                if (duplicate != nullptr) {
                    *duplicate = "";
                }
                return false;
            }
            if (!names->insert(schemas[index].name).second) {
                if (duplicate != nullptr) {
                    *duplicate = schemas[index].name;
                }
                return false;
            }
        }
        return true;
    };
    std::unordered_set<std::string> rule_names;
    std::string duplicate;
    if (!collect_names(RuleFields(), &rule_names, &duplicate)) {
        if (reason != nullptr) {
            *reason = duplicate.empty()
                ? std::string(CanonicalName()) + " declares an empty rule field name"
                : std::string(CanonicalName()) + " declares duplicate rule field: " + duplicate;
        }
        return false;
    }
    std::unordered_set<std::string> synapse_names;
    if (!collect_names(SynapseFields(), &synapse_names, &duplicate)) {
        if (reason != nullptr) {
            *reason = duplicate.empty()
                ? std::string(CanonicalName()) + " declares an empty synapse field name"
                : std::string(CanonicalName()) + " declares duplicate synapse field: " + duplicate;
        }
        return false;
    }

    std::vector<unsigned char> seen_slots(static_cast<std::size_t>(slot_count), 0);
    const std::vector<DenseFieldSlotBinding> slots = FieldSlots();
    for (std::size_t index = 0; index < slots.size(); ++index) {
        const DenseFieldSlotBinding& slot = slots[index];
        if (slot.slot_id < 0 || slot.slot_id >= slot_count) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " declares an invalid field slot";
            }
            return false;
        }
        if (seen_slots[static_cast<std::size_t>(slot.slot_id)] != 0) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " declares a duplicate field slot";
            }
            return false;
        }
        seen_slots[static_cast<std::size_t>(slot.slot_id)] = 1;
        if (synapse_names.find(slot.field_name) == synapse_names.end()) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " binds slot to unknown synapse field: " +
                    slot.field_name;
            }
            return false;
        }
    }
    for (int slot = 0; slot < slot_count; ++slot) {
        if (seen_slots[static_cast<std::size_t>(slot)] == 0) {
            if (reason != nullptr) {
                std::ostringstream oss;
                oss << CanonicalName() << " leaves field slot " << slot << " unbound";
                *reason = oss.str();
            }
            return false;
        }
    }
    const unsigned char flags = Flags();
    if ((flags & kDenseLearningUsesTriggerRouting) != 0 &&
        (flags & kDenseLearningUsesTrigger) == 0) {
        if (reason != nullptr) {
            *reason = std::string(CanonicalName()) +
                " uses trigger routing without declaring trigger support";
        }
        return false;
    }
    return true;
}

DenseLearningRuleFactory& DenseLearningRuleFactory::Instance() {
    static DenseLearningRuleFactory factory;
    return factory;
}

DenseLearningRuleFactory::DenseLearningRuleFactory() {
#define NPGR_DENSE_LEARNING_MODEL(symbol, id, host_class, pre_fn, post_fn, trigger_fn) \
    Register(std::unique_ptr<IDenseLearningRuleModel>(new host_class()));
#include "dense_subnetwork/learning/DenseLearningModelList.inc"
#undef NPGR_DENSE_LEARNING_MODEL
}

void DenseLearningRuleFactory::Register(std::unique_ptr<IDenseLearningRuleModel> model) {
    if (model) {
        models_.push_back(std::move(model));
    }
}

bool DenseLearningRuleFactory::IsSupportedLegacyRuleId(int legacy_rule_id) const {
    return legacy_rule_id >= 0;
}

unsigned char DenseLearningRuleFactory::FlagsForLegacyRuleId(int legacy_rule_id) const {
    if (!IsSupportedLegacyRuleId(legacy_rule_id)) {
        return 0;
    }
    return static_cast<unsigned char>(kDenseLearningUsesPre | kDenseLearningUsesPost);
}

const IDenseLearningRuleModel* DenseLearningRuleFactory::FindByLegacyName(
    const std::string& name) const {
    const std::string canonical_name =
        LearningRuleCatalog::Instance().ResolveCanonicalName(name);
    for (std::size_t index = 0; index < models_.size(); ++index) {
        if (models_[index]->CanonicalName() == canonical_name ||
            models_[index]->MatchesLegacyName(name)) {
            return models_[index].get();
        }
    }
    return nullptr;
}

const IDenseLearningRuleModel* DenseLearningRuleFactory::FindByModelId(
    int factory_model_id) const {
    for (std::size_t index = 0; index < models_.size(); ++index) {
        if (models_[index]->FactoryModelId() == factory_model_id) {
            return models_[index].get();
        }
    }
    return nullptr;
}

DenseLearningRuleInfo DenseLearningRuleFactory::DescribeLegacyRule(
    int legacy_rule_id,
    const std::vector<DenseLearningRuleInfo>& table) const {
    if (legacy_rule_id < 0 ||
        legacy_rule_id >= static_cast<int>(table.size())) {
        return DenseLearningRuleInfo{};
    }
    return table[static_cast<std::size_t>(legacy_rule_id)];
}

bool DenseLearningRuleFactory::BuildLegacyRuleTable(
    const std::list<LearningRuleDescription>& learning_rule_list,
    std::vector<DenseLearningRuleInfo>* rule_table,
    DenseLearningRuleFieldTable* rule_fields,
    std::string* reason) const {
    if (!ValidateRegisteredModels(reason)) {
        return false;
    }
    if (rule_table == nullptr || rule_fields == nullptr) {
        if (reason != nullptr) {
            *reason = "dense learning rule table outputs must not be null";
        }
        return false;
    }
    // First allocate the union of all registered rule parameter schemas. The
    // table is rule-indexed, so every legacy rule id has a fixed slot even when
    // its concrete model uses only a subset of the fields.
    std::vector<DenseFieldSchema> rule_schemas;
    for (std::size_t model_index = 0; model_index < models_.size(); ++model_index) {
        const std::vector<DenseFieldSchema> model_schemas = models_[model_index]->RuleFields();
        rule_schemas.insert(rule_schemas.end(), model_schemas.begin(), model_schemas.end());
    }
    if (!BuildFlatFieldsFromSchemas(rule_schemas,
                                    static_cast<int>(learning_rule_list.size()),
                                    rule_fields,
                                    reason)) {
        return false;
    }
    rule_table->clear();
    // Preserve legacy rule ids by pushing one metadata entry per source rule,
    // including unsupported rules. Dense synapses that bind to unsupported
    // entries are rejected later during binding compilation.
    rule_table->reserve(learning_rule_list.size());
    int rule_id = 0;
    for (std::list<LearningRuleDescription>::const_iterator it = learning_rule_list.begin();
         it != learning_rule_list.end();
         ++it, ++rule_id) {
        DenseLearningRuleInfo info;
        info.rule_name = it->RuleName;
        const IDenseLearningRuleModel* model = FindByLegacyName(it->RuleName);
        if (model != nullptr) {
            info = model->DefaultRuleInfo();
            info.rule_name = it->RuleName;
            if (!model->FillRuleFields(rule_id, *it, rule_fields, reason)) {
                return false;
            }
        }
        rule_table->push_back(info);
    }
    return true;
}

bool DenseLearningRuleFactory::BuildFieldTable(const std::vector<DenseLearningModelSpec>& specs,
                                               int synapse_count,
                                               DenseLearningHostFieldTable* fields,
                                               DenseLearningModelFieldIndexTable* indices,
                                               std::string* reason) const {
    if (!ValidateRegisteredModels(reason)) {
        return false;
    }
    if (fields == nullptr || indices == nullptr) {
        if (reason != nullptr) {
            *reason = "dense learning field table outputs must not be null";
        }
        return false;
    }
    fields->fields.clear();
    fields->field_id_by_name.clear();
    fields->float_pool.clear();
    fields->int_pool.clear();
    fields->byte_pool.clear();
    indices->spans.clear();
    indices->field_indices.clear();

    // Add a synapse-indexed field exactly once by name. This allows future
    // models to intentionally share a compatible runtime field without
    // changing the table builder.
    auto add_field = [&](const DenseFieldSchema& schema) {
        // find the specific parameter by name
        std::unordered_map<std::string, int>::const_iterator existing =
            fields->field_id_by_name.find(schema.name);
        if (existing != fields->field_id_by_name.end()) {
            return existing->second;
        }
        // if this is a new parameter,then build a new span
        DenseFieldSpan span;
        // assign the new span's field id
        span.field_id = static_cast<int>(fields->fields.size());
        // assign the span name
        span.name = schema.name;
        // assign the storage value type
        span.storage = schema.storage;
        // assign the storage value role
        span.role = schema.role;
        // give the span count
        span.count = synapse_count;
        // add the new parameter to the total field type
        if (schema.storage == DenseFieldStorage::Float32) {
            span.offset = static_cast<int>(fields->float_pool.size());
            fields->float_pool.insert(fields->float_pool.end(),
                                      static_cast<std::size_t>(synapse_count),
                                      schema.default_float);
        } else if (schema.storage == DenseFieldStorage::Int32) {
            span.offset = static_cast<int>(fields->int_pool.size());
            fields->int_pool.insert(fields->int_pool.end(),
                                    static_cast<std::size_t>(synapse_count),
                                    schema.default_int);
        } else {
            span.offset = static_cast<int>(fields->byte_pool.size());
            fields->byte_pool.insert(fields->byte_pool.end(),
                                     static_cast<std::size_t>(synapse_count),
                                     schema.default_byte);
        }
        // assign the float type
        fields->field_id_by_name[span.name] = span.field_id;
        // push back the float type
        fields->fields.push_back(span);
        return span.field_id;
    };
    // scan all the learning models
    for (std::size_t spec_index = 0; spec_index < specs.size(); ++spec_index) {
        const DenseLearningModelSpec& spec = specs[spec_index];
        // find the learning rule model by the model id in the Learning model description
        const IDenseLearningRuleModel* model = FindByModelId(spec.factory_model_id);
        if (model == nullptr) {
            if (reason != nullptr) {
                *reason = "dense learning model spec references an unregistered model id";
            }
            return false;
        }
        // find the needed schemas of the model
        const std::vector<DenseFieldSchema> schemas = model->SynapseFields();
        //  scan all the data structure of the model
        for (std::size_t field_index = 0; field_index < schemas.size(); ++field_index) {
            add_field(schemas[field_index]);
        }
        // give the learning model span
        DenseLearningModelSpan span;
        span.model_id = spec.model_id;
        span.factory_model_id = spec.factory_model_id;
        // get the model's field index offset
        span.field_index_offset = static_cast<int>(indices->field_indices.size());
        // make a new vector to store the parameter of the model field parameterss
        std::vector<int> field_indices(static_cast<std::size_t>(model->FieldSlotCount()), -1);
        // Convert model-owned slot names into runtime field ids. Device code
        // receives only the compact integer array.
        // get the needed slots of the model
        const std::vector<DenseFieldSlotBinding> slots = model->FieldSlots();
        // scann all the needed slots
        for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
            const DenseFieldSlotBinding& slot = slots[slot_index];
            if (slot.slot_id < 0 || slot.slot_id >= model->FieldSlotCount()) {
                if (reason != nullptr) {
                    *reason = "dense learning model declares an invalid field slot";
                }
                return false;
            }
            // find the field name of the slot
            const std::unordered_map<std::string, int>::const_iterator found =
                fields->field_id_by_name.find(slot.field_name);
            if (found == fields->field_id_by_name.end()) {
                if (reason != nullptr) {
                    *reason = "dense learning model field slot references an unknown field";
                }
                return false;
            }
            field_indices[static_cast<std::size_t>(slot.slot_id)] = found->second;
        }
        for (std::size_t index = 0; index < field_indices.size(); ++index) {
            if (field_indices[index] < 0) {
                if (reason != nullptr) {
                    *reason = "dense learning model has an unbound field slot";
                }
                return false;
            }
        }
        // insert the span's field_indics to the total field_indics
        indices->field_indices.insert(indices->field_indices.end(),
                                      field_indices.begin(),
                                      field_indices.end());
        // assign the span's index count
        span.field_index_count =
            static_cast<int>(indices->field_indices.size()) - span.field_index_offset;
        indices->spans.push_back(span);
    }
    return true;
}

bool DenseLearningRuleFactory::BuildRegisteredFieldTable(
    int synapse_count,
    DenseLearningHostFieldTable* fields,
    DenseLearningModelFieldIndexTable* indices,
    std::string* reason) const {
    std::vector<const IDenseLearningRuleModel*> ordered_models;
    ordered_models.reserve(models_.size());
    for (std::size_t index = 0; index < models_.size(); ++index) {
        ordered_models.push_back(models_[index].get());
    }
    std::sort(ordered_models.begin(), ordered_models.end(),
              [](const IDenseLearningRuleModel* lhs, const IDenseLearningRuleModel* rhs) {
                  return lhs->FactoryModelId() < rhs->FactoryModelId();
              });
    std::vector<DenseLearningModelSpec> specs;
    specs.reserve(ordered_models.size());
    for (std::size_t index = 0; index < ordered_models.size(); ++index) {
        if (ordered_models[index]->FactoryModelId() != static_cast<int>(index)) {
            if (reason != nullptr) {
                *reason = "dense learning model ids must be contiguous from zero";
            }
            return false;
        }
        // The registered table keeps model_id == factory_model_id so the GPU
        // kernel can use syn_plastic_model_id directly as the span index.
        DenseLearningModelSpec spec;
        spec.factory_model_id = ordered_models[index]->FactoryModelId();
        spec.model_id = ordered_models[index]->FactoryModelId();
        spec.legacy_rule_name = ordered_models[index]->CanonicalName();
        specs.push_back(spec);
    }
    return BuildFieldTable(specs, synapse_count, fields, indices, reason);
}

bool DenseLearningRuleFactory::CompileSynapseLearningBindings(
    int synapse_count,
    const std::vector<int>& plastic_rule_ids,
    const std::vector<int>& trigger_rule_ids,
    const std::vector<DenseLearningRuleInfo>& rule_table,
    std::vector<int>* plastic_model_ids,
    std::vector<unsigned char>* plastic_flags,
    std::vector<int>* trigger_model_ids,
    std::vector<unsigned char>* trigger_flags,
    std::string* reason) const {
    if (plastic_model_ids == nullptr || plastic_flags == nullptr ||
        trigger_model_ids == nullptr || trigger_flags == nullptr) {
        if (reason != nullptr) {
            *reason = "dense learning binding outputs must not be null";
        }
        return false;
    }
    if (plastic_rule_ids.size() != static_cast<std::size_t>(synapse_count) ||
        trigger_rule_ids.size() != static_cast<std::size_t>(synapse_count)) {
        if (reason != nullptr) {
            *reason = "dense learning rule arrays must be parallel to synapses";
        }
        return false;
    }
    plastic_model_ids->assign(static_cast<std::size_t>(synapse_count), -1);
    plastic_flags->assign(static_cast<std::size_t>(synapse_count), 0);
    trigger_model_ids->assign(static_cast<std::size_t>(synapse_count), -1);
    trigger_flags->assign(static_cast<std::size_t>(synapse_count), 0);
    // Map a legacy rule id to the registered model id and phase flags. This is
    // metadata-only; model-specific parameters have already been parsed into
    // DenseLearningRuleFieldTable.
    auto compile_rule = [&](int rule_id,
                            int synapse_id,
                            const char* role,
                            std::vector<int>* model_ids,
                            std::vector<unsigned char>* flags) {
        if (rule_id < 0) {
            return true;
        }
        // get the learning rule model index id in the model factory
        const DenseLearningRuleInfo info = DescribeLegacyRule(rule_id, rule_table);
        // check whether the rule is legal
        if (!info.supported || info.factory_model_id < 0) {
            if (reason != nullptr) {
                *reason = std::string("dense ") + role +
                    " synapse references an unsupported learning rule id";
            }
            return false;
        }
        const std::size_t index = static_cast<std::size_t>(synapse_id);
        // set the connection factory model id
        (*model_ids)[index] = info.factory_model_id;
        // set the connection learning flag
        (*flags)[index] = info.flags;
        return true;
    };
    // scan all the synapse 
    for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
        // get the index
        const std::size_t index = static_cast<std::size_t>(synapse_id);
        // Compile plastic and trigger attachments independently because a
        // single synapse can be a trigger carrier without being plastic itself.
        if (!compile_rule(plastic_rule_ids[index],
                          synapse_id,
                          "plastic",
                          plastic_model_ids,
                          plastic_flags) ||
            !compile_rule(trigger_rule_ids[index],
                          synapse_id,
                          "trigger",
                          trigger_model_ids,
                          trigger_flags)) {
            return false;
        }
    }
    return true;
}

bool DenseLearningRuleFactory::PopulateFieldTableFromRuleBindings(
    int synapse_count,
    const std::vector<int>& plastic_rule_ids,
    const std::vector<int>& plastic_model_ids,
    const std::vector<int>& trigger_rule_ids,
    const std::vector<int>& trigger_model_ids,
    const std::vector<float>& max_weights,
    const std::vector<DenseLearningRuleInfo>& rule_table,
    const DenseLearningRuleFieldTable& rule_fields,
    DenseLearningHostFieldTable* fields,
    std::string* reason) const {
    if (fields == nullptr) {
        if (reason != nullptr) {
            *reason = "dense learning field table must not be null";
        }
        return false;
    }
    if (plastic_rule_ids.size() != static_cast<std::size_t>(synapse_count) ||
        plastic_model_ids.size() != static_cast<std::size_t>(synapse_count) ||
        trigger_rule_ids.size() != static_cast<std::size_t>(synapse_count) ||
        trigger_model_ids.size() != static_cast<std::size_t>(synapse_count) ||
        max_weights.size() != static_cast<std::size_t>(synapse_count)) {
        if (reason != nullptr) {
            *reason = "dense learning rule bindings must be parallel to synapses";
        }
        return false;
    }

    // Each registered model expands its own rule-indexed parameters into the
    // shared synapse-indexed runtime table. The factory does not inspect
    // concrete field names here.
    // scan all the learning rule models and Fill the parameters
    for (std::size_t model_index = 0; model_index < models_.size(); ++model_index) {
        // fill the model parameters
        if (!models_[model_index]->FillSynapseFields(synapse_count,
                                                     plastic_rule_ids,
                                                     plastic_model_ids,
                                                     trigger_rule_ids,
                                                     trigger_model_ids,
                                                     max_weights,
                                                     rule_table,
                                                     rule_fields,
                                                     fields,
                                                     reason)) {
            return false;
        }
    }
    return true;
}

float DenseLearningRuleFactory::MaxBufferedTimeForRule(
    int legacy_rule_id,
    const std::vector<DenseLearningRuleInfo>& rule_table,
    const DenseLearningRuleFieldTable& rule_fields) const {
    const DenseLearningRuleInfo info = RuleOrDefault(legacy_rule_id, rule_table);
    const IDenseLearningRuleModel* model = FindByModelId(info.factory_model_id);
    if (model == nullptr) {
        return 0.0f;
    }
    return model->SpikeBufferWindowMs(legacy_rule_id, rule_fields);
}

float DenseLearningRuleFactory::MaxBufferedTimeForBinding(
    int legacy_rule_id,
    int factory_model_id,
    const std::vector<DenseLearningRuleInfo>& rule_table,
    const DenseLearningRuleFieldTable& rule_fields) const {
    DenseLearningRuleInfo info = RuleOrDefault(legacy_rule_id, rule_table);
    const IDenseLearningRuleModel* model = FindByModelId(factory_model_id);
    if (!info.supported) {
        if (model == nullptr) {
            return 0.0f;
        }
        info = model->DefaultRuleInfo();
    }
    if (model == nullptr) {
        model = FindByModelId(info.factory_model_id);
    }
    if (model == nullptr) {
        return 0.0f;
    }
    return model->SpikeBufferWindowMs(legacy_rule_id, rule_fields);
}

bool DenseLearningRuleFactory::ValidateRegisteredModels(std::string* reason) const {
    std::set<int> ids;
    std::set<std::string> canonical_names;
    for (std::size_t index = 0; index < models_.size(); ++index) {
        const IDenseLearningRuleModel* model = models_[index].get();
        if (model == nullptr) {
            if (reason != nullptr) {
                *reason = "dense learning factory contains a null model";
            }
            return false;
        }
        if (!model->Validate(reason)) {
            return false;
        }
        if (!ids.insert(model->FactoryModelId()).second) {
            if (reason != nullptr) {
                std::ostringstream oss;
                oss << "duplicate dense learning model id: " << model->FactoryModelId();
                *reason = oss.str();
            }
            return false;
        }
        const std::string canonical_name = model->CanonicalName();
        if (!canonical_names.insert(canonical_name).second) {
            if (reason != nullptr) {
                *reason = "duplicate dense learning canonical name: " + canonical_name;
            }
            return false;
        }
        if (!LearningRuleCatalog::Instance().IsDenseGpuSupported(canonical_name)) {
            if (reason != nullptr) {
                *reason = "dense learning model is missing from LearningRuleCatalog: " +
                    canonical_name;
            }
            return false;
        }
    }
    const std::vector<std::string> dense_names =
        LearningRuleCatalog::Instance().DenseGpuSupportedRuleNames();
    for (std::size_t index = 0; index < dense_names.size(); ++index) {
        if (FindByLegacyName(dense_names[index]) == nullptr) {
            if (reason != nullptr) {
                *reason = "LearningRuleCatalog marks an unregistered dense rule as supported: " +
                    dense_names[index];
            }
            return false;
        }
    }
    return true;
}

}  // namespace npgr
