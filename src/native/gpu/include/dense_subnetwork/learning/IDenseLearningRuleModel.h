#ifndef NPGR_I_DENSE_LEARNING_RULE_MODEL_H
#define NPGR_I_DENSE_LEARNING_RULE_MODEL_H

#include "dense_subnetwork/learning/DenseLearningModelSpec.h"
#include "dense_subnetwork/model/DenseNeuronFieldSchema.h"
#include "dense_subnetwork/model/IDenseNeuronModel.h"
#include "source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"

#include <string>
#include <vector>

namespace npgr {

struct DenseLearningRuleInfo;

// Extension point for dense learning models. The dense build pipeline remains
// model-agnostic: it discovers concrete models through DenseLearningRuleFactory
// and asks each model to declare both rule-level parameters and per-synapse
// runtime fields.
class IDenseLearningRuleModel {
public:
    virtual ~IDenseLearningRuleModel() {}

    // FactoryModelId is the stable id stored in compiled synapse arrays and
    // later used by the CUDA dispatch path.
    virtual int FactoryModelId() const = 0;
    // CanonicalName is used for diagnostics and generated model specs.
    virtual const char* CanonicalName() const = 0;
    // MatchesLegacyName maps a legacy LearningRuleDescription::RuleName to this
    // dense implementation. Aliases are handled inside the concrete model.
    virtual bool MatchesLegacyName(const std::string& name) const = 0;
    // Flags describe which learning hooks this model needs. They are compiled
    // once per synapse so kernels can cheaply skip unsupported hook phases.
    virtual unsigned char Flags() const = 0;

    // RuleFields are rule-indexed parameter fields:
    // rule_fields[field][rule_id]. They replace model-specific payload members
    // in DenseLearningRuleInfo and keep parameters owned by the model.
    virtual std::vector<DenseFieldSchema> RuleFields() const = 0;
    // SynapseFields are synapse-indexed runtime fields:
    // learning_fields[field][synapse_id]. These are the fields uploaded to the
    // GPU and consumed by device learning kernels.
    virtual std::vector<DenseFieldSchema> SynapseFields() const = 0;
    // FieldSlotCount and FieldSlots define the device slot order for this
    // model. The factory converts names to field ids, which avoids duplicating
    // fragile enum/name ordering in the main build path.
    virtual int FieldSlotCount() const = 0;
    virtual std::vector<DenseFieldSlotBinding> FieldSlots() const = 0;
    // Validate model-owned metadata during host initialization. This catches
    // extension mistakes before any CUDA learning kernel consumes field ids.
    virtual bool Validate(std::string* reason) const;

    // DefaultRuleInfo contains only model metadata. Concrete parameters must go
    // into RuleFields through FillRuleFields.
    virtual DenseLearningRuleInfo DefaultRuleInfo() const = 0;
    // Parse one legacy rule instance into the rule-indexed field table.
    // rule_id is the index of the legacy LearningRuleDescription in the global
    // learning rule list.
    virtual bool FillRuleFields(int rule_id,
                                const LearningRuleDescription& source,
                                DenseLearningRuleFieldTable* rule_fields,
                                std::string* reason) const = 0;

    // Expand rule-indexed parameters into synapse-indexed runtime fields. This
    // step intentionally duplicates only the values needed by CUDA kernels so
    // the runtime can update weights without rule-table lookups.
    virtual bool FillSynapseFields(int synapse_count,
                                   const std::vector<int>& plastic_rule_ids,
                                   const std::vector<int>& plastic_model_ids,
                                   const std::vector<int>& trigger_rule_ids,
                                   const std::vector<int>& trigger_model_ids,
                                   const std::vector<float>& max_weights,
                                   const std::vector<DenseLearningRuleInfo>& rule_table,
                                   const DenseLearningRuleFieldTable& rule_fields,
                                   DenseLearningHostFieldTable* fields,
                                   std::string* reason) const = 0;

    // Return the pre-spike history window required by models that use buffered
    // activity. Models that do not need a learning spike ring return 0.
    virtual float SpikeBufferWindowMs(int rule_id,
                                      const DenseLearningRuleFieldTable& rule_fields) const {
        (void)rule_id;
        (void)rule_fields;
        return 0.0f;
    }
};

}  // namespace npgr

#endif
