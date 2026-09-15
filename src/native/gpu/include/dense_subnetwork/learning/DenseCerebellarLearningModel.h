#ifndef NPGR_DENSECEREBELLARLEARNINGMODEL_H
#define NPGR_DENSECEREBELLARLEARNINGMODEL_H

#include "dense_subnetwork/learning/DenseBuiltinLearningRuleModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this learning rule and define the compact field-id
// order consumed by the CUDA pre/trigger entries.
enum DenseCerebellarFieldSlot {
    kDenseCerebellarA1Pre = 0,
    kDenseCerebellarA2PrePre,
    kDenseCerebellarInitPos,
    kDenseCerebellarMaxPos,
    kDenseCerebellarKernelStepSize,
    kDenseCerebellarInvKernelStepSize,
    kDenseCerebellarMaxTimeMeasured,
    kDenseCerebellarMaxWeight,
    kDenseCerebellarMinWeight,
    kDenseCerebellarRandomSeed,
    kDenseCerebellarRuleId,
    kDenseCerebellarFieldCount,
};

class DenseCerebellarLearningModel : public DenseLearningRuleModelBase {
public:
    int FactoryModelId() const override { return kDenseLearningCerebellarModelId; }
    const char* CanonicalName() const override { return "CerebullarLearningRule"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "CerebullarLearningRule";
    }
    unsigned char Flags() const override {
        return static_cast<unsigned char>(
            kDenseLearningUsesPre | kDenseLearningUsesTrigger |
            kDenseLearningUsesTriggerRouting | kDenseLearningUsesKernelTable);
    }
    std::vector<DenseFieldSchema> RuleFields() const override {
        // Cerebellar rule fields own the kernel support parameters used both to
        // initialize runtime fields and to size the learning spike ring.
        return {
            {"cerebellar.rule.a1pre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"cerebellar.rule.a2prepre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"cerebellar.rule.initpos", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.rule.maxpos", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"cerebellar.rule.kernel_step_size", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"cerebellar.rule.min_weight", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.rule.random_seed", DenseFieldStorage::Int32, DenseFieldRole::Parameter, 0.0f, 17},
        };
    }
    std::vector<DenseFieldSchema> SynapseFields() const override {
        return {
            {"cerebellar.a1pre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.a2prepre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.initpos", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.maxpos", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.kernel_step_size", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"cerebellar.inv_kernel_step_size", DenseFieldStorage::Float32, DenseFieldRole::Derived, 1.0f},
            {"cerebellar.max_time_measured", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"cerebellar.max_weight", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.min_weight", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"cerebellar.random_seed", DenseFieldStorage::Int32, DenseFieldRole::Parameter, 0.0f, 17},
            {"cerebellar.rule_id", DenseFieldStorage::Int32, DenseFieldRole::Derived, 0.0f, -1},
        };
    }
    int FieldSlotCount() const override { return kDenseCerebellarFieldCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kDenseCerebellarA1Pre, "cerebellar.a1pre"},
            {kDenseCerebellarA2PrePre, "cerebellar.a2prepre"},
            {kDenseCerebellarInitPos, "cerebellar.initpos"},
            {kDenseCerebellarMaxPos, "cerebellar.maxpos"},
            {kDenseCerebellarKernelStepSize, "cerebellar.kernel_step_size"},
            {kDenseCerebellarInvKernelStepSize, "cerebellar.inv_kernel_step_size"},
            {kDenseCerebellarMaxTimeMeasured, "cerebellar.max_time_measured"},
            {kDenseCerebellarMaxWeight, "cerebellar.max_weight"},
            {kDenseCerebellarMinWeight, "cerebellar.min_weight"},
            {kDenseCerebellarRandomSeed, "cerebellar.random_seed"},
            {kDenseCerebellarRuleId, "cerebellar.rule_id"},
        };
    }
    bool FillRuleFields(int rule_id,
                        const LearningRuleDescription& source,
                        DenseLearningRuleFieldTable* rule_fields,
                        std::string*) const override {
        WriteFloat(rule_fields,
                   "cerebellar.rule.a1pre",
                   rule_id,
                   ReadLegacyFloatAny(source.RuleParameter, {"a1pre", "fixweightchange"}, 1.0f));
        WriteFloat(rule_fields,
                   "cerebellar.rule.a2prepre",
                   rule_id,
                   ReadLegacyFloatAny(source.RuleParameter, {"a2prepre", "kernalchange"}, 1.0f));
        WriteFloat(rule_fields,
                   "cerebellar.rule.initpos",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "initpos", 0.0f));
        WriteFloat(rule_fields,
                   "cerebellar.rule.maxpos",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "maxpos", 1.0f));
        WriteFloat(rule_fields,
                   "cerebellar.rule.kernel_step_size",
                   rule_id,
                   ClampPositive(ReadLegacyFloat(source.RuleParameter, "kernel_step_size", 1.0f)));
        WriteFloat(rule_fields,
                   "cerebellar.rule.min_weight",
                   rule_id,
                   ReadLegacyFloatAny(source.RuleParameter,
                                      {"min_weight", "minimum_weight", "w_gcpc_min"},
                                      0.0f));
        WriteInt(rule_fields,
                 "cerebellar.rule.random_seed",
                 rule_id,
                 ReadLegacyInt(source.RuleParameter, "random_seed", 17));
        return true;
    }
    bool FillSynapseFields(int synapse_count,
                           const std::vector<int>& plastic_rule_ids,
                           const std::vector<int>& plastic_model_ids,
                           const std::vector<int>& trigger_rule_ids,
                           const std::vector<int>& trigger_model_ids,
                           const std::vector<float>& max_weights,
                           const std::vector<DenseLearningRuleInfo>&,
                           const DenseLearningRuleFieldTable& rule_fields,
                           DenseLearningHostFieldTable* fields,
                           std::string*) const override {
        // The CUDA cerebellar update reads expanded per-synapse fields. The
        // expansion also precomputes inverse step size and measured window.
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            const std::size_t index = static_cast<std::size_t>(synapse_id);
            int rule_id = -1;
            if (plastic_model_ids[index] == FactoryModelId()) {
                rule_id = plastic_rule_ids[index];
            } else if (trigger_model_ids[index] == FactoryModelId()) {
                rule_id = trigger_rule_ids[index];
            }
            if (rule_id >= 0) {
                const float initpos =
                    ReadFloat(rule_fields, "cerebellar.rule.initpos", rule_id, 0.0f);
                const float maxpos =
                    std::max(initpos + 1.0e-6f,
                             ReadFloat(rule_fields, "cerebellar.rule.maxpos", rule_id, 1.0f));
                const float kernel_step_size =
                    std::max(1.0e-6f,
                             ReadFloat(rule_fields, "cerebellar.rule.kernel_step_size", rule_id, 1.0f));
                WriteFloat(fields, "cerebellar.a1pre", synapse_id,
                           ReadFloat(rule_fields, "cerebellar.rule.a1pre", rule_id, 1.0f));
                WriteFloat(fields, "cerebellar.a2prepre", synapse_id,
                           ReadFloat(rule_fields, "cerebellar.rule.a2prepre", rule_id, 1.0f));
                WriteFloat(fields, "cerebellar.initpos", synapse_id, initpos);
                WriteFloat(fields, "cerebellar.maxpos", synapse_id, maxpos);
                WriteFloat(fields, "cerebellar.kernel_step_size", synapse_id, kernel_step_size);
                WriteFloat(fields, "cerebellar.inv_kernel_step_size", synapse_id, 1.0f / kernel_step_size);
                WriteFloat(fields,
                           "cerebellar.max_time_measured",
                           synapse_id,
                           CerebellarMaxTimeMeasured(initpos, maxpos, kernel_step_size));
                WriteFloat(fields,
                           "cerebellar.min_weight",
                           synapse_id,
                           ReadFloat(rule_fields, "cerebellar.rule.min_weight", rule_id, 0.0f));
                WriteInt(fields,
                         "cerebellar.random_seed",
                         synapse_id,
                         ReadInt(rule_fields, "cerebellar.rule.random_seed", rule_id, 17));
                WriteInt(fields, "cerebellar.rule_id", synapse_id, rule_id);
            }
            WriteFloat(fields, "cerebellar.max_weight", synapse_id, max_weights[index]);
        }
        return true;
    }
    float SpikeBufferWindowMs(int rule_id,
                              const DenseLearningRuleFieldTable& rule_fields) const override {
        // Finalization asks this method whether a rule needs buffered pre-spike
        // history. Returning a positive window allocates a shared ring bucket.
        const float initpos = ReadFloat(rule_fields, "cerebellar.rule.initpos", rule_id, 0.0f);
        const float maxpos = ReadFloat(rule_fields, "cerebellar.rule.maxpos", rule_id, 1.0f);
        const float kernel_step_size =
            ReadFloat(rule_fields, "cerebellar.rule.kernel_step_size", rule_id, 1.0f);
        return CerebellarMaxTimeMeasured(initpos, maxpos, kernel_step_size);
    }
};


}  // namespace
}  // namespace npgr

#endif
