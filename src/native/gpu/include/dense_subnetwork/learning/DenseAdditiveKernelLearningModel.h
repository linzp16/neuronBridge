#ifndef NPGR_DENSEADDITIVEKERNELLEARNINGMODEL_H
#define NPGR_DENSEADDITIVEKERNELLEARNINGMODEL_H

#include "dense_subnetwork/learning/DenseBuiltinLearningRuleModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this learning rule and define the compact field-id
// order consumed by the CUDA pre/trigger entries.
enum DenseAdditiveKernelFieldSlot {
    kDenseAdditivePreTrace = 0,
    kDenseAdditiveLastUpdateStep,
    kDenseAdditiveA1Pre,
    kDenseAdditiveA2PrePre,
    kDenseAdditiveInvLtpTau,
    kDenseAdditiveMaxWeight,
    kDenseAdditiveFieldCount,
};

class DenseAdditiveKernelLearningModel : public DenseLearningRuleModelBase {
public:
    int FactoryModelId() const override { return kDenseLearningAdditiveKernelModelId; }
    const char* CanonicalName() const override { return "AdditiveKernalChange"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "AdditiveKernalChange";
    }
    unsigned char Flags() const override {
        return static_cast<unsigned char>(
            kDenseLearningUsesPre | kDenseLearningUsesTrigger |
            kDenseLearningUsesTriggerRouting);
    }
    std::vector<DenseFieldSchema> RuleFields() const override {
        // Kernel-shape constants are rule-level. The pre_trace and last update
        // fields remain per synapse because they evolve during simulation.
        return {
            {"additive.rule.a1pre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"additive.rule.a2prepre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"additive.rule.ltp_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 16.8f},
        };
    }
    std::vector<DenseFieldSchema> SynapseFields() const override {
        return {
            {"additive.pre_trace", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"additive.last_update_step", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 0},
            {"additive.a1pre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"additive.a2prepre", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"additive.inv_ltp_tau", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"additive.max_weight", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
        };
    }
    int FieldSlotCount() const override { return kDenseAdditiveFieldCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kDenseAdditivePreTrace, "additive.pre_trace"},
            {kDenseAdditiveLastUpdateStep, "additive.last_update_step"},
            {kDenseAdditiveA1Pre, "additive.a1pre"},
            {kDenseAdditiveA2PrePre, "additive.a2prepre"},
            {kDenseAdditiveInvLtpTau, "additive.inv_ltp_tau"},
            {kDenseAdditiveMaxWeight, "additive.max_weight"},
        };
    }
    bool FillRuleFields(int rule_id,
                        const LearningRuleDescription& source,
                        DenseLearningRuleFieldTable* rule_fields,
                        std::string*) const override {
        WriteFloat(rule_fields,
                   "additive.rule.a1pre",
                   rule_id,
                   ReadLegacyFloatAny(source.RuleParameter, {"a1pre", "fixweightchange"}, 1.0f));
        WriteFloat(rule_fields,
                   "additive.rule.a2prepre",
                   rule_id,
                   ReadLegacyFloatAny(source.RuleParameter, {"a2prepre", "kernalchange"}, 1.0f));
        WriteFloat(rule_fields,
                   "additive.rule.ltp_tau",
                   rule_id,
                   ClampPositive(ReadLegacyFloat(source.RuleParameter, "LTP_tau", 16.8f)));
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
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            const std::size_t index = static_cast<std::size_t>(synapse_id);
            int rule_id = -1;
            if (plastic_model_ids[index] == FactoryModelId()) {
                rule_id = plastic_rule_ids[index];
            } else if (trigger_model_ids[index] == FactoryModelId()) {
                rule_id = trigger_rule_ids[index];
            }
            if (rule_id >= 0) {
                const float ltp_tau = ReadFloat(rule_fields, "additive.rule.ltp_tau", rule_id, 16.8f);
                WriteFloat(fields, "additive.a1pre", synapse_id,
                           ReadFloat(rule_fields, "additive.rule.a1pre", rule_id, 1.0f));
                WriteFloat(fields, "additive.a2prepre", synapse_id,
                           ReadFloat(rule_fields, "additive.rule.a2prepre", rule_id, 1.0f));
                WriteFloat(fields, "additive.inv_ltp_tau", synapse_id, 1.0f / std::max(1.0e-6f, ltp_tau));
            }
            WriteFloat(fields, "additive.max_weight", synapse_id, max_weights[index]);
        }
        return true;
    }
};

}  // namespace
}  // namespace npgr

#endif
