#ifndef NPGR_DENSERSTDPLEARNINGMODEL_H
#define NPGR_DENSERSTDPLEARNINGMODEL_H

#include "dense_subnetwork/learning/DenseBuiltinLearningRuleModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this learning rule and define the compact field-id
// order consumed by the CUDA pre/post/trigger entries.
enum DenseRStdpFieldSlot {
    kDenseRStdpApre = 0,
    kDenseRStdpApost,
    kDenseRStdpEligibility,
    kDenseRStdpLastUpdateStep,
    kDenseRStdpMaxLtp,
    kDenseRStdpMaxLtd,
    kDenseRStdpInvLtpTau,
    kDenseRStdpInvLtdTau,
    kDenseRStdpRewardFactor,
    kDenseRStdpPunishmentFactor,
    kDenseRStdpClearEligibility,
    kDenseRStdpMaxWeight,
    kDenseRStdpFieldCount,
};

class DenseRStdpLearningModel : public DenseLearningRuleModelBase {
public:
    int FactoryModelId() const override { return kDenseLearningRStdpModelId; }
    const char* CanonicalName() const override { return "R_STDP"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "R_STDP" || name == "R-STDP";
    }
    unsigned char Flags() const override {
        return static_cast<unsigned char>(
            kDenseLearningUsesPre | kDenseLearningUsesPost |
            kDenseLearningUsesTrigger | kDenseLearningUsesTriggerRouting);
    }
    std::vector<DenseFieldSchema> RuleFields() const override {
        // R-STDP stores STDP timing parameters plus reinforcement behavior in
        // the rule table; eligibility traces remain synapse-local runtime state.
        return {
            {"rstdp.rule.max_ltp", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.92f},
            {"rstdp.rule.ltp_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 16.8f},
            {"rstdp.rule.max_ltd", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.53f},
            {"rstdp.rule.ltd_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 33.1f},
            {"rstdp.rule.reward_factor", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"rstdp.rule.punishment_factor", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -1.0f},
            {"rstdp.rule.clear_eligibility_after_trigger", DenseFieldStorage::UInt8, DenseFieldRole::Parameter, 0.0f, 0, 1u},
        };
    }
    std::vector<DenseFieldSchema> SynapseFields() const override {
        return {
            {"rstdp.apre", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"rstdp.apost", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"rstdp.eligibility", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"rstdp.last_update_step", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 0},
            {"rstdp.max_ltp", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"rstdp.max_ltd", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"rstdp.inv_ltp_tau", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"rstdp.inv_ltd_tau", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"rstdp.reward_factor", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"rstdp.punishment_factor", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"rstdp.clear_eligibility_after_trigger", DenseFieldStorage::UInt8, DenseFieldRole::Parameter, 0.0f, 0, 1u},
            {"rstdp.max_weight", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
        };
    }
    int FieldSlotCount() const override { return kDenseRStdpFieldCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kDenseRStdpApre, "rstdp.apre"},
            {kDenseRStdpApost, "rstdp.apost"},
            {kDenseRStdpEligibility, "rstdp.eligibility"},
            {kDenseRStdpLastUpdateStep, "rstdp.last_update_step"},
            {kDenseRStdpMaxLtp, "rstdp.max_ltp"},
            {kDenseRStdpMaxLtd, "rstdp.max_ltd"},
            {kDenseRStdpInvLtpTau, "rstdp.inv_ltp_tau"},
            {kDenseRStdpInvLtdTau, "rstdp.inv_ltd_tau"},
            {kDenseRStdpRewardFactor, "rstdp.reward_factor"},
            {kDenseRStdpPunishmentFactor, "rstdp.punishment_factor"},
            {kDenseRStdpClearEligibility, "rstdp.clear_eligibility_after_trigger"},
            {kDenseRStdpMaxWeight, "rstdp.max_weight"},
        };
    }
    bool FillRuleFields(int rule_id,
                        const LearningRuleDescription& source,
                        DenseLearningRuleFieldTable* rule_fields,
                        std::string*) const override {
        WriteFloat(rule_fields,
                   "rstdp.rule.max_ltp",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "Max_LTP", 0.92f));
        WriteFloat(rule_fields,
                   "rstdp.rule.ltp_tau",
                   rule_id,
                   ClampPositive(ReadLegacyFloat(source.RuleParameter, "LTP_tau", 16.8f)));
        WriteFloat(rule_fields,
                   "rstdp.rule.max_ltd",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "Max_LTD", 0.53f));
        WriteFloat(rule_fields,
                   "rstdp.rule.ltd_tau",
                   rule_id,
                   ClampPositive(ReadLegacyFloat(source.RuleParameter, "LTD_tau", 33.1f)));
        WriteFloat(rule_fields,
                   "rstdp.rule.reward_factor",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "RewardFactor", 1.0f));
        WriteFloat(rule_fields,
                   "rstdp.rule.punishment_factor",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "PunishmentFactor", -1.0f));
        WriteByte(rule_fields,
                  "rstdp.rule.clear_eligibility_after_trigger",
                  rule_id,
                  ReadLegacyBool(source.RuleParameter, "ClearEligibilityAfterTrigger", true) ? 1u : 0u);
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
        // R-STDP can appear either as the plastic rule or as the trigger rule.
        // The same model-owned rule fields feed both cases.
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            const std::size_t index = static_cast<std::size_t>(synapse_id);
            int rule_id = -1;
            if (plastic_model_ids[index] == FactoryModelId()) {
                rule_id = plastic_rule_ids[index];
            } else if (trigger_model_ids[index] == FactoryModelId()) {
                rule_id = trigger_rule_ids[index];
            }
            if (rule_id >= 0) {
                const float ltp_tau = ReadFloat(rule_fields, "rstdp.rule.ltp_tau", rule_id, 16.8f);
                const float ltd_tau = ReadFloat(rule_fields, "rstdp.rule.ltd_tau", rule_id, 33.1f);
                WriteFloat(fields, "rstdp.max_ltp", synapse_id,
                           ReadFloat(rule_fields, "rstdp.rule.max_ltp", rule_id, 0.92f));
                WriteFloat(fields, "rstdp.max_ltd", synapse_id,
                           ReadFloat(rule_fields, "rstdp.rule.max_ltd", rule_id, 0.53f));
                WriteFloat(fields, "rstdp.inv_ltp_tau", synapse_id, 1.0f / std::max(1.0e-6f, ltp_tau));
                WriteFloat(fields, "rstdp.inv_ltd_tau", synapse_id, 1.0f / std::max(1.0e-6f, ltd_tau));
                WriteFloat(fields, "rstdp.reward_factor", synapse_id,
                           ReadFloat(rule_fields, "rstdp.rule.reward_factor", rule_id, 1.0f));
                WriteFloat(fields, "rstdp.punishment_factor", synapse_id,
                           ReadFloat(rule_fields, "rstdp.rule.punishment_factor", rule_id, -1.0f));
                WriteByte(fields,
                          "rstdp.clear_eligibility_after_trigger",
                          synapse_id,
                          ReadByte(rule_fields, "rstdp.rule.clear_eligibility_after_trigger", rule_id, 1u));
            }
            WriteFloat(fields, "rstdp.max_weight", synapse_id, max_weights[index]);
        }
        return true;
    }
};

}  // namespace
}  // namespace npgr

#endif
