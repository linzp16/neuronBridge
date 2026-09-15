#ifndef NPGR_DENSESTDPLEARNINGMODEL_H
#define NPGR_DENSESTDPLEARNINGMODEL_H

#include "dense_subnetwork/learning/DenseBuiltinLearningRuleModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this learning rule and define the compact field-id
// order consumed by the CUDA pre/post entries.
enum DenseStdpFieldSlot {
    kDenseStdpApre = 0,
    kDenseStdpApost,
    kDenseStdpLastUpdateStep,
    kDenseStdpMaxLtp,
    kDenseStdpMaxLtd,
    kDenseStdpInvLtpTau,
    kDenseStdpInvLtdTau,
    kDenseStdpMaxWeight,
    kDenseStdpFieldCount,
};

class DenseStdpLearningModel : public DenseLearningRuleModelBase {
public:
    int FactoryModelId() const override { return kDenseLearningStdpModelId; }
    const char* CanonicalName() const override { return "STDP"; }
    bool MatchesLegacyName(const std::string& name) const override { return name == "STDP"; }
    unsigned char Flags() const override {
        return static_cast<unsigned char>(kDenseLearningUsesPre | kDenseLearningUsesPost);
    }
    std::vector<DenseFieldSchema> RuleFields() const override {
        // Rule fields are compact per-rule parameters parsed from the legacy
        // LearningRuleDescription. They are not uploaded directly to kernels.
        return {
            {"stdp.rule.max_ltp", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.92f},
            {"stdp.rule.ltp_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 16.8f},
            {"stdp.rule.max_ltd", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.53f},
            {"stdp.rule.ltd_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 33.1f},
        };
    }
    std::vector<DenseFieldSchema> SynapseFields() const override {
        // Synapse fields are the GPU-facing STDP state and parameters. Timing
        // constants are expanded from rule fields so each synapse kernel can
        // read field[synapse_id] without chasing rule indirection.
        return {
            {"stdp.apre", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"stdp.apost", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"stdp.last_update_step", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 0},
            {"stdp.max_ltp", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"stdp.max_ltd", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"stdp.inv_ltp_tau", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"stdp.inv_ltd_tau", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"stdp.max_weight", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
        };
    }
    int FieldSlotCount() const override { return kDenseStdpFieldCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kDenseStdpApre, "stdp.apre"},
            {kDenseStdpApost, "stdp.apost"},
            {kDenseStdpLastUpdateStep, "stdp.last_update_step"},
            {kDenseStdpMaxLtp, "stdp.max_ltp"},
            {kDenseStdpMaxLtd, "stdp.max_ltd"},
            {kDenseStdpInvLtpTau, "stdp.inv_ltp_tau"},
            {kDenseStdpInvLtdTau, "stdp.inv_ltd_tau"},
            {kDenseStdpMaxWeight, "stdp.max_weight"},
        };
    }
    bool FillRuleFields(int rule_id,
                        const LearningRuleDescription& source,
                        DenseLearningRuleFieldTable* rule_fields,
                        std::string*) const override {
        // Parse once per legacy rule instance directly into rule-indexed
        // storage. No model-specific parameter object is retained.
        WriteFloat(rule_fields,
                   "stdp.rule.max_ltp",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "Max_LTP", 0.92f));
        WriteFloat(rule_fields,
                   "stdp.rule.ltp_tau",
                   rule_id,
                   ClampPositive(ReadLegacyFloat(source.RuleParameter, "LTP_tau", 16.8f)));
        WriteFloat(rule_fields,
                   "stdp.rule.max_ltd",
                   rule_id,
                   ReadLegacyFloat(source.RuleParameter, "Max_LTD", 0.53f));
        WriteFloat(rule_fields,
                   "stdp.rule.ltd_tau",
                   rule_id,
                   ClampPositive(ReadLegacyFloat(source.RuleParameter, "LTD_tau", 33.1f)));
        return true;
    }
    bool FillSynapseFields(int synapse_count,
                           const std::vector<int>& plastic_rule_ids,
                           const std::vector<int>& plastic_model_ids,
                           const std::vector<int>&,
                           const std::vector<int>&,
                           const std::vector<float>& max_weights,
                           const std::vector<DenseLearningRuleInfo>&,
                           const DenseLearningRuleFieldTable& rule_fields,
                           DenseLearningHostFieldTable* fields,
                           std::string*) const override {
        // Expand only synapses bound to this model. max_weight is connection
        // specific, so it is written for every synapse slot in this model table.
        for (int synapse_id = 0; synapse_id < synapse_count; ++synapse_id) {
            const std::size_t index = static_cast<std::size_t>(synapse_id);
            if (plastic_model_ids[index] == FactoryModelId()) {
                const int rule_id = plastic_rule_ids[index];
                const float ltp_tau = ReadFloat(rule_fields, "stdp.rule.ltp_tau", rule_id, 16.8f);
                const float ltd_tau = ReadFloat(rule_fields, "stdp.rule.ltd_tau", rule_id, 33.1f);
                WriteFloat(fields, "stdp.max_ltp", synapse_id,
                           ReadFloat(rule_fields, "stdp.rule.max_ltp", rule_id, 0.92f));
                WriteFloat(fields, "stdp.max_ltd", synapse_id,
                           ReadFloat(rule_fields, "stdp.rule.max_ltd", rule_id, 0.53f));
                WriteFloat(fields, "stdp.inv_ltp_tau", synapse_id, 1.0f / std::max(1.0e-6f, ltp_tau));
                WriteFloat(fields, "stdp.inv_ltd_tau", synapse_id, 1.0f / std::max(1.0e-6f, ltd_tau));
            }
            WriteFloat(fields, "stdp.max_weight", synapse_id, max_weights[index]);
        }
        return true;
    }
};

}  // namespace
}  // namespace npgr

#endif
