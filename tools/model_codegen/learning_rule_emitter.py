"""Orchestration and static Dense ABI templates for generated learning rules."""

from pathlib import Path
from string import Template

from .learning_rule_generic import dense_default_injections, render, render_cuda
from .learning_rule_ir import compile_learning_rule
from .model_ir import ModelIrError


def emit_runtime(root: Path, specs):
    outputs = {}
    registry, entries, dense_models, dense_includes, aggregate = [], [], [], [], []
    device_includes = []
    for spec in specs:
        rule = compile_learning_rule(spec)
        class_name = rule.implementation_name
        header, source = render(spec)
        outputs[f"include/neuronbridge_codegen/{class_name}.h"] = header
        outputs[f"src/{class_name}.cpp"] = source
        aggregate.append(f'#include "{class_name}.cpp"')
        registry.append(f'NPGR_CUSTOM_LEGACY_RULE("{class_name}", {class_name})')

        if rule.catalog_category == "pair_based":
            category = "LearningRuleCategory::PairBased"
        elif rule.catalog_category == "reward_modulated":
            category = "LearningRuleCategory::RewardModulated"
        else:
            raise ModelIrError(f"unsupported learning-rule catalog category {rule.catalog_category!r}")
        trigger_semantics = (
            "LearningRuleTriggerSemantics::RewardPunishment"
            if rule.has_trigger else "LearningRuleTriggerSemantics::None"
        )
        entries.append(
            f'{{"{class_name}", {{}}, true, {category}, {trigger_semantics}, false}}'
        )

        if not rule.dense_gpu:
            continue
        dense_class = "Dense" + class_name
        dense_symbol = "kDense" + class_name + "ModelId"
        dense_includes.append(f'#include "neuronbridge_codegen/{dense_class}.h"')
        device_includes.append(
            f'#include "neuronbridge_codegen/{dense_class}DeviceUpdate.cuh"'
        )
        outputs[f"include/neuronbridge_codegen/{dense_class}DeviceUpdate.cuh"] = (
            render_cuda(spec)
        )
        if rule.dense_capability == "pair_stdp_v1":
            outputs[f"include/neuronbridge_codegen/{dense_class}.h"] = (
                PAIR_DENSE_HEADER.substitute(
                    dense_class=dense_class,
                    dense_symbol=dense_symbol,
                    class_name=class_name,
                    default_injections=dense_default_injections(rule),
                )
            )
            trigger_fn = "ApplyDenseLearningNoTriggerDeviceEntry"
        elif rule.dense_capability == "reward_stdp_v1":
            outputs[f"include/neuronbridge_codegen/{dense_class}.h"] = (
                REWARD_DENSE_HEADER.substitute(
                    dense_class=dense_class,
                    dense_symbol=dense_symbol,
                    class_name=class_name,
                    clear_default="true" if rule.clear_default else "false",
                    default_injections=dense_default_injections(rule),
                )
            )
            trigger_fn = f"ApplyDense{class_name}TriggerDeviceEntry"
        else:
            raise ModelIrError("no Dense host lowering for this learning-rule capability")
        dense_models.append(
            f"NPGR_DENSE_LEARNING_MODEL({dense_symbol}, {rule.dense_factory_model_id}, "
            f"{dense_class}, ApplyDense{class_name}PreDeviceEntry, "
            f"ApplyDense{class_name}PostDeviceEntry, {trigger_fn})"
        )

    outputs.update({
        "include/neuronbridge_codegen/CustomGeneratedLearningRules.h":
            "#pragma once\n" + "".join(
                f'#include "neuronbridge_codegen/{spec["implementation_name"]}.h"\n'
                for spec in specs
            ),
        "include/neuronbridge_codegen/CustomGeneratedDenseLearningRules.h":
            "#pragma once\n" + "\n".join(dense_includes) + "\n",
        "include/neuronbridge_codegen/CustomGeneratedDenseLearningDeviceUpdates.cuh":
            "#pragma once\n" + "\n".join(device_includes) + "\n",
        "include/neuronbridge_codegen/CustomDenseLearningModelList.inc":
            "\n".join(dense_models) + "\n",
        "include/neuronbridge_codegen/CustomLegacyLearningRuleRegistry.inc":
            "\n".join(registry) + "\n",
        "src/GeneratedLearningRuleModels.cpp": "\n".join(aggregate) + "\n",
        "src/GeneratedLearningRuleCatalogEntries.cpp":
            '#include "learning_rule/GeneratedLearningRuleCatalog.h"\nnamespace npgr {\n'
            'std::vector<LearningRuleCatalogEntry> GeneratedLearningRuleCatalogEntries() {\n'
            f'    return {{{",".join(entries)}}};\n}}\n}}\n',
    })
    for relative, contents in outputs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")


PAIR_DENSE_HEADER = Template('''#pragma once
#include "dense_subnetwork/learning/DenseStdpLearningModel.h"
namespace npgr {
class $dense_class final : public DenseStdpLearningModel {
public:
    int FactoryModelId() const override { return $dense_symbol; }
    const char* CanonicalName() const override { return "$class_name"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "$class_name";
    }
    bool FillRuleFields(int rule_id,
                        const LearningRuleDescription& source,
                        DenseLearningRuleFieldTable* fields,
                        std::string* reason) const override {
        LearningRuleDescription resolved = source;
$default_injections
        return DenseStdpLearningModel::FillRuleFields(
            rule_id, resolved, fields, reason);
    }
};
}  // namespace npgr
''')


REWARD_DENSE_HEADER = Template('''#pragma once
#include "dense_subnetwork/learning/DenseRStdpLearningModel.h"
namespace npgr {
class $dense_class final : public DenseRStdpLearningModel {
public:
    int FactoryModelId() const override { return $dense_symbol; }
    const char* CanonicalName() const override { return "$class_name"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "$class_name";
    }
    bool FillRuleFields(int rule_id,
                        const LearningRuleDescription& source,
                        DenseLearningRuleFieldTable* fields,
                        std::string* reason) const override {
        LearningRuleDescription resolved = source;
$default_injections
        if (resolved.RuleParameter.find("ClearEligibilityAfterTrigger") ==
            resolved.RuleParameter.end()) {
            resolved.RuleParameter["ClearEligibilityAfterTrigger"] = $clear_default;
        }
        return DenseRStdpLearningModel::FillRuleFields(
            rule_id, resolved, fields, reason);
    }
};
}  // namespace npgr
''')


PAIR_CUDA_HEADER = Template('''#pragma once
#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"
#include "dense_subnetwork/learning/DenseStdpLearningModel.h"

namespace npgr {

__device__ inline void $pre_name(
    DenseLearningDeviceFieldTable fields, const int* field_ids,
    DenseDeviceLearningSpikeBufferTable, int synapse_id, int time_step,
    float dt_ms, float* syn_weight) {
    float* pre_values = LearningFloatField(fields, field_ids[kDenseStdpApre]);
    float* post_values = LearningFloatField(fields, field_ids[kDenseStdpApost]);
    int* last_update = LearningIntField(fields, field_ids[kDenseStdpLastUpdateStep]);
    const float* inv_ltp = LearningFloatField(fields, field_ids[kDenseStdpInvLtpTau]);
    const float* inv_ltd = LearningFloatField(fields, field_ids[kDenseStdpInvLtdTau]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseStdpMaxWeight]);
    DecayPairTrace(synapse_id, time_step, dt_ms, pre_values, post_values,
                   last_update, inv_ltp, inv_ltd);
    float pre_trace = pre_values[synapse_id];
    float post_trace = post_values[synapse_id];
    float weight = syn_weight[synapse_id];
    const float max_ltd =
        LearningFloatField(fields, field_ids[kDenseStdpMaxLtd])[synapse_id];
$on_pre
    pre_values[synapse_id] = pre_trace;
    post_values[synapse_id] = post_trace;
    syn_weight[synapse_id] =
        ClampDenseLearningWeight(weight, max_weight[synapse_id]);
}

__device__ inline void $post_name(
    DenseLearningDeviceFieldTable fields, const int* field_ids, int synapse_id,
    int time_step, float dt_ms, float* syn_weight) {
    float* pre_values = LearningFloatField(fields, field_ids[kDenseStdpApre]);
    float* post_values = LearningFloatField(fields, field_ids[kDenseStdpApost]);
    int* last_update = LearningIntField(fields, field_ids[kDenseStdpLastUpdateStep]);
    const float* inv_ltp = LearningFloatField(fields, field_ids[kDenseStdpInvLtpTau]);
    const float* inv_ltd = LearningFloatField(fields, field_ids[kDenseStdpInvLtdTau]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseStdpMaxWeight]);
    DecayPairTrace(synapse_id, time_step, dt_ms, pre_values, post_values,
                   last_update, inv_ltp, inv_ltd);
    float pre_trace = pre_values[synapse_id];
    float post_trace = post_values[synapse_id];
    float weight = syn_weight[synapse_id];
    const float max_ltp =
        LearningFloatField(fields, field_ids[kDenseStdpMaxLtp])[synapse_id];
$on_post
    pre_values[synapse_id] = pre_trace;
    post_values[synapse_id] = post_trace;
    syn_weight[synapse_id] =
        ClampDenseLearningWeight(weight, max_weight[synapse_id]);
}

}  // namespace npgr
''')


CUDA_HEADER = Template('''#pragma once
#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"
#include "dense_subnetwork/learning/DenseRStdpLearningModel.h"

namespace npgr {

__device__ inline void $pre_name(
    DenseLearningDeviceFieldTable fields, const int* field_ids,
    DenseDeviceLearningSpikeBufferTable, int synapse_id, int time_step,
    float dt_ms, float*) {
    float* apre_values = LearningFloatField(fields, field_ids[kDenseRStdpApre]);
    float* apost_values = LearningFloatField(fields, field_ids[kDenseRStdpApost]);
    float* eligibility_values =
        LearningFloatField(fields, field_ids[kDenseRStdpEligibility]);
    int* last_update =
        LearningIntField(fields, field_ids[kDenseRStdpLastUpdateStep]);
    const float* inv_ltp =
        LearningFloatField(fields, field_ids[kDenseRStdpInvLtpTau]);
    const float* inv_ltd =
        LearningFloatField(fields, field_ids[kDenseRStdpInvLtdTau]);
    DecayPairTrace(synapse_id, time_step, dt_ms, apre_values, apost_values,
                   last_update, inv_ltp, inv_ltd);
    float pre_trace = apre_values[synapse_id];
    float post_trace = apost_values[synapse_id];
    float eligibility = eligibility_values[synapse_id];
    const float max_ltd =
        LearningFloatField(fields, field_ids[kDenseRStdpMaxLtd])[synapse_id];
$on_pre
    apre_values[synapse_id] = pre_trace;
    apost_values[synapse_id] = post_trace;
    eligibility_values[synapse_id] = eligibility;
}

__device__ inline void $post_name(
    DenseLearningDeviceFieldTable fields, const int* field_ids, int synapse_id,
    int time_step, float dt_ms, float*) {
    float* apre_values = LearningFloatField(fields, field_ids[kDenseRStdpApre]);
    float* apost_values = LearningFloatField(fields, field_ids[kDenseRStdpApost]);
    float* eligibility_values =
        LearningFloatField(fields, field_ids[kDenseRStdpEligibility]);
    int* last_update =
        LearningIntField(fields, field_ids[kDenseRStdpLastUpdateStep]);
    const float* inv_ltp =
        LearningFloatField(fields, field_ids[kDenseRStdpInvLtpTau]);
    const float* inv_ltd =
        LearningFloatField(fields, field_ids[kDenseRStdpInvLtdTau]);
    DecayPairTrace(synapse_id, time_step, dt_ms, apre_values, apost_values,
                   last_update, inv_ltp, inv_ltd);
    float pre_trace = apre_values[synapse_id];
    float post_trace = apost_values[synapse_id];
    float eligibility = eligibility_values[synapse_id];
    const float max_ltp =
        LearningFloatField(fields, field_ids[kDenseRStdpMaxLtp])[synapse_id];
$on_post
    apre_values[synapse_id] = pre_trace;
    apost_values[synapse_id] = post_trace;
    eligibility_values[synapse_id] = eligibility;
}

__device__ inline void $trigger_name(
    DenseLearningDeviceFieldTable fields, const int* field_ids,
    DenseDeviceLearningSpikeBufferTable, int trigger_synapse_id,
    unsigned char trigger_type, DenseDeviceLearningTriggerRouteTable routes,
    int time_step, float dt_ms, float* syn_weight) {
    float* apre_values = LearningFloatField(fields, field_ids[kDenseRStdpApre]);
    float* apost_values = LearningFloatField(fields, field_ids[kDenseRStdpApost]);
    float* eligibility_values =
        LearningFloatField(fields, field_ids[kDenseRStdpEligibility]);
    int* last_update =
        LearningIntField(fields, field_ids[kDenseRStdpLastUpdateStep]);
    const float* inv_ltp =
        LearningFloatField(fields, field_ids[kDenseRStdpInvLtpTau]);
    const float* inv_ltd =
        LearningFloatField(fields, field_ids[kDenseRStdpInvLtdTau]);
    const float* reward =
        LearningFloatField(fields, field_ids[kDenseRStdpRewardFactor]);
    const float* punishment =
        LearningFloatField(fields, field_ids[kDenseRStdpPunishmentFactor]);
    const unsigned char* clear_values =
        LearningByteField(fields, field_ids[kDenseRStdpClearEligibility]);
    const float* max_weight =
        LearningFloatField(fields, field_ids[kDenseRStdpMaxWeight]);
    const float trigger_factor = trigger_type == 1u
        ? punishment[trigger_synapse_id] : reward[trigger_synapse_id];
    const int start = routes.start[trigger_synapse_id];
    const int count = routes.count[trigger_synapse_id];
    for (int offset = 0; offset < count; ++offset) {
        const int synapse_id = routes.synapse_ids[start + offset];
        DecayPairTrace(synapse_id, time_step, dt_ms, apre_values, apost_values,
                       last_update, inv_ltp, inv_ltd);
        float pre_trace = apre_values[synapse_id];
        float post_trace = apost_values[synapse_id];
        float eligibility = eligibility_values[synapse_id];
        float weight = syn_weight[synapse_id];
$on_trigger
        syn_weight[synapse_id] =
            ClampDenseLearningWeight(weight, max_weight[synapse_id]);
        if (clear_values[synapse_id] != 0u) eligibility = 0.0f;
        apre_values[synapse_id] = pre_trace;
        apost_values[synapse_id] = post_trace;
        eligibility_values[synapse_id] = eligibility;
    }
}

}  // namespace npgr
''')
