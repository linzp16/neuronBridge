#include "learning_rule/LearningRuleCatalog.h"
#include <iostream>
#include <stdexcept>
#include <utility>

using npgr::LearningRuleCatalog;
using npgr::LearningRuleCatalogEntry;

void Require(bool ok) {
    if (!ok) throw std::runtime_error("Learning rule catalog assertion failed");
}

void Reject(std::vector<LearningRuleCatalogEntry> entries) {
    try { LearningRuleCatalog catalog(std::move(entries)); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Invalid catalog accepted");
}

int main() {
    try {
        const auto& original = LearningRuleCatalog::Instance();
        Require(original.Resolve("R-STDP") == original.Resolve("R_STDP"));
        Require(original.ResolveCanonicalName("missing") == "missing");
        Require(!original.IsDenseGpuSupported("missing"));
        Require(original.DenseGpuSupportedRuleNames() == std::vector<std::string>{
            "STDP", "R_STDP", "AdditiveKernalChange", "CerebullarLearningRule",
            "CustomPairStdpV1", "CustomRStdpPersistentV1", "CustomRStdpV1"});
        const auto* pair_rule = original.Resolve("CustomPairStdpV1");
        Require(pair_rule != nullptr);
        Require(pair_rule->category == npgr::LearningRuleCategory::PairBased);
        Require(pair_rule->trigger_semantics == npgr::LearningRuleTriggerSemantics::None);
        const auto* reward_rule = original.Resolve("CustomRStdpV1");
        Require(reward_rule != nullptr);
        Require(reward_rule->category == npgr::LearningRuleCategory::RewardModulated);
        Require(reward_rule->trigger_semantics ==
                npgr::LearningRuleTriggerSemantics::RewardPunishment);
        LearningRuleCatalog extended({{"TestRule", {"TestAlias"}, false}});
        Require(extended.ResolveCanonicalName("TestAlias") == "TestRule");
        Require(!extended.IsDenseGpuSupported("TestRule"));
        Require(extended.DenseGpuSupportedRuleNames() == std::vector<std::string>{
            "STDP", "R_STDP", "AdditiveKernalChange", "CerebullarLearningRule"});
        Require(original.Resolve("TestRule") == nullptr);
        Reject({{"", {}, false}});
        Reject({{"Test", {""}, false}});
        Reject({{"STDP", {}, false}});
        Reject({{"R-STDP", {}, false}});
        Reject({{"Test", {"STDP"}, false}});
        Reject({{"Test", {"Test"}, false}});
        Reject({{"Test", {"Alias", "Alias"}, false}});
        Reject({{"Test", {}, false}, {"Test", {}, false}});
        Reject({{"Test", {"Alias"}, false}, {"Other", {"Alias"}, false}});
        std::cout << "PASS: builtins, aliases, isolated extension, 9 conflict cases\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
