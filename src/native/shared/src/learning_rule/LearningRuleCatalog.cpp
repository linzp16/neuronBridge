#include "learning_rule/LearningRuleCatalog.h"
#include "learning_rule/GeneratedLearningRuleCatalog.h"
#include <iterator>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace npgr {
LearningRuleCatalog& LearningRuleCatalog::Instance() {
    static LearningRuleCatalog catalog;
    return catalog;
}

LearningRuleCatalog::LearningRuleCatalog()
    : LearningRuleCatalog(GeneratedLearningRuleCatalogEntries()) {}

LearningRuleCatalog::LearningRuleCatalog(std::vector<LearningRuleCatalogEntry> additional_entries) {
        entries_ = {
            {"STDP",
             {},
             true,
             LearningRuleCategory::PairBased,
             LearningRuleTriggerSemantics::None},
            {"R_STDP",
             {"R-STDP"},
             true,
             LearningRuleCategory::RewardModulated,
             LearningRuleTriggerSemantics::RewardPunishment},
            {"AdditiveKernalChange",
             {},
             true,
             LearningRuleCategory::TriggerBased,
             LearningRuleTriggerSemantics::GenericTrigger},
            {"CerebullarLearningRule",
             {},
             true,
             LearningRuleCategory::TriggerBased,
             LearningRuleTriggerSemantics::GenericTrigger},
        };
    entries_.insert(entries_.end(), std::make_move_iterator(additional_entries.begin()),
                    std::make_move_iterator(additional_entries.end()));
    std::unordered_set<std::string> names;
    const auto validate = [&names](const std::string& name) {
        if (name.empty() || !names.insert(name).second) {
            throw std::invalid_argument("Empty or duplicate learning rule catalog name: " + name);
        }
    };
    for (const auto& entry : entries_) {
        validate(entry.canonical_name);
        for (const auto& alias : entry.aliases) validate(alias);
    }
}

const LearningRuleCatalogEntry* LearningRuleCatalog::Resolve(const std::string& name) const {
    for (const auto& entry : entries_) {
        if (entry.canonical_name == name ||
            std::find(entry.aliases.begin(), entry.aliases.end(), name) != entry.aliases.end()) return &entry;
    }
    return nullptr;
}

std::string LearningRuleCatalog::ResolveCanonicalName(const std::string& name) const {
    const auto* entry = Resolve(name);
    return entry ? entry->canonical_name : name;
}

bool LearningRuleCatalog::IsDenseGpuSupported(const std::string& name) const {
    const auto* entry = Resolve(name);
    return entry && entry->dense_gpu_supported;
}

std::vector<std::string> LearningRuleCatalog::DenseGpuSupportedRuleNames() const {
    std::vector<std::string> names;
    for (const auto& entry : entries_) {
        if (entry.dense_gpu_supported) names.push_back(entry.canonical_name);
    }
    return names;
}
}  // namespace npgr
