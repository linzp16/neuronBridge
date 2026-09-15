#ifndef NPGR_LEARNING_RULE_CATALOG_H
#define NPGR_LEARNING_RULE_CATALOG_H

#include <algorithm>
#include <string>
#include <vector>

namespace npgr {

enum class LearningRuleCategory {
    PairBased,
    RewardModulated,
    TriggerBased,
};

enum class LearningRuleTriggerSemantics {
    None,
    RewardPunishment,
    GenericTrigger,
};

struct LearningRuleCatalogEntry {
    std::string canonical_name;
    std::vector<std::string> aliases;
    bool dense_gpu_supported = false;
    LearningRuleCategory category = LearningRuleCategory::PairBased;
    LearningRuleTriggerSemantics trigger_semantics = LearningRuleTriggerSemantics::None;
    bool public_user_model = true;
};

class LearningRuleCatalog {
public:
    static LearningRuleCatalog& Instance();
    explicit LearningRuleCatalog(std::vector<LearningRuleCatalogEntry> additional_entries);
    const LearningRuleCatalogEntry* Resolve(const std::string& name) const;
    std::string ResolveCanonicalName(const std::string& name) const;
    bool IsDenseGpuSupported(const std::string& name) const;
    std::vector<std::string> DenseGpuSupportedRuleNames() const;
private:
    LearningRuleCatalog();
    std::vector<LearningRuleCatalogEntry> entries_;
};
}  // namespace npgr
#endif
