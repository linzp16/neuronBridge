#include "LearningRuleModelFactory.h"
#include <iostream>
#include "learning_rule/LearningRuleCatalog.h"
#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomGeneratedLearningRules.h"
#endif
#include "../source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "../source_file_realtime_v1_async/LearningRule/inc/Rule/R_STDP.h"
#include "../source_file_realtime_v1_async/LearningRule/inc/Rule/STDP.h"
#include "../source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "../source_file_realtime_v1_async/LearningRule/inc/Rule/CerebullarLearningRule.h"


LearningRule* LearningRuleModelFactory::createLearningRuleModel(LearningRuleDescription lrDescription) {
			lrDescription.RuleName = npgr::LearningRuleCatalog::Instance().ResolveCanonicalName(lrDescription.RuleName);
#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#define NPGR_CUSTOM_LEGACY_RULE(name, type) if (lrDescription.RuleName == name) return new type(lrDescription.RuleParameter);
#include "neuronbridge_codegen/CustomLegacyLearningRuleRegistry.inc"
#undef NPGR_CUSTOM_LEGACY_RULE
#endif
			if (lrDescription.RuleName == "STDP") {
				LearningRule* Rule_Model = new STDP(lrDescription.RuleParameter);
				return Rule_Model;
			}
			else if (lrDescription.RuleName == "R_STDP" || lrDescription.RuleName == "R-STDP") {
				LearningRule* Rule_Model = new R_STDP(lrDescription.RuleParameter);
				return Rule_Model;
			}
			else if (lrDescription.RuleName == "CerebullarLearningRule") {
				LearningRule* Rule_Model = new CerebullarLearningRule(lrDescription.RuleParameter);
				return Rule_Model;
			}
			else {
				std::cout << "Unknown learning rule type: " << lrDescription.RuleName << std::endl;
				return 0;
			}
		}
