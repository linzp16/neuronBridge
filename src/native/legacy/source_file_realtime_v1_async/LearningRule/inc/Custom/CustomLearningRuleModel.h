#pragma once

#include "source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"

// Shared ownership and event routing only; concrete rules own their equations.
class CustomLearningRuleModel : public LearningRule {
public:
    CustomLearningRuleModel(bool post, bool trigger) : post_(post), trigger_(trigger) {}
    bool ImplementPostSynaptic() override { return post_; }
    bool ImplementTriggerSynaptic() override { return trigger_; }
protected:
    template<class Action>
    void ForEachPostConnection(Neuron* neuron, Action&& action) {
        const int rule = LearningRuleID;
        for (int i = 0; i < neuron->PostSynapticLearning_Number[rule]; ++i) {
            const int index = neuron->IndexOfInputLearningIndex[0][rule][i];
            if (index < 0) continue;
            action(neuron->PostSynapticLearning[rule][i], index);
        }
    }
    template<class Action>
    void ForEachPostTriggerConnection(Neuron* neuron, Action&& action) {
        const int rule = LearningRuleID;
        for (int i = 0; i < neuron->TriggerAndPostSynapticLearning_Number[rule]; ++i) {
            const int index = neuron->IndexOfInputLearningIndex[2][rule][i];
            if (index < 0) continue;
            auto* connection = neuron->TriggerAndPostSynapticLearning[rule][i];
            if (!connection->TriggerLearning) action(connection, index);
        }
    }
private:
    bool post_;
    bool trigger_;
};
