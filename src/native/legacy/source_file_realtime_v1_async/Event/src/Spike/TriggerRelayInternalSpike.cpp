#include "../source_file_realtime_v1_async/Event/inc/Spike/TriggerRelayInternalSpike.h"

#include "../source_file_realtime_v1_async/Event/inc/Spike/PropogatedSpike.h"
#include "../source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "../source_file_realtime_v1_async/Neuron/inc/Neuron.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"

TriggerRelayInternalSpike::TriggerRelayInternalSpike() : InternalSpike() {}

TriggerRelayInternalSpike::TriggerRelayInternalSpike(Neuron* neuron, int time, int queue_index)
    : InternalSpike(neuron, time, queue_index) {}

TriggerRelayInternalSpike::~TriggerRelayInternalSpike() {}

void TriggerRelayInternalSpike::ProcessEvent(Simulation* simulation) {
    this->ProcessEvent(simulation, ALL_EVENTS_ENABLED);
}

void TriggerRelayInternalSpike::ProcessEvent(Simulation* simulation, RealTimeRestrictionLevel level) {
    if (level >= SPIKES_DISABLED || simulation == nullptr || this->SourceNeuron == nullptr) {
        return;
    }

    simulation->WriteSpike(this);
    // Relay spikes use the same propagated-spike path as ordinary internal
    // spikes, so downstream synapses and learning hooks observe normal events.
    if (this->SourceNeuron->PropogationStructure != nullptr) {
        for (int queue = 0; queue < simulation->NumberOfQueue; ++queue) {
            if (this->SourceNeuron->Output_Synaps_Number[queue] == 0) {
                continue;
            }
            PropogatedSpike* prop_spike = new PropogatedSpike(
                this->getTime() + this->SourceNeuron->Output_Synaps[queue][0]->delay,
                queue,
                this->SourceNeuron,
                0,
                this->SourceNeuron->PropogationStructure->NDifferentdelays[queue]);
            if (queue == this->getIndex()) {
                simulation->EventHeap->Insert_a_Event(prop_spike, queue);
            } else {
                simulation->EventHeap->Insert_a_Event_to_Buffer(prop_spike, this->getIndex(), queue);
            }
        }
    }

    if (level >= LEARNING_RULES_DISABLED || simulation->network == nullptr) {
        return;
    }
    const int learning_rule_count = simulation->network->LearningRuleNum;
    for (int rule = 0; rule < learning_rule_count; ++rule) {
        // The relay is a real spike source, so post-side and trigger/post
        // learning semantics match other legacy internal spikes.
        if (this->SourceNeuron->PostSynapticLearning_Number[rule] > 0 &&
            this->SourceNeuron->PostSynapticLearning[rule][0]->LearningRule_withPost != nullptr) {
            this->SourceNeuron->PostSynapticLearning[rule][0]
                ->LearningRule_withPost
                ->ApplyPostSynaticSpike(this->SourceNeuron, this->getTime(), simulation);
        }
        if (this->SourceNeuron->TriggerAndPostSynapticLearning_Number[rule] > 0 &&
            this->SourceNeuron->TriggerAndPostSynapticLearning[rule][0]->LearningRule_withPostAndTrigger != nullptr) {
            this->SourceNeuron->TriggerAndPostSynapticLearning[rule][0]
                ->LearningRule_withPostAndTrigger
                ->ApplyPostSynaticSpike(this->SourceNeuron, this->getTime(), simulation);
        }
    }
}

enum EventPriority TriggerRelayInternalSpike::getPriority() {
    return INTERNALSPIKE;
}
