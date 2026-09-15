/*
 * Interconnections.h
 *
 * Defines one concrete synaptic connection in the compiled legacy network.
 * ConnectionDescription is the input format; Interconnections is the runtime
 * object used by neurons, propagation structures, current delivery, and
 * learning rules.
 */
#ifndef INTERCONNECTIONS_H
#define INTERCONNECTIONS_H

#include "../source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuronModel.h"
#include "../source_file_realtime_v1_async/Neuron/inc/Neuron.h"

class LearningRule;

class Interconnections {
public:
    // Source neuron that emits spikes or current through this connection.
    Neuron* SourceNeuron;

    // Target neuron that receives this connection.
    Neuron* TargetNeuron;

    // Target neuron model that owns the target neuron's state vector.
    NeuronModel* TargetNeuronModel;

    // Transmission delay in simulation steps.
    int delay;

    // Current synaptic weight.
    float weight;

    // Upper bound used when plasticity increases the weight.
    float maximum_weight;

    // Stable connection index in Network::inters.
    int Index;

    // Target neuron's local index inside TargetNeuronModel.
    int TargetNeuronModelIndex;

    // Connection type: 0 excitatory, 1 inhibitory, 2 NMDA, 3 external current.
    int type;

    // Local current-input slot for current-delivery synapses.
    int subindex_type;

    // Index into post-synaptic learning-rule state, or -1 when unused.
    int LearningRuleIndex_withPost;

    // Index into trigger/pre-synaptic learning-rule state, or -1 when unused.
    int LearningRuleIndex_withTrigger;

    // Index for learning rules that need both pre and post activity.
    int LearningRuleIndex_withPostAndTrigger;

    // Position of this connection in the post-synaptic learning-rule target list.
    int LearningRuleIndex_withPost_Target;

    // Position of this connection in the trigger learning-rule target list.
    int LearningRuleIndex_withTrigger_Target;

    // Position of this connection in the combined pre/post learning target list.
    int LearningRuleIndex_withPostAndTrigger_Target;

    // Learning rule applied from post-synaptic activity, if any.
    LearningRule* LearningRule_withPost;

    // Learning rule applied from trigger/pre-synaptic activity, if any.
    LearningRule* LearningRule_withTrigger;

    // Learning rule applied when both pre and post activity are required.
    LearningRule* LearningRule_withPostAndTrigger;

    // True when this connection participates in trigger/teaching learning.
    bool TriggerLearning = false;

    // Creates an empty connection; fields are initialized in the implementation.
    Interconnections();

    // Creates a fully specified connection between source and target neurons.
    Interconnections(Neuron* source,
                     Neuron* target,
                     NeuronModel* targetModel,
                     int delay,
                     float weight,
                     int type,
                     int index);

    // Releases resources owned by this connection.
    ~Interconnections();

    // Sets the stable index in Network::inters.
    void SetIndex(int index);

    // Sets the source neuron pointer.
    void SetSourceNeuron(Neuron* source);

    // Sets the target neuron pointer.
    void SetTargetNeuron(Neuron* target);

    // Sets the target neuron model pointer.
    void SetTargetNeuronModel(NeuronModel* targetModel);

    // Sets the target neuron's local index inside its model.
    void SetTargetNeuronModelIndex(int index);

    // Sets the current synaptic weight.
    void SetWeight(float weight);

    // Sets the transmission delay in simulation steps.
    void SetDelay(int delay);

    // Sets the connection type used by neuron and synapse models.
    void SetType(int type);

    // Applies a plasticity increment and clamps the weight to [0, maximum_weight].
    inline void UpdateWeight(float Increament) {
        this->weight += Increament;
        if (this->weight > this->maximum_weight) {
            this->weight = this->maximum_weight;
        }
        else if (this->weight < 0.0) {
            this->weight = 0.0;
        }
    }
};

#endif
