/*
 * NetworkConstructStructure.h
 *
 * Lightweight description structs used by examples and loaders to build a
 * Network. The Network constructor consumes these values and expands them into
 * concrete neuron models, neuron instances, synapses, learning rules, and
 * optional outer-dynamics bindings.
 */
#ifndef NETWORKCONSTRUCTSTRUCTURE_H
#define NETWORKCONSTRUCTSTRUCTURE_H

#include <boost/any.hpp>
#include <map>
#include <string>
#include <vector>

/* Generic named model configuration. */
struct ModelDescription {
    // Name passed to the corresponding model factory.
    std::string ModelName;

    // Factory-specific parameter map. Values are stored as boost::any because
    // different neuron, synapse, and learning models expect different types.
    std::map<std::string, boost::any> ModelParameter;
};

/* Describes one contiguous layer of neurons in the original network ID space. */
struct NeuronLayerDescription {
    // Number of neurons in this layer.
    int numberofneuron;
    // Neuron model name used by NeuronModelFactory.
    std::string ModelName;
    // Update period, measured in multiples of the simulation base timestep.
    int update_timestep = 1;
    // Whether neurons in this layer should be written to monitor outputs.
    bool isMonitored = false;
    // Whether neurons in this layer are considered output neurons.
    bool isOutput = false;
    // Whether this layer receives spikes from a communication driver.
    bool isCommunicationInput = false;
    // Model-specific parameters passed into the neuron model factory.
    std::map<std::string, boost::any> NeuronParameter;
};

/*
 * Describes a batch of synaptic connections. All vectors are parallel arrays:
 * entry i in each vector describes one source-to-target connection.
 */
struct ConnectionDescription {
    // Original source neuron IDs.
    std::vector<int> SourceNeuron;
    // Original target neuron IDs.
    std::vector<int> TargetNeuron;
    // Synapse/input type: 0 excitatory, 1 inhibitory, 2 NMDA, 3 current input.
    std::vector<int> Type;
    // Initial synaptic weights.
    std::vector<float> Weight;
    // Per-connection maximum weights used by plasticity updates.
    std::vector<float> MaxWeight;
    // Transmission delay in simulation steps.
    std::vector<int> Delay;
    // Learning rule index for post-synaptic or normal plasticity; -1 disables it.
    std::vector<int> SynapseRule;
    // Learning rule index for trigger/teaching plasticity; -1 disables it.
    std::vector<int> TriggerRule;
};

/* Describes one learning-rule model to create through the learning factory. */
struct LearningRuleDescription {
    // Learning rule factory name.
    std::string RuleName;
    // Rule-specific parameter map.
    std::map<std::string, boost::any> RuleParameter;
};

/* Product-coded feedback target map used by OuterDynamic models.
 * Each joint owns a flat product table. The concrete model defines the
 * encoded variables and maps their uniform bin ids into one product index. */
struct OuterDynamicFeedbackProductEncoding {
    std::vector<std::vector<int> > neuron_indices_by_joint;
    std::vector<int> bins;
};

/* Single-variable feedback target map used by OuterDynamic models.
 * Shape is [joint][variable][bin]. This is intentionally variable-oriented
 * rather than arbitrary channel-oriented so model semantics stay explicit. */
struct OuterDynamicFeedbackSingleEncoding {
    std::vector<std::vector<std::vector<int> > > neuron_indices_by_joint_variable;
    std::vector<int> bins;
};

/* Describes an optional external plant or dynamics model coupled to the SNN. */
struct OuterDynamicDescription {
    // Optional user-visible name. When empty, Simulation assigns ModelName#index.
    std::string name;
    // Outer-dynamics model factory name.
    std::string ModelName;
    // Model-specific parameters.
    std::map<std::string, boost::any> ModelParameter;
    // Update period, measured in multiples of the simulation base timestep.
    int update_timestep = 1;
    // Suggested communication window used by outer models for decoding/smoothing.
    int communication_interval = 1;
    // OpenMP/event queue index assigned to this outer model.
    int queue_index = 0;
    // GC input neuron IDs per joint. Total-model reproductions drive these
    // layers directly with one-hot spikes.
    std::vector<std::vector<int>> gc_neuron_indices_by_joint;
    // MF input neuron IDs per joint, ordered as [q_des, q_act, qd_des, qd_act].
    std::vector<std::vector<int>> mf_neuron_indices_by_joint;
    // Positive-error climbing-fiber input neuron IDs per joint.
    std::vector<std::vector<int>> cf_positive_neuron_indices_by_joint;
    // Negative-error climbing-fiber input neuron IDs per joint.
    std::vector<std::vector<int>> cf_negative_neuron_indices_by_joint;
    // Positive-torque DCN output neuron IDs per joint.
    std::vector<std::vector<int>> dcn_positive_neuron_indices_by_joint;
    // Negative-torque DCN output neuron IDs per joint.
    std::vector<std::vector<int>> dcn_negative_neuron_indices_by_joint;
    // Optional product-coded state feedback targets, typically GC-like inputs.
    OuterDynamicFeedbackProductEncoding state_feedback_product;
    // Optional single-variable state feedback targets.
    OuterDynamicFeedbackSingleEncoding state_feedback_single;
    // Optional product-coded error feedback targets.
    OuterDynamicFeedbackProductEncoding error_feedback_product;
    // Optional single-variable error feedback targets, typically CF-like inputs.
    OuterDynamicFeedbackSingleEncoding error_feedback_single;
};

/* Describes network-to-OuterDynamic torque input connections.
 * These are translated during Simulation construction into ordinary
 * connections targeting hidden OuterDynamicInterfaceNeuronModel endpoints. */
struct OuterDynamicConnectionDescription {
    std::vector<int> SourceNeuron;
    std::vector<int> TargetOuterDynamic;
    std::vector<int> TargetJoint;
    std::vector<int> Type;
    std::vector<float> Weight;
    std::vector<int> Delay;
};

#endif
