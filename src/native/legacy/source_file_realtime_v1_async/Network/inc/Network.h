/*
 * Network.h
 *
 * Runtime representation of the compiled legacy spiking neural network.
 * Network owns the concrete neurons, neuron models, interconnections, learning
 * rules, and propagation lookup structures derived from the description structs.
 */
#ifndef NETWORK_H
#define NETWORK_H

class Interconnections;
class Simulation;
namespace npgr { namespace streaming { class ConnectionRecordSource; } }

#include "../source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "../source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "../source_file_realtime_v1_async/Neuron/inc/Neuron.h"
#include <list>
#include <vector>

class Network {
public:
    // Flat array of all compiled synaptic connections.
    Interconnections* inters;

    // Number of entries in inters.
    int intersNum;

    // Network timestep multiplier relative to the simulation base timestep.
    int timesteps;

    // Neuron model instances grouped by model type and queue index.
    NeuronModel*** neurontypes;

    // Number of distinct neuron model types.
    int neurontypesNum;

    // Flat array of all compiled neurons.
    Neuron* neurons;

    // Number of entries in neurons.
    int neuronsNum;

    // CPU time-driven neurons grouped as [model type][queue][local neuron].
    Neuron**** TimeDrivenNeurons;

    // Counts for TimeDrivenNeurons grouped as [model type][queue].
    int** TimeDrivenNeuronsNum;

    // GPU time-driven neurons grouped as [model type][queue][local neuron].
    Neuron**** TimeDrivenNeuronsGPU;

    // Counts for TimeDrivenNeuronsGPU grouped as [model type][queue].
    int** TimeDrivenNeuronsNumGPU;

    // Number of OpenMP/event queues used by the simulation.
    int NumberOfQueue;

    // Mapping from sorted weight order back to the original connection objects.
    Interconnections** wordination;

    // Number of learning-rule model instances.
    int LearningRuleNum;

    // Learning-rule model instances.
    LearningRule** LearningRules;

    // True when at least one neuron is configured for monitor output.
    bool isMonitor;

    // Base simulation timestep size in milliseconds.
    float basetemestepsize;

    // Initial weight snapshot retained for explicit restoration support.
    std::vector<float> initial_weights;

    // Creates an empty network shell.
    Network();

    // Compiles a network from layer, connection, and learning-rule descriptions.
    Network(const std::list<NeuronLayerDescription>& neuron_layer_list,
            const std::list<ConnectionDescription>& connection_list,
            const std::list<LearningRuleDescription>& learning_rule_list,
            int NumberOfQueue,
            float basetimestepsize,
            Simulation* simulation);

    // Compiles a main-only network directly from an nbnet connection source.
    // The reader is scanned once for plasticity counts and once to populate the
    // final Interconnections array, without materializing ConnectionDescription.
    Network(const std::list<NeuronLayerDescription>& neuron_layer_list,
            const npgr::streaming::ConnectionRecordSource& source,
            const std::list<LearningRuleDescription>& learning_rule_list,
            int NumberOfQueue,
            float basetimestepsize,
            Simulation* simulation);

    // Releases all compiled network state owned by this object.
    ~Network();

    // Sorts outgoing connections and builds per-neuron propagation structures.
    void CaculateOutputConnection();

    // Builds incoming connection lists used by neurons and learning rules.
    void CaculateInputConnection();

    // Runs the full compilation pipeline from descriptions to runtime objects.
    void CompileNetwork(const std::list<NeuronLayerDescription>& neuron_layer_list,
                        const std::list<ConnectionDescription>& connection_list,
                        const std::list<LearningRuleDescription>& learning_rule_list,
                        Simulation* simulation);

    void CompileNetworkStreaming(const std::list<NeuronLayerDescription>& neuron_layer_list,
                                 const npgr::streaming::ConnectionRecordSource& source,
                                 const std::list<LearningRuleDescription>& learning_rule_list,
                                 Simulation* simulation);

    // Creates neuron model instances and assigns neurons to those models.
    void CreateNeuronModel(const std::list<NeuronLayerDescription>& neuron_layer_list,
                           int timesteps,
                           float basetimestepsize,
                           Simulation* simulation);

    // Initializes neuron state vectors after model allocation.
    void InitializeStates(int** Neurons);

    // Creates concrete Interconnections from the description arrays.
    void CreateConnections(const std::list<ConnectionDescription>& connection_list,
                           std::vector<int>& N_connectionsPerRule);

    void CreateConnectionsStreaming(const npgr::streaming::ConnectionRecordSource& source,
                                    std::vector<int>& N_connectionsPerRule);

    // Builds the mapping between sorted connection order and original objects.
    void setWeightOrdination();

    // Allocates and initializes plasticity state for learning-rule connections.
    void InitializeSynapticPlasticityState(const std::vector<int>& N_connectionsPerRule);

    // Creates learning-rule models and returns per-rule connection counts.
    std::vector<int> CreateWeightChange(const std::list<LearningRuleDescription>& learning_rule_list);

    // Returns the minimum delay for cross-queue propagation synchronization.
    int GetMinInterpropagationTime();

    // Writes current connection weights to a text file.
    void SaveWeightsToFile(const char* filename);

    // Loads connection weights from a text file.
    void LoadWeightsFromFile(const char* filename);

    // Resets dynamic neuron/synapse state without changing current weights.
    void ResetDynamicState();

    // Stores the current weights as the restore baseline.
    void SnapshotInitialWeights();

    // Restores weights from the stored baseline snapshot.
    void RestoreInitialWeights();
};

#endif
