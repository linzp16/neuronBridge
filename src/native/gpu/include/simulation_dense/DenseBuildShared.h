#ifndef NPGR_DENSE_BUILD_SHARED_H
#define NPGR_DENSE_BUILD_SHARED_H

#include "dense_subnetwork/DenseNeuronModelSpec.h"
#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"
#include "gpu_runtime/GpuPropagationLayout.h"
#include "gpu_runtime/GpuPropagationRuntime.h"
#include "simulation_dense/SimulationConnectionWeightRef.h"
#include "source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"

#include <boost/any.hpp>

#include <cstdint>
#include <list>
#include <map>
#include <string>
#include <vector>

class Network;
class NeuronModel;
class Interconnections;

namespace npgr {
namespace streaming { class NbnetReader; }
namespace sim_support {

struct DenseSubnetworkBuildSpec {
    // Dense subnetwork name shared by all layers tagged with this block.
    std::string name;
    // GPU-ready internal synapse layout and propagation schedule.
    GpuPropagationLayout layout;
    // Dense neuron model records consumed directly by the unified GPU neuron runtime.
    std::vector<DenseNeuronModelSpec> neuron_models;
    // Dense runtime model ids retained for debug and compatibility views.
    std::vector<int> internal_model_ids;
    // Dense-local neuron id -> DenseNeuronModelSpec::model_id.
    std::vector<int> neuron_model_id_by_neuron;
    // Dense-local source and delay arrays parallel to layout.synapses, used only
    // to generate layout.pre_delay_slices during finalization.
    std::vector<int> synapse_source_local_ids;
    std::vector<int> synapse_delay_slots;
    // Parsed legacy learning rules used by the dense learning factory while
    // materializing model-owned flat field pools. learning_rule_table contains
    // only rule metadata; learning_rule_fields stores the concrete rule
    // parameters in rule-indexed arrays.
    std::vector<DenseLearningRuleInfo> learning_rule_table;
    DenseLearningRuleFieldTable learning_rule_fields;
    // Boundary input slot arrays. The slot index addresses all vectors below.
    std::vector<int> input_source_main_ids;
    std::vector<int> input_interface_main_ids;
    std::vector<int> input_target_local_ids;
    std::vector<unsigned char> input_uses_current;
    std::vector<std::uint8_t> input_pending_channels;
    std::vector<float> input_scales;
    std::vector<int> input_representative_source_main_ids;
    // Boundary input connection arrays. The connection index addresses all vectors below.
    std::vector<int> input_connection_source_main_ids;
    std::vector<int> input_connection_slot_indices;
    std::vector<int> input_connection_delays;
    std::vector<float> input_connection_weights;
    std::vector<float> input_connection_max_weights;
    std::vector<int> input_connection_types;
    std::vector<std::uint8_t> input_connection_pending_channels;
    std::vector<float> input_connection_effect_scales;
    // Boundary output arrays. The output index addresses all vectors below.
    std::vector<int> output_source_local_ids;
    std::vector<int> output_target_main_ids;
    std::vector<int> output_delays;
    std::vector<float> output_weights;
    std::vector<int> output_types;
    // Dense-to-dense boundary output arrays. These route source-subnetwork
    // firings into interface slots owned by another dense subnetwork.
    std::vector<int> output_dense_source_local_ids;
    std::vector<int> output_dense_target_spec_ids;
    std::vector<int> output_dense_target_slot_indices;
    std::vector<int> output_dense_delays;
    std::vector<float> output_dense_weights;
    std::vector<int> output_dense_types;
    // Fast output lookup mask indexed by dense-local neuron id.
    std::vector<unsigned char> output_neuron_mask;
    // Dense-local neuron id -> original user-visible neuron id for monitor output.
    std::vector<int> monitor_local_to_original_ids;
    // Dense-local monitor candidates collected from layer.isMonitored.
    std::vector<unsigned char> monitor_candidate_mask;
    // Runtime settings applied to this dense block only.
    RuntimeConfig config;
    int queue_index = 0;
    int update_timestep = 1;

    bool IsValid(std::string* reason = nullptr) const;
    bool IsOutputNeuron(int neuron_id) const;
};

struct DenseSubnetworkLayerConfig {
    // Dense subnetwork name assigned to the layer.
    std::string name;
    // True when runtime_config should be written into the layer parameters.
    bool has_runtime_config = false;
    // Optional dense runtime overrides serialized into NeuronParameter.
    RuntimeConfig runtime_config;
    // True when queue_index should be written into the layer parameters.
    bool has_queue_index = false;
    // Event-queue index requested for this dense subnetwork.
    int queue_index = 0;
};

struct PreparedSimulationBuild {
    // Layers and connections that remain in the legacy/event-driven main network.
    std::list<NeuronLayerDescription> main_layers;
    std::list<ConnectionDescription> main_connections;
    // Original neuron id -> compacted main-network neuron id; dense-only neurons map to -1.
    std::vector<int> original_to_main_neuron_id;
    // Original connection_list flattened index -> live runtime weight owner.
    std::vector<RuntimeConnectionWeightRef> original_connection_weight_refs;
    // One direct-build spec per named dense subnetwork.
    std::vector<DenseSubnetworkBuildSpec> dense_specs;
    // Non-empty when dense build rejected an unsupported boundary rule.
    std::string build_error;
};

// Reads the dense_subnetwork_name marker from a layer parameter map.
bool TryGetDenseSubnetworkName(const NeuronLayerDescription& layer, std::string* dense_name);
// Returns true when the layer's neuron model has a dense runtime implementation.
bool IsDenseManagedLayer(const NeuronLayerDescription& layer);
// Fills a RuntimeConfig with conservative defaults for a newly discovered dense block.
void InitializeDefaultDenseRuntimeConfig(RuntimeConfig* config, float basetimestep);
// Applies per-layer dense runtime overrides such as queue index and buffer sizes.
void ApplyDenseRuntimeOverridesFromLayer(RuntimeConfig* config,
                                         int* queue_index,
                                         const NeuronLayerDescription& layer);
// Writes a dense subnetwork tag and optional runtime overrides into a layer description.
void ApplyDenseSubnetworkLayerConfigToLayer(NeuronLayerDescription* layer,
                                            const DenseSubnetworkLayerConfig& config,
                                            float basetimestep);

// Splits the original build description into a compact legacy main network and
// one runtime-ready DenseSubnetworkBuildSpec per named dense block.
PreparedSimulationBuild PrepareBlackBoxDenseBuild(const std::list<NeuronLayerDescription>& neuron_layer_list,
                                                  const std::list<ConnectionDescription>& connection_list,
                                                  const std::list<LearningRuleDescription>& learning_rule_list,
                                                  float basetimestep);
PreparedSimulationBuild PrepareBlackBoxDenseBuildStreaming(
    const std::list<NeuronLayerDescription>& neuron_layer_list,
    const streaming::NbnetReader& reader,
    const std::list<LearningRuleDescription>& learning_rule_list,
    float basetimestep);
// Collects all legacy neuron-model pointers from an already built Network.
std::vector<const NeuronModel*> CollectAllNeuronModels(const Network* network);
// Finds the concrete main-network connection used as a dense interface binding.
Interconnections* FindMainConnection(Network* network, int source_id, int target_id);

}  // namespace sim_support
}  // namespace npgr

#endif
