#include "simulation_dense/DenseBuildShared.h"
#include "streaming_build/NbnetReader.h"

#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "dense_subnetwork/model/IDenseNeuronModel.h"
#include "neuron_model/NeuronModelCatalog.h"
#include "source_file_realtime_v1_async/Network/inc/Network.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>

namespace npgr {
namespace sim_support {
namespace {

bool TryGetStringParameter(const std::map<std::string, boost::any>& parameters,
                           const std::string& key,
                           std::string* value) {
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end() || value == nullptr) {
        return false;
    }
    try {
        *value = boost::any_cast<std::string>(found->second);
        return !value->empty();
    } catch (const boost::bad_any_cast&) {
        return false;
    }
}

bool TryGetFloatParameter(const std::map<std::string, boost::any>& parameters,
                          const std::string& key,
                          float* value) {
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end() || value == nullptr) {
        return false;
    }
    try {
        *value = boost::any_cast<float>(found->second);
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<float>(boost::any_cast<double>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<float>(boost::any_cast<int>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    return false;
}

bool TryGetFloatParameterAny(const std::map<std::string, boost::any>& parameters,
                             const char* primary_key,
                             const char* alias_key,
                             float* value) {
    if (TryGetFloatParameter(parameters, primary_key, value)) {
        return true;
    }
    if (alias_key != nullptr && TryGetFloatParameter(parameters, alias_key, value)) {
        return true;
    }
    return false;
}

bool TryGetFloatParameterAny3(const std::map<std::string, boost::any>& parameters,
                              const char* primary_key,
                              const char* alias_key1,
                              const char* alias_key2,
                              float* value) {
    if (TryGetFloatParameter(parameters, primary_key, value)) {
        return true;
    }
    if (alias_key1 != nullptr && TryGetFloatParameter(parameters, alias_key1, value)) {
        return true;
    }
    if (alias_key2 != nullptr && TryGetFloatParameter(parameters, alias_key2, value)) {
        return true;
    }
    return false;
}

bool TryGetIntParameter(const std::map<std::string, boost::any>& parameters,
                        const std::string& key,
                        int* value) {
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end() || value == nullptr) {
        return false;
    }
    try {
        *value = boost::any_cast<int>(found->second);
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<int>(boost::any_cast<float>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    try {
        *value = static_cast<int>(boost::any_cast<double>(found->second));
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    return false;
}

bool TryGetBoolParameter(const std::map<std::string, boost::any>& parameters,
                         const std::string& key,
                         bool* value) {
    std::map<std::string, boost::any>::const_iterator found = parameters.find(key);
    if (found == parameters.end() || value == nullptr) {
        return false;
    }
    try {
        *value = boost::any_cast<bool>(found->second);
        return true;
    } catch (const boost::bad_any_cast&) {
    }
    int int_value = 0;
    if (TryGetIntParameter(parameters, key, &int_value)) {
        *value = (int_value != 0);
        return true;
    }
    return false;
}

bool Fail(std::string* reason, const std::string& message) {
    if (reason != nullptr) {
        *reason = message;
    }
    return false;
}

bool CheckSize(std::string* reason,
               std::size_t actual,
               std::size_t expected,
               const char* message) {
    if (actual == expected) {
        return true;
    }
    return Fail(reason, message);
}

void FillLayerParams(DenseNeuronModelSpec* spec,
                     const NeuronLayerDescription& layer,
                     const DenseNeuronRange& range,
                     int neuron_count) {
    if (spec == nullptr || neuron_count <= 0 || layer.numberofneuron <= 0) {
        return;
    }
    const IDenseNeuronModel* model =
        DenseNeuronModelFactory::Instance().FindByLegacyName(layer.ModelName);
    if (model != nullptr) {
        std::string ignored_reason;
        model->FillLayerParams(spec, layer, range, neuron_count, &ignored_reason);
    }
}

bool ValidateLayerModelOwnership(const NeuronLayerDescription& layer,
                                 const std::string& dense_name,
                                 std::string* reason) {
    const NeuronModelCatalog& catalog = NeuronModelCatalog::Instance();
    if (catalog.Resolve(layer.ModelName) == nullptr) {
        return Fail(reason, "Unknown neuron model '" + layer.ModelName + "'.");
    }
    const NeuronModelRole role = catalog.RoleOf(layer.ModelName);
    const bool is_dense_layer = !dense_name.empty();
    if (is_dense_layer) {
        if (role == NeuronModelRole::InputSpike ||
            role == NeuronModelRole::InputCurrent) {
            return Fail(reason,
                        "Input neuron model '" + layer.ModelName +
                            "' cannot be placed inside dense subnetwork '" + dense_name +
                            "'; declare it in the main network and connect it to dense target neurons instead.");
        }
        if (role == NeuronModelRole::Special) {
            return Fail(reason,
                        "Special neuron model '" + layer.ModelName +
                            "' is public but cannot be placed inside dense subnetwork '" + dense_name + "'.");
        }
        if (!catalog.IsSupported(layer.ModelName, NeuronBackend::DenseGpu)) {
            return Fail(reason,
                        "Neuron model '" + layer.ModelName +
                            "' is not supported by dense GPU subnetwork '" + dense_name + "'.");
        }
        return true;
    }
    if (role == NeuronModelRole::DenseOnlyHelper) {
        return Fail(reason,
                    "Dense-only helper model '" + layer.ModelName +
                        "' must be assigned to a dense subnetwork; set dense_subnetwork_name on its layer.");
    }
    return true;
}
}  // namespace
bool DenseSubnetworkBuildSpec::IsValid(std::string* reason) const {
    if (name.empty()) {
        return Fail(reason, "dense subnetwork name must not be empty");
    }
    if (!layout.IsValid(reason)) {
        return false;
    }
    if (!output_neuron_mask.empty() &&
        output_neuron_mask.size() != static_cast<std::size_t>(layout.stats.neuron_count)) {
        return Fail(reason, "output_neuron_mask size must match layout neuron_count");
    }
    if (!neuron_model_id_by_neuron.empty() &&
        neuron_model_id_by_neuron.size() != static_cast<std::size_t>(layout.stats.neuron_count)) {
        return Fail(reason, "neuron_model_id_by_neuron size must match layout neuron_count");
    }
    for (std::size_t neuron_index = 0; neuron_index < neuron_model_id_by_neuron.size(); ++neuron_index) {
        const int model_id = neuron_model_id_by_neuron[neuron_index];
        if (model_id < -1 || model_id >= static_cast<int>(neuron_models.size())) {
            return Fail(reason, "neuron_model_id_by_neuron contains an out-of-range model id");
        }
    }
    for (std::size_t model_index = 0; model_index < neuron_models.size(); ++model_index) {
        const DenseNeuronModelSpec& model = neuron_models[model_index];
        if (model.model_id < 0) {
            return Fail(reason, "dense neuron model spec contains an invalid model_id");
        }
        if (model.range.begin < 0 || model.range.count < 0 ||
            model.range.begin + model.range.count > layout.stats.neuron_count) {
            return Fail(reason, "dense neuron model spec contains an invalid neuron range");
        }
    }
    const std::size_t input_count = input_target_local_ids.size();
    if (!CheckSize(reason, input_source_main_ids.size(), input_count, "dense input slot arrays have mismatched sizes") ||
        !CheckSize(reason, input_interface_main_ids.size(), input_count, "dense input slot arrays have mismatched sizes") ||
        !CheckSize(reason, input_uses_current.size(), input_count, "dense input slot arrays have mismatched sizes") ||
        !CheckSize(reason, input_pending_channels.size(), input_count, "dense input slot arrays have mismatched sizes") ||
        !CheckSize(reason, input_scales.size(), input_count, "dense input slot arrays have mismatched sizes") ||
        !CheckSize(reason, input_representative_source_main_ids.size(), input_count, "dense input slot arrays have mismatched sizes")) {
        return false;
    }
    for (std::size_t index = 0; index < input_count; ++index) {
        if (input_target_local_ids[index] < 0 ||
            input_target_local_ids[index] >= layout.stats.neuron_count) {
            return Fail(reason, "dense interface slot contains invalid target neuron id");
        }
    }
    const std::size_t input_connection_count = input_connection_source_main_ids.size();
    if (!CheckSize(reason, input_connection_slot_indices.size(), input_connection_count, "dense input connection arrays have mismatched sizes") ||
        !CheckSize(reason, input_connection_delays.size(), input_connection_count, "dense input connection arrays have mismatched sizes") ||
        !CheckSize(reason, input_connection_weights.size(), input_connection_count, "dense input connection arrays have mismatched sizes") ||
        !CheckSize(reason, input_connection_max_weights.size(), input_connection_count, "dense input connection arrays have mismatched sizes") ||
        !CheckSize(reason, input_connection_types.size(), input_connection_count, "dense input connection arrays have mismatched sizes") ||
        !CheckSize(reason, input_connection_pending_channels.size(), input_connection_count, "dense input connection arrays have mismatched sizes") ||
        !CheckSize(reason, input_connection_effect_scales.size(), input_connection_count, "dense input connection arrays have mismatched sizes")) {
        return false;
    }
    for (std::size_t index = 0; index < input_connection_count; ++index) {
        if (input_connection_slot_indices[index] < 0 ||
            input_connection_slot_indices[index] >= static_cast<int>(input_count) ||
            input_connection_delays[index] < 0) {
            return Fail(reason, "dense input connection contains invalid ids");
        }
    }
    const std::size_t output_count = output_source_local_ids.size();
    if (!CheckSize(reason, output_target_main_ids.size(), output_count, "dense output arrays have mismatched sizes") ||
        !CheckSize(reason, output_delays.size(), output_count, "dense output arrays have mismatched sizes") ||
        !CheckSize(reason, output_weights.size(), output_count, "dense output arrays have mismatched sizes") ||
        !CheckSize(reason, output_types.size(), output_count, "dense output arrays have mismatched sizes")) {
        return false;
    }
    for (std::size_t index = 0; index < output_count; ++index) {
        if (output_source_local_ids[index] < 0 ||
            output_source_local_ids[index] >= layout.stats.neuron_count ||
            output_target_main_ids[index] < 0 ||
            output_delays[index] < 0) {
            return Fail(reason, "dense external output contains invalid ids");
        }
    }
    const std::size_t dense_output_count = output_dense_source_local_ids.size();
    if (!CheckSize(reason, output_dense_target_spec_ids.size(), dense_output_count, "dense-to-dense output arrays have mismatched sizes") ||
        !CheckSize(reason, output_dense_target_slot_indices.size(), dense_output_count, "dense-to-dense output arrays have mismatched sizes") ||
        !CheckSize(reason, output_dense_delays.size(), dense_output_count, "dense-to-dense output arrays have mismatched sizes") ||
        !CheckSize(reason, output_dense_weights.size(), dense_output_count, "dense-to-dense output arrays have mismatched sizes") ||
        !CheckSize(reason, output_dense_types.size(), dense_output_count, "dense-to-dense output arrays have mismatched sizes")) {
        return false;
    }
    for (std::size_t index = 0; index < dense_output_count; ++index) {
        if (output_dense_source_local_ids[index] < 0 ||
            output_dense_source_local_ids[index] >= layout.stats.neuron_count ||
            output_dense_target_spec_ids[index] < 0 ||
            output_dense_target_slot_indices[index] < 0 ||
            output_dense_delays[index] < 0) {
            return Fail(reason, "dense-to-dense output contains invalid ids");
        }
    }
    return true;
}

bool DenseSubnetworkBuildSpec::IsOutputNeuron(int neuron_id) const {
    if (neuron_id < 0 || neuron_id >= layout.stats.neuron_count) {
        return false;
    }
    if (!output_neuron_mask.empty()) {
        return output_neuron_mask[static_cast<std::size_t>(neuron_id)] != 0;
    }
    for (std::size_t index = 0; index < output_source_local_ids.size(); ++index) {
        if (output_source_local_ids[index] == neuron_id) {
            return true;
        }
    }
    for (std::size_t index = 0; index < output_dense_source_local_ids.size(); ++index) {
        if (output_dense_source_local_ids[index] == neuron_id) {
            return true;
        }
    }
    return false;
}
bool TryGetDenseSubnetworkName(const NeuronLayerDescription& layer, std::string* dense_name) {
    return TryGetStringParameter(layer.NeuronParameter, "dense_subnetwork_name", dense_name);
}

bool IsDenseManagedLayer(const NeuronLayerDescription& layer) {
    return DenseNeuronModelFactory::Instance().IsDenseManagedLayer(layer);
}

// generate the default subnetwok config
void InitializeDefaultDenseRuntimeConfig(RuntimeConfig* config, float basetimestep) {
    if (config == nullptr) {
        return;
    }
    config->steps_to_keep = 128;
    config->dt_ms = basetimestep;
}

void ApplyDenseRuntimeOverridesFromLayer(RuntimeConfig* config,
                                         int* queue_index,
                                         const NeuronLayerDescription& layer) {
    if (config == nullptr || queue_index == nullptr) {
        return;
    }

    int int_value = 0;
    float float_value = 0.0f;
    if (TryGetIntParameter(layer.NeuronParameter, "dense_steps_to_keep", &int_value)) {
        config->steps_to_keep = std::max(config->steps_to_keep, int_value);
    }
    if (TryGetIntParameter(layer.NeuronParameter, "dense_queue_index", &int_value)) {
        *queue_index = std::max(0, int_value);
    }
}

void ApplyDenseSubnetworkLayerConfigToLayer(NeuronLayerDescription* layer,
                                            const DenseSubnetworkLayerConfig& config,
                                            float basetimestep) {
    if (layer == nullptr || config.name.empty()) {
        return;
    }

    layer->NeuronParameter["dense_subnetwork_name"] = config.name;
    if (config.has_queue_index) {
        layer->NeuronParameter["dense_queue_index"] = std::max(0, config.queue_index);
    }

    if (!config.has_runtime_config) {
        return;
    }

    RuntimeConfig default_config;
    InitializeDefaultDenseRuntimeConfig(&default_config, basetimestep);
    const RuntimeConfig& runtime = config.runtime_config;
    if (runtime.steps_to_keep != default_config.steps_to_keep) {
        layer->NeuronParameter["dense_steps_to_keep"] = runtime.steps_to_keep;
    }
    if (runtime.dt_ms != default_config.dt_ms) {
        layer->NeuronParameter["dense_dt_ms"] = runtime.dt_ms;
    }
}

ConnectionDescription& EnsureConnectionBucket(std::list<ConnectionDescription>* connections) {
    if (connections->empty()) {
        connections->push_back(ConnectionDescription());
    }
    return connections->back();
}

void AppendConnection(ConnectionDescription* connection,
                      int source,
                      int target,
                      int delay,
                      float weight,
                      float max_weight,
                      int type,
                      int synapse_rule,
                      int trigger_rule) {
    connection->SourceNeuron.push_back(source);
    connection->TargetNeuron.push_back(target);
    connection->Type.push_back(type);
    connection->Weight.push_back(weight);
    connection->MaxWeight.push_back(max_weight);
    connection->Delay.push_back(delay);
    connection->SynapseRule.push_back(synapse_rule);
    connection->TriggerRule.push_back(trigger_rule);
}

int AppendDenseInputSlot(DenseSubnetworkBuildSpec* target_spec,
                         int target_local_id,
                         int type,
                         bool uses_current,
                         int representative_source_main_id) {
    if (target_spec == nullptr ||
        target_local_id < 0 ||
        target_local_id >= target_spec->layout.stats.neuron_count) {
        return -1;
    }
    // get the target neuron model id
    const int target_model_id =
        target_spec->neuron_model_id_by_neuron[static_cast<std::size_t>(target_local_id)];
    // define the boundary channel by whether it use the current channel
    PendingChannel boundary_channel =
        uses_current ? PendingChannel::Current : PendingChannel::ExcitatoryConductance;
    float boundary_scale = 1.0f;
    if (target_model_id >= 0 &&
        target_model_id < static_cast<int>(target_spec->neuron_models.size())) {
        // get the channel according to the type
        DenseNeuronModelFactory::Instance().ResolveSpikeEffect(
            target_spec->neuron_models[static_cast<std::size_t>(target_model_id)].factory_model_id,
            type,
            &boundary_channel,
            &boundary_scale);
    }
    // append the input slot description
    const int slot_index = static_cast<int>(target_spec->input_target_local_ids.size());
    target_spec->input_source_main_ids.push_back(representative_source_main_id);
    target_spec->input_interface_main_ids.push_back(-1);
    target_spec->input_target_local_ids.push_back(target_local_id);
    target_spec->input_uses_current.push_back(uses_current ? 1 : 0);
    target_spec->input_pending_channels.push_back(static_cast<std::uint8_t>(boundary_channel));
    target_spec->input_scales.push_back(boundary_scale);
    target_spec->input_representative_source_main_ids.push_back(representative_source_main_id);
    return slot_index;
}

void AppendMainToDenseInputConnection(DenseSubnetworkBuildSpec* target_spec,
                                      int source_main_id,
                                      int slot_index,
                                      int delay,
                                      float weight,
                                      float max_weight,
                                      int type) {
    if (target_spec == nullptr || slot_index < 0 ||
        slot_index >= static_cast<int>(target_spec->input_target_local_ids.size())) {
        return;
    }
    target_spec->input_connection_source_main_ids.push_back(source_main_id);
    target_spec->input_connection_slot_indices.push_back(slot_index);
    target_spec->input_connection_delays.push_back(delay);
    target_spec->input_connection_weights.push_back(weight);
    target_spec->input_connection_max_weights.push_back(max_weight);
    target_spec->input_connection_types.push_back(type);
    target_spec->input_connection_pending_channels.push_back(
        target_spec->input_pending_channels[static_cast<std::size_t>(slot_index)]);
    target_spec->input_connection_effect_scales.push_back(
        target_spec->input_scales[static_cast<std::size_t>(slot_index)]);
}

std::vector<const NeuronModel*> CollectAllNeuronModels(const Network* network) {
    std::vector<const NeuronModel*> models;
    if (network == nullptr || network->neurontypes == nullptr) {
        return models;
    }
    for (int model_index = 0; model_index < network->neurontypesNum; ++model_index) {
        if (network->neurontypes[model_index] == nullptr || network->neurontypes[model_index][0] == nullptr) {
            continue;
        }
        models.push_back(network->neurontypes[model_index][0]);
    }
    return models;
}

Interconnections* FindMainConnection(Network* network, int source_id, int target_id) {
    if (network == nullptr) {
        return nullptr;
    }
    for (int inter_index = 0; inter_index < network->intersNum; ++inter_index) {
        Interconnections* inter = network->inters + inter_index;
        if (inter->SourceNeuron != nullptr &&
            inter->TargetNeuron != nullptr &&
            inter->SourceNeuron->Neuron_index == source_id &&
            inter->TargetNeuron->Neuron_index == target_id) {
            return inter;
        }
    }
    return nullptr;
}

namespace {

class NbnetConnectionBlockRange {
public:
    explicit NbnetConnectionBlockRange(const streaming::NbnetReader& reader) : reader_(reader) {}

    class Iterator {
    public:
        Iterator(const streaming::NbnetReader* reader, std::uint64_t offset)
            : reader_(reader), offset_(offset) {
            Load();
        }
        const ConnectionDescription& operator*() const { return block_; }
        const ConnectionDescription* operator->() const { return &block_; }
        Iterator& operator++() {
            offset_ += block_.SourceNeuron.size();
            Load();
            return *this;
        }
        bool operator!=(const Iterator& other) const { return offset_ != other.offset_; }

    private:
        void Load() {
            block_ = ConnectionDescription();
            if (reader_ == nullptr || offset_ >= reader_->connection_count()) return;
            const std::vector<streaming::ConnectionRecordV1> records =
                reader_->ReadConnectionBatch(offset_, reader_->batch_records());
            block_.SourceNeuron.reserve(records.size());
            block_.TargetNeuron.reserve(records.size());
            block_.Type.reserve(records.size());
            block_.Weight.reserve(records.size());
            block_.MaxWeight.reserve(records.size());
            block_.Delay.reserve(records.size());
            block_.SynapseRule.reserve(records.size());
            block_.TriggerRule.reserve(records.size());
            for (const auto& record : records) {
                block_.SourceNeuron.push_back(static_cast<int>(record.source));
                block_.TargetNeuron.push_back(static_cast<int>(record.target));
                block_.Type.push_back(record.synapse_type);
                block_.Weight.push_back(record.weight);
                block_.MaxWeight.push_back(record.max_weight);
                block_.Delay.push_back(static_cast<int>(record.delay));
                block_.SynapseRule.push_back(record.synapse_rule);
                block_.TriggerRule.push_back(record.trigger_rule);
            }
        }
        const streaming::NbnetReader* reader_;
        std::uint64_t offset_;
        ConnectionDescription block_;
    };

    Iterator begin() const { return Iterator(&reader_, 0); }
    Iterator end() const { return Iterator(&reader_, reader_.connection_count()); }

private:
    const streaming::NbnetReader& reader_;
};

template <typename ConnectionRange>
PreparedSimulationBuild PrepareBlackBoxDenseBuildImpl(
    const std::list<NeuronLayerDescription>& neuron_layer_list,
    const ConnectionRange& connection_list,
    const std::list<LearningRuleDescription>& learning_rule_list,
    float basetimestep,
    bool retain_main_connections) {
    PreparedSimulationBuild prepared;
    auto reject_build = [&](const std::string& message) {
        prepared.build_error = message;
        return prepared;
    };
    // First pass records the original contiguous neuron-id span for every layer.
    // Dense tagging happens at the layer level through "dense_subnetwork_name".
    struct LayerRecord {
        // a struct which records the layer information
        NeuronLayerDescription layer;
        std::string dense_name;
        int original_start = 0;
        int original_end = 0;
    };
    std::vector<LayerRecord> layers;
    int total_original_neurons = 0;
    // Scan all the neuron layers
    for (std::list<NeuronLayerDescription>::const_iterator it = neuron_layer_list.begin(); it != neuron_layer_list.end(); ++it) {
        LayerRecord record;
        record.layer = *it;
        // get the start index of the neuron
        record.original_start = total_original_neurons;
        // accumulate the layer neuron number
        total_original_neurons += it->numberofneuron;
        // get the end of the layer
        record.original_end = total_original_neurons;
        // define whether it is a subnetwork layer by getting its name
        TryGetDenseSubnetworkName(*it, &record.dense_name);
        // Validate public model placement before splitting the user graph.
        // TriggerRelayNeuronModel is shared by main and dense networks, while
        // input devices remain main-network owned.
        if (!ValidateLayerModelOwnership(record.layer, record.dense_name, &prepared.build_error)) {
            return prepared;
        }
        layers.push_back(record);
    }
    // build the map between the original neuron to the main network neuron
    prepared.original_to_main_neuron_id.assign(static_cast<std::size_t>(total_original_neurons), -1);
    // original_to_dense_spec identifies which dense block owns each original neuron.
    // original_to_dense_local is the compact dense-local neuron id inside that block.
    std::vector<int> original_to_dense_spec(static_cast<std::size_t>(total_original_neurons), -1);
    std::vector<int> original_to_dense_local(static_cast<std::size_t>(total_original_neurons), -1);
    std::map<std::string, int> dense_name_to_spec;
    std::vector<DenseLearningRuleInfo> dense_learning_rules;
    DenseLearningRuleFieldTable dense_learning_rule_fields;
    // An empty learning_rule_list is the explicit switch for "no dense
    // plasticity system". In that case dense synapses must not reference
    // learning rule ids, and the finalizer will skip learning field materialization.
    if (!learning_rule_list.empty()) {
        if (!DenseLearningRuleFactory::Instance().BuildLegacyRuleTable(
                learning_rule_list,
                &dense_learning_rules,
                &dense_learning_rule_fields,
                &prepared.build_error)) {
            return prepared;
        }
    }

    // define the lambda function to find which layer the neuron belong to
    const auto find_layer_record_by_neuron =
        [&layers](int neuron_id) -> const LayerRecord* {
            for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
                const LayerRecord& record = layers[layer_index];
                if (neuron_id >= record.original_start && neuron_id < record.original_end) {
                    return &record;
                }
            }
            return nullptr;
        };

    int next_main_neuron_id = 0;
    // scan all the layers to find which network it belong to
    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const LayerRecord& record = layers[layer_index];
        if (record.dense_name.empty()) {
            // Untagged layers stay in the main legacy Network and get compacted
            // main-network ids because dense-only neurons are removed from it.
            prepared.main_layers.push_back(record.layer);
            for (int neuron_id = record.original_start; neuron_id < record.original_end; ++neuron_id) {
                prepared.original_to_main_neuron_id[static_cast<std::size_t>(neuron_id)] = next_main_neuron_id++;
            }
            continue;
        }
        if (record.layer.isOutput) {
            prepared.build_error =
                "dense subnetwork layer cannot be marked isOutput=true; "
                "declare dense outputs by adding dense->main or dense->dense connections";
            return prepared;
        }

        int spec_index = -1;
        std::map<std::string, int>::iterator found = dense_name_to_spec.find(record.dense_name);
        // check if it is a new subnetwork
        if (found == dense_name_to_spec.end()) {
            // create a new spec if it is a new subnet
            spec_index = static_cast<int>(prepared.dense_specs.size());
            dense_name_to_spec[record.dense_name] = spec_index;
            prepared.dense_specs.push_back(DenseSubnetworkBuildSpec());
            prepared.dense_specs.back().name = record.dense_name;
            prepared.dense_specs.back().learning_rule_table = dense_learning_rules;
            prepared.dense_specs.back().learning_rule_fields = dense_learning_rule_fields;
            // set the defaut runtime config.
            InitializeDefaultDenseRuntimeConfig(&prepared.dense_specs.back().config, basetimestep);
        } else {
            // get the index of the subnetwork.
            spec_index = found->second;
        }
        // get the particuler subnetwork
        DenseSubnetworkBuildSpec& spec = prepared.dense_specs[static_cast<std::size_t>(spec_index)];
        // set the specific parameter related to simulation step from the layer description
        ApplyDenseRuntimeOverridesFromLayer(&spec.config, &spec.queue_index, record.layer);
        // Dense layers are written directly into the runtime spec. The old
        // layer-build records are intentionally skipped to keep one dense graph
        // description in memory.
        // get the start index of the neuron in the spec subnetwork
        const int local_begin = spec.layout.stats.neuron_count;
        const int layer_count = record.layer.numberofneuron;
        // accumulate the neuron count
        spec.layout.stats.neuron_count += layer_count;
        // expand the neuron model id map
        spec.neuron_model_id_by_neuron.resize(
            static_cast<std::size_t>(spec.layout.stats.neuron_count),
            -1);
        spec.monitor_local_to_original_ids.resize(
            static_cast<std::size_t>(spec.layout.stats.neuron_count),
            -1);
        spec.monitor_candidate_mask.resize(
            static_cast<std::size_t>(spec.layout.stats.neuron_count),
            0);
        // create the neuron_model layer description
        DenseNeuronModelSpec model;
        model.model_id = static_cast<int>(spec.neuron_models.size());
        model.legacy_model_name = record.layer.ModelName;
        // get the neuronmodel decription
        const IDenseNeuronModel* dense_model =
            DenseNeuronModelFactory::Instance().FindByLegacyName(record.layer.ModelName);
        // Store the factory-owned model id; model.model_id remains the
        // build-time dense model range id.
        model.factory_model_id = dense_model != nullptr
                                     ? dense_model->FactoryModelId()
                                     : DenseNeuronModelFactory::kUnknownModelId;
        // set the model range
        model.range = DenseNeuronRange{local_begin, layer_count};
        // set the neuron model parameter
        FillLayerParams(&model,
                        record.layer,
                        DenseNeuronRange{local_begin, layer_count},
                        spec.layout.stats.neuron_count);
        // assign the model id to the neuron model id map
        for (int offset = 0; offset < layer_count; ++offset) {
            spec.neuron_model_id_by_neuron[static_cast<std::size_t>(local_begin + offset)] = model.model_id;
        }
        // add the model to the spec
        spec.internal_model_ids.push_back(model.model_id);
        spec.neuron_models.push_back(model);
        spec.layout.stats.model_count = static_cast<int>(spec.neuron_models.size());

        spec.update_timestep = (spec.neuron_models.size() == 1)
                                   ? std::max(1, record.layer.update_timestep)
                                   : std::min(spec.update_timestep, std::max(1, record.layer.update_timestep));

        int next_local = 0;
        // Count previously assigned dense-local neurons in this subnetwork.
        for (int neuron_id = 0; neuron_id < record.original_start; ++neuron_id) {
            if (original_to_dense_spec[static_cast<std::size_t>(neuron_id)] == spec_index) {
                ++next_local;
            }
        }
        for (int neuron_id = record.original_start; neuron_id < record.original_end; ++neuron_id) {
            // Every tagged original neuron is mapped to a dense-local id and is
            // intentionally left unmapped in original_to_main_neuron_id.
            original_to_dense_spec[static_cast<std::size_t>(neuron_id)] = spec_index;
            const int local_id = next_local++;
            original_to_dense_local[static_cast<std::size_t>(neuron_id)] = local_id;
            spec.monitor_local_to_original_ids[static_cast<std::size_t>(local_id)] = neuron_id;
            spec.monitor_candidate_mask[static_cast<std::size_t>(local_id)] =
                record.layer.isMonitored ? 1 : 0;
        }
    }

    // Output masks are indexed by dense-local neuron id. Initialize them once
    // after all dense layers have contributed their neurons, so edge routing
    // only has to mark output sources.
    for (std::size_t spec_index = 0; spec_index < prepared.dense_specs.size(); ++spec_index) {
        DenseSubnetworkBuildSpec& spec = prepared.dense_specs[spec_index];
        spec.output_neuron_mask.assign(static_cast<std::size_t>(spec.layout.stats.neuron_count), 0);
    }

    // Second pass routes every original edge into one of four buckets:
    // main->main, main->dense, dense->main, or dense->dense.
    ConnectionDescription* main_bucket_ptr = nullptr;
    int next_main_runtime_connection_index = 0;
    prepared.original_connection_weight_refs.clear();
    for (auto it = connection_list.begin(); it != connection_list.end(); ++it) {
        for (std::size_t edge_index = 0; edge_index < it->SourceNeuron.size(); ++edge_index) {
            const int source_original = it->SourceNeuron[edge_index];
            const int target_original = it->TargetNeuron[edge_index];
            const int source_dense_spec = original_to_dense_spec[static_cast<std::size_t>(source_original)];
            const int target_dense_spec = original_to_dense_spec[static_cast<std::size_t>(target_original)];
            RuntimeConnectionWeightRef weight_ref;
            weight_ref.initial_weight = it->Weight[edge_index];

            if (source_dense_spec < 0 && target_dense_spec < 0) {
                // Pure main-network edges are copied with compacted main ids.
                if (retain_main_connections && main_bucket_ptr == nullptr) {
                    main_bucket_ptr = &EnsureConnectionBucket(&prepared.main_connections);
                }
                weight_ref.owner = RuntimeConnectionWeightOwner::MainNetwork;
                weight_ref.runtime_weight_index = next_main_runtime_connection_index++;
                // add the pure main connection
                if (retain_main_connections) {
                    AppendConnection(main_bucket_ptr,
                                     prepared.original_to_main_neuron_id[static_cast<std::size_t>(source_original)],
                                     prepared.original_to_main_neuron_id[static_cast<std::size_t>(target_original)],
                                     it->Delay[edge_index],
                                     it->Weight[edge_index],
                                     it->MaxWeight[edge_index],
                                     it->Type[edge_index],
                                     it->SynapseRule[edge_index],
                                     it->TriggerRule[edge_index]);
                }
            } else if (source_dense_spec < 0 && target_dense_spec >= 0) {
                if (it->SynapseRule[edge_index] >= 0) {
                    return reject_build("main->dense SynapseRule is not supported; use a dense trigger relay neuron");
                }
                if (it->TriggerRule[edge_index] >= 0) {
                    return reject_build(
                        "main->dense TriggerRule is not supported directly; route ordinary excitatory input into TriggerRelayNeuronModel and attach the trigger rule to a dense-internal synapse");
                }
                // Main->dense edges become ordinary boundary inputs. External
                // trigger mediation is expressed by a dense-local relay neuron.
                // Main->dense edges become main-network connections into
                // synthetic interface neurons. The dense runtime reads those
                // interface slots instead of owning the external source neuron.
                // get the source layer
                const LayerRecord* source_layer = find_layer_record_by_neuron(source_original);
                // get the target subnetwork description
                DenseSubnetworkBuildSpec& target_spec =
                    prepared.dense_specs[static_cast<std::size_t>(target_dense_spec)];
                // get the source main id
                const int input_source_main_id =
                    prepared.original_to_main_neuron_id[static_cast<std::size_t>(source_original)];
                // get the target local id
                const int input_target_local_id =
                    original_to_dense_local[static_cast<std::size_t>(target_original)];
                const bool input_uses_current = source_layer != nullptr &&
                                                source_layer->layer.ModelName == "InputCurrentNeuronModel";
                // add the 
                const int slot_index = AppendDenseInputSlot(&target_spec,
                                                            input_target_local_id,
                                                            it->Type[edge_index],
                                                            input_uses_current,
                                                            input_source_main_id);
                AppendMainToDenseInputConnection(&target_spec,
                                                 input_source_main_id,
                                                 slot_index,
                                                 it->Delay[edge_index],
                                                 it->Weight[edge_index],
                                                 it->MaxWeight[edge_index],
                                                 it->Type[edge_index]);
            } else if (source_dense_spec >= 0 && target_dense_spec < 0) {
                // Dense->main boundary delivery stays synthetic-output only;
                // learning is kept inside same-subnetwork dense synapses.
                if (it->SynapseRule[edge_index] >= 0 || it->TriggerRule[edge_index] >= 0) {
                    return reject_build("dense->main SynapseRule/TriggerRule is not supported");
                }
                // Dense->main edges are represented as synthetic output
                // Interconnections after the main Network has been built.
                DenseSubnetworkBuildSpec& spec = prepared.dense_specs[static_cast<std::size_t>(source_dense_spec)];
                const int output_source_local_id =
                    original_to_dense_local[static_cast<std::size_t>(source_original)];
                spec.output_source_local_ids.push_back(output_source_local_id);
                spec.output_target_main_ids.push_back(
                    prepared.original_to_main_neuron_id[static_cast<std::size_t>(target_original)]);
                spec.output_delays.push_back(it->Delay[edge_index]);
                spec.output_weights.push_back(it->Weight[edge_index]);
                spec.output_types.push_back(it->Type[edge_index]);
                if (output_source_local_id >= 0 && output_source_local_id < spec.layout.stats.neuron_count) {
                    // mark the output neuron
                    spec.output_neuron_mask[static_cast<std::size_t>(output_source_local_id)] = 1;
                }
            } else if (source_dense_spec >= 0 && target_dense_spec >= 0 &&
                       source_dense_spec != target_dense_spec) {
                // Cross-subnetwork routes are interface routes, not internal
                // dense synapses, so dense learning metadata is rejected here.
                if (it->SynapseRule[edge_index] >= 0 || it->TriggerRule[edge_index] >= 0) {
                    return reject_build("cross-subnetwork dense->dense SynapseRule/TriggerRule is not supported");
                }
                // Dense->dense edges stay as boundary routes. The source dense
                // block exports a firing, then the event queue delivers it into
                // the target block's interface slot after the configured delay.
                DenseSubnetworkBuildSpec& source_spec =
                    prepared.dense_specs[static_cast<std::size_t>(source_dense_spec)];
                DenseSubnetworkBuildSpec& target_spec =
                    prepared.dense_specs[static_cast<std::size_t>(target_dense_spec)];
                const int source_local_id =
                    original_to_dense_local[static_cast<std::size_t>(source_original)];
                const int target_local_id =
                    original_to_dense_local[static_cast<std::size_t>(target_original)];
                const int target_slot = AppendDenseInputSlot(&target_spec,
                                                             target_local_id,
                                                             it->Type[edge_index],
                                                             false,
                                                             -1);
                // add the output connections to the main layer
                source_spec.output_dense_source_local_ids.push_back(source_local_id);
                source_spec.output_dense_target_spec_ids.push_back(target_dense_spec);
                source_spec.output_dense_target_slot_indices.push_back(target_slot);
                source_spec.output_dense_delays.push_back(it->Delay[edge_index]);
                source_spec.output_dense_weights.push_back(it->Weight[edge_index]);
                source_spec.output_dense_types.push_back(it->Type[edge_index]);
                if (source_local_id >= 0 && source_local_id < source_spec.layout.stats.neuron_count) {
                    // mark the neuron as the output neuron
                    source_spec.output_neuron_mask[static_cast<std::size_t>(source_local_id)] = 1;
                }
            } else if (source_dense_spec == target_dense_spec) {
                // Same-subnetwork dense edges are the only dense-owned learning
                // surface. Trigger-only edges are retained as trigger routes
                // that can update ordinary plastic synapses in the same block.
                // Dense-internal edges are packed directly into the runtime
                // layout. A parallel slice key keeps only the source/delay
                // data needed to generate pre_delay_slices later.
                DenseSubnetworkBuildSpec& spec = prepared.dense_specs[static_cast<std::size_t>(source_dense_spec)];
                const int source_local_id = original_to_dense_local[static_cast<std::size_t>(source_original)];
                const int target_local_id = original_to_dense_local[static_cast<std::size_t>(target_original)];
                const int type = it->Type[edge_index];
                const int post_model = spec.neuron_model_id_by_neuron[static_cast<std::size_t>(target_local_id)];
                // this is just a default fallback
                PendingChannel effect_channel = PendingChannel::ExcitatoryConductance;
                float effect_scale = 1.0f;
                // get the pending channle by the connection's type
                if (post_model < 0 ||
                    post_model >= static_cast<int>(spec.neuron_models.size()) ||
                    !DenseNeuronModelFactory::Instance().ResolveSpikeEffect(
                        spec.neuron_models[static_cast<std::size_t>(post_model)].factory_model_id,
                        type,
                        &effect_channel,
                        &effect_scale)) {
                    prepared.original_connection_weight_refs.push_back(weight_ref);
                    continue;
                }
                weight_ref.owner = RuntimeConnectionWeightOwner::DenseInternal;
                weight_ref.dense_subnetwork_index = source_dense_spec;
                weight_ref.runtime_weight_index =
                    static_cast<int>(spec.layout.synapses.weight.size());
                spec.layout.synapses.post_neuron.push_back(target_local_id);
                spec.layout.synapses.post_model.push_back(post_model);
                spec.layout.synapses.weight.push_back(it->Weight[edge_index]);
                spec.layout.synapses.max_weight.push_back(it->MaxWeight[edge_index]);
                spec.layout.synapses.type.push_back(static_cast<std::uint8_t>(type));
                spec.layout.synapses.effect_channel.push_back(static_cast<std::uint8_t>(effect_channel));
                spec.layout.synapses.effect_scale.push_back(effect_scale);
                // Learning metadata stays parallel to the flat synapse arrays
                // so device kernels can map one synapse to one CUDA thread.
                const int synapse_rule = it->SynapseRule[edge_index];
                const DenseLearningRuleInfo learning_info =
                    DenseLearningRuleFactory::Instance().DescribeLegacyRule(synapse_rule, dense_learning_rules);
                const int trigger_rule = it->TriggerRule[edge_index];
                const DenseLearningRuleInfo trigger_info =
                    DenseLearningRuleFactory::Instance().DescribeLegacyRule(trigger_rule, dense_learning_rules);
                if ((synapse_rule >= 0 && !learning_info.supported) ||
                    (trigger_rule >= 0 && !trigger_info.supported)) {
                    return reject_build("dense internal learning rule is not supported by the GPU backend");
                }
                // check this connection has a plastic learning rule or trigger rule
                const bool has_plastic_rule = synapse_rule >= 0 && learning_info.supported;
                const bool has_trigger_rule = trigger_rule >= 0 && trigger_info.supported;
                // Store only legacy rule bindings here. The finalizer asks the
                // learning factory to compile runtime model ids and hook flags.
                spec.layout.synapses.plastic_rule_id.push_back(has_plastic_rule ? synapse_rule : -1);
                // append the current post neuron size(the synapse index) as the plastic state index
                spec.layout.synapses.plastic_state_index.push_back(
                    has_plastic_rule ? static_cast<int>(spec.layout.synapses.post_neuron.size()) - 1 : -1);
                // append the whether trigger flag
                spec.layout.synapses.trigger_rule_id.push_back(has_trigger_rule ? trigger_rule : -1);
                // get the source local neuron id
                spec.synapse_source_local_ids.push_back(source_local_id);
                // get the delay
                spec.synapse_delay_slots.push_back(it->Delay[edge_index]);
            }
            prepared.original_connection_weight_refs.push_back(weight_ref);
        }
    }
    // scan all the subnetwork
    for (std::size_t spec_index = 0; spec_index < prepared.dense_specs.size(); ++spec_index) {
        DenseSubnetworkBuildSpec& spec = prepared.dense_specs[spec_index];
        // Multiple external sources can target the same dense neuron through
        // the same delivery kind. They share one dense interface slot, while
        // keeping one main-network connection per original external source.
        std::unordered_map<int, int> interface_slot_by_target_and_kind; // <target, kind>
        std::vector<int> deduped_input_source_main_ids;
        std::vector<int> deduped_input_interface_main_ids;
        std::vector<int> deduped_input_target_local_ids;
        std::vector<unsigned char> deduped_input_uses_current;
        std::vector<std::uint8_t> deduped_input_pending_channels;
        std::vector<float> deduped_input_scales;
        std::vector<int> deduped_input_representative_source_main_ids;
        std::vector<int> deduped_connection_source_main_ids;
        std::vector<int> deduped_connection_slot_indices;
        std::vector<int> deduped_connection_delays;
        std::vector<float> deduped_connection_weights;
        std::vector<float> deduped_connection_max_weights;
        std::vector<int> deduped_connection_types;
        std::vector<std::uint8_t> deduped_connection_pending_channels;
        std::vector<float> deduped_connection_effect_scales;
        // reserve the memory
        deduped_input_source_main_ids.reserve(spec.input_source_main_ids.size());
        deduped_input_interface_main_ids.reserve(spec.input_interface_main_ids.size());
        deduped_input_target_local_ids.reserve(spec.input_target_local_ids.size());
        deduped_input_uses_current.reserve(spec.input_uses_current.size());
        deduped_input_pending_channels.reserve(spec.input_pending_channels.size());
        deduped_input_scales.reserve(spec.input_scales.size());
        deduped_input_representative_source_main_ids.reserve(spec.input_representative_source_main_ids.size());
        deduped_connection_source_main_ids.reserve(spec.input_connection_source_main_ids.size());
        deduped_connection_slot_indices.reserve(spec.input_connection_slot_indices.size());
        deduped_connection_delays.reserve(spec.input_connection_delays.size());
        deduped_connection_weights.reserve(spec.input_connection_weights.size());
        deduped_connection_max_weights.reserve(spec.input_connection_max_weights.size());
        deduped_connection_types.reserve(spec.input_connection_types.size());
        deduped_connection_pending_channels.reserve(spec.input_connection_pending_channels.size());
        deduped_connection_effect_scales.reserve(spec.input_connection_effect_scales.size());
        std::vector<int> old_slot_to_new_slot(spec.input_target_local_ids.size(), -1);
        // Deduplicate interface slots independently from main-network input
        // connections; dense-to-dense routes also target these slots.
        // buid the list of the merged input slot
        for (std::size_t input_index = 0; input_index < spec.input_target_local_ids.size(); ++input_index) {
            // caculate the target slot index key
            const int dedupe_key =
                spec.input_target_local_ids[input_index] * DenseNeuronModelFactory::kPendingChannelKeyStride +
                static_cast<int>(spec.input_pending_channels[input_index]);
            // check whether it is a new key from the map(the map <target channel key, interface_slot_index>)
            std::unordered_map<int, int>::const_iterator found = interface_slot_by_target_and_kind.find(dedupe_key);
            int interface_slot_index = -1;
            if (found == interface_slot_by_target_and_kind.end()) {
                // if this is a new target slot
                interface_slot_index = static_cast<int>(deduped_input_target_local_ids.size());
                interface_slot_by_target_and_kind[dedupe_key] = interface_slot_index;
                // add the new interface slot
                // add the source_main_ids
                deduped_input_source_main_ids.push_back(spec.input_source_main_ids[input_index]);
                // init the interface main ids
                deduped_input_interface_main_ids.push_back(-1);
                // add the target neuron local ids
                deduped_input_target_local_ids.push_back(spec.input_target_local_ids[input_index]);
                // add the uses current flag
                deduped_input_uses_current.push_back(spec.input_uses_current[input_index]);
                // add the pending channels and scale
                deduped_input_pending_channels.push_back(spec.input_pending_channels[input_index]);
                deduped_input_scales.push_back(spec.input_scales[input_index]);
                // add the representative source main ids
                const int representative_source =
                    spec.input_representative_source_main_ids[input_index] >= 0
                        ? spec.input_representative_source_main_ids[input_index]
                        : spec.input_source_main_ids[input_index];
                deduped_input_representative_source_main_ids.push_back(representative_source);
            } else {
                interface_slot_index = found->second;
            }
            // merge the old slots to new slot(map the origin input index to the merged interface_slot_index)
            old_slot_to_new_slot[input_index] = interface_slot_index;
        }
        // Deduplicate main-network input connections to interface slots.
        // build the list of the merged input connections
        for (std::size_t input_index = 0; input_index < spec.input_connection_source_main_ids.size(); ++input_index) {
            // get the old slot index
            const int old_slot_index = spec.input_connection_slot_indices[input_index];
            // check whether is the slot index legal
            if (old_slot_index < 0 ||
                old_slot_index >= static_cast<int>(old_slot_to_new_slot.size())) {
                continue;
            }
            // map the old slot index to the new slot index
            const int interface_slot_index =
                old_slot_to_new_slot[static_cast<std::size_t>(old_slot_index)];
            if (interface_slot_index < 0) {
                continue;
            }
            // build the connection description list
            deduped_connection_source_main_ids.push_back(spec.input_connection_source_main_ids[input_index]);
            deduped_connection_slot_indices.push_back(interface_slot_index);
            deduped_connection_delays.push_back(spec.input_connection_delays[input_index]);
            deduped_connection_weights.push_back(spec.input_connection_weights[input_index]);
            deduped_connection_max_weights.push_back(spec.input_connection_max_weights[input_index]);
            deduped_connection_types.push_back(spec.input_connection_types[input_index]);
            deduped_connection_pending_channels.push_back(spec.input_connection_pending_channels[input_index]);
            deduped_connection_effect_scales.push_back(spec.input_connection_effect_scales[input_index]);
        }
        spec.input_source_main_ids.swap(deduped_input_source_main_ids);
        spec.input_interface_main_ids.swap(deduped_input_interface_main_ids);
        spec.input_target_local_ids.swap(deduped_input_target_local_ids);
        spec.input_uses_current.swap(deduped_input_uses_current);
        spec.input_pending_channels.swap(deduped_input_pending_channels);
        spec.input_scales.swap(deduped_input_scales);
        spec.input_representative_source_main_ids.swap(deduped_input_representative_source_main_ids);
        spec.input_connection_source_main_ids.swap(deduped_connection_source_main_ids);
        spec.input_connection_slot_indices.swap(deduped_connection_slot_indices);
        spec.input_connection_delays.swap(deduped_connection_delays);
        spec.input_connection_weights.swap(deduped_connection_weights);
        spec.input_connection_max_weights.swap(deduped_connection_max_weights);
        spec.input_connection_types.swap(deduped_connection_types);
        spec.input_connection_pending_channels.swap(deduped_connection_pending_channels);
        spec.input_connection_effect_scales.swap(deduped_connection_effect_scales);
        // Deduplicate dense->dense connections to runtime interface slots.
        // scan all the subnetworks to get the cross subnetwork connection
        for (std::size_t source_spec_index = 0;
             source_spec_index < prepared.dense_specs.size();
             ++source_spec_index) {
            // get the source subnetwork description
            DenseSubnetworkBuildSpec& source_spec = prepared.dense_specs[source_spec_index];
            for (std::size_t output_index = 0;
                 output_index < source_spec.output_dense_target_spec_ids.size();
                 ++output_index) {
                if (source_spec.output_dense_target_spec_ids[output_index] != static_cast<int>(spec_index)) {
                    continue;
                }
                // get the old target subnetwork slot index
                const int old_slot_index = source_spec.output_dense_target_slot_indices[output_index];
                // check whether it is legal
                if (old_slot_index < 0 ||
                    old_slot_index >= static_cast<int>(old_slot_to_new_slot.size())) {
                    source_spec.output_dense_target_slot_indices[output_index] = -1;
                    continue;
                }
                // map the old slot index to the new slot index
                source_spec.output_dense_target_slot_indices[output_index] =
                    old_slot_to_new_slot[static_cast<std::size_t>(old_slot_index)];
            }
        }

        if (spec.input_target_local_ids.empty()) {
            continue;
        }

        // Synthetic main-network interface neurons are needed only for
        // main->dense edges. Dense->dense edges target runtime interface slots
        // directly through synthetic Interconnections created after all dense
        // runtimes exist.
        // find the slot used by the main network's input
        std::vector<unsigned char> slot_used_by_main_input(
            spec.input_target_local_ids.size(), 0);
        // Mark all the slots that are used by the main network.
        for (std::size_t input_index = 0;
             input_index < spec.input_connection_slot_indices.size();
             ++input_index) {
            // get the each input's slot index(only the main->subnetwork connection are add to the input_connection_slot_indices)
            const int slot_index = spec.input_connection_slot_indices[input_index];
            if (slot_index >= 0 &&
                slot_index < static_cast<int>(slot_used_by_main_input.size())) {
                slot_used_by_main_input[static_cast<std::size_t>(slot_index)] = 1;
            }
        }
        int spike_interface_start = -1;
        int current_interface_start = -1;
        int spike_interface_count = 0;
        int current_interface_count = 0;
        // scan all the input from the main or the other subnetwork, caculate the number of the inputSpike and inputCurrent neurons
        for (std::size_t input_index = 0; input_index < spec.input_target_local_ids.size(); ++input_index) {
            if (slot_used_by_main_input[input_index] == 0) {
                continue;
            }
            if (spec.input_uses_current[input_index] != 0) {
                ++current_interface_count;
            } else {
                ++spike_interface_count;
            }
        }
        // add the inputSpike and inputCurrent neurons to the main network
        if (spike_interface_count > 0) {
            NeuronLayerDescription interface_layer;
            interface_layer.ModelName = "InputSpikeNeuronModel";
            interface_layer.numberofneuron = spike_interface_count;
            interface_layer.isMonitored = false;
            interface_layer.isOutput = false;
            prepared.main_layers.push_back(interface_layer);
            spike_interface_start = next_main_neuron_id;
            next_main_neuron_id += interface_layer.numberofneuron;
        }
        if (current_interface_count > 0) {
            NeuronLayerDescription interface_layer;
            interface_layer.ModelName = "InputCurrentNeuronModel";
            interface_layer.numberofneuron = current_interface_count;
            interface_layer.isMonitored = false;
            interface_layer.isOutput = false;
            prepared.main_layers.push_back(interface_layer);
            current_interface_start = next_main_neuron_id;
            next_main_neuron_id += interface_layer.numberofneuron;
        }

        if (!spec.input_connection_source_main_ids.empty() && main_bucket_ptr == nullptr) {
            main_bucket_ptr = &EnsureConnectionBucket(&prepared.main_connections);
        }
        int spike_offset = 0;
        int current_offset = 0;
        // caculate the interface neuron id
        for (std::size_t input_index = 0; input_index < spec.input_target_local_ids.size(); ++input_index) {
            // if this is not a main->subnetwork connection, skip it
            if (slot_used_by_main_input[input_index] == 0) {
                spec.input_interface_main_ids[input_index] = -1;
                continue;
            }
            // if this is a main->subnetwork connection, add the connection to the main network
            if (spec.input_uses_current[input_index] != 0) {
                spec.input_interface_main_ids[input_index] = current_interface_start + current_offset++;
            } else {
                spec.input_interface_main_ids[input_index] = spike_interface_start + spike_offset++;
            }
        }
        // add connection to the main network
        for (std::size_t input_index = 0; input_index < spec.input_connection_source_main_ids.size(); ++input_index) {
            const int interface_slot_index = spec.input_connection_slot_indices[input_index];
            if (interface_slot_index < 0 ||
                interface_slot_index >= static_cast<int>(spec.input_target_local_ids.size())) {
                continue;
            }
            if (main_bucket_ptr == nullptr ||
                spec.input_interface_main_ids[static_cast<std::size_t>(interface_slot_index)] < 0) {
                continue;
            }
            spec.input_connection_delays[input_index] = std::max(0, spec.input_connection_delays[input_index]);
            AppendConnection(main_bucket_ptr,
                             spec.input_connection_source_main_ids[input_index],
                             spec.input_interface_main_ids[static_cast<std::size_t>(interface_slot_index)],
                             spec.input_connection_delays[input_index],
                             spec.input_connection_weights[input_index],
                             spec.input_connection_max_weights[input_index],
                             spec.input_connection_types[input_index],
                             -1,
                             -1);
        }
    }

    return prepared;
}

}  // namespace

PreparedSimulationBuild PrepareBlackBoxDenseBuild(
    const std::list<NeuronLayerDescription>& neuron_layer_list,
    const std::list<ConnectionDescription>& connection_list,
    const std::list<LearningRuleDescription>& learning_rule_list,
    float basetimestep) {
    return PrepareBlackBoxDenseBuildImpl(
        neuron_layer_list, connection_list, learning_rule_list, basetimestep, true);
}

PreparedSimulationBuild PrepareBlackBoxDenseBuildStreaming(
    const std::list<NeuronLayerDescription>& neuron_layer_list,
    const streaming::NbnetReader& reader,
    const std::list<LearningRuleDescription>& learning_rule_list,
    float basetimestep) {
    return PrepareBlackBoxDenseBuildImpl(
        neuron_layer_list, NbnetConnectionBlockRange(reader), learning_rule_list, basetimestep, false);
}

}  // namespace sim_support
}  // namespace npgr
