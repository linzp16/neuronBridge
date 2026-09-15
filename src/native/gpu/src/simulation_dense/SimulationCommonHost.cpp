#include "simulation_dense/SimulationCommonHost.h"

#include "simulation_dense/DenseBuildShared.h"
#include "simulation_dense/DenseInterfaceSpikeNeuronModel.h"
#include "simulation_dense/DenseSubnetworkBuildFinalizer.h"
#include "simulation_dense/DenseSubnetworkLayoutCompactor.h"
#include "simulation_dense/DenseSubnetworkModel.h"
#include "simulation_dense/DenseSubnetworkUpdateEvent.h"
#include "bridge/LegacyNetworkBridge.h"
#include "debug_monitor/DebugMonitorTypes.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "source_file_realtime_v1_async/EventQueue/inc/TimingWheelEventQueue.h"
#include "source_file_realtime_v1_async/Event/inc/EndSimulationTime.h"
#include "source_file_realtime_v1_async/Event/inc/TimeEventUpdateNeuron.h"
#include "source_file_realtime_v1_async/Event/inc/Synchronize/SynchronizeActivityEvent.h"
#include "source_file_realtime_v1_async/Event/inc/Synchronize/SaveWeightEvent.h"
#include "source_file_realtime_v1_async/Event/inc/Synchronize/CommunicationEvent.h"
#include "source_file_realtime_v1_async/Event/inc/Outer/OuterUpdateEvent.h"
#include "source_file_realtime_v1_async/Event/inc/InputConv/UpdateInputConvEvent.h"
#include "source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"
#include "source_file_realtime_v1_async/InputSpikeDriver/inc/ArrayInputSpikeDriver.h"
#include "source_file_realtime_v1_async/InputCurrentDriver/inc/ArrayInputCurrentDriver.h"
#include "source_file_realtime_v1_async/communication/inc/ArrayOutputSpikeDriver.h"
#include "source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeBuffer.h"
#include "source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"
#include "source_file_realtime_v1_async/Network/inc/Network.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include "source_file_realtime_v1_async/Simulation/inc/RealTimeRestriction.h"

#if NPGR_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sstream>

namespace npgr {
namespace {

Interconnections* CreateDenseToMainConnection(Network* network,
                                              int target_main_id,
                                              float weight,
                                              int type,
                                              int delay) {
    if (network == nullptr || target_main_id < 0) {
        return nullptr;
    }
    Interconnections* synthetic_connection = new Interconnections();
    synthetic_connection->SetTargetNeuron(&(network->neurons[target_main_id]));
    synthetic_connection->SetTargetNeuronModel(network->neurons[target_main_id].neuron_model);
    synthetic_connection->SetTargetNeuronModelIndex(
        network->neurons[target_main_id].index_in_NeuronModel);
    synthetic_connection->SetWeight(weight);
    synthetic_connection->SetType(type);
    synthetic_connection->SetDelay(delay);
    if (synthetic_connection->TargetNeuronModel != nullptr) {
        synthetic_connection->TargetNeuronModel->CheckType(synthetic_connection);
    }
    return synthetic_connection;
}

int CountLayerNeurons(const std::list<NeuronLayerDescription>& layers) {
    int count = 0;
    for (std::list<NeuronLayerDescription>::const_iterator it = layers.begin();
         it != layers.end();
         ++it) {
        count += it->numberofneuron;
    }
    return count;
}

void AppendOuterDynamicConnection(ConnectionDescription* connection,
                                  int source,
                                  int target,
                                  int type,
                                  float weight,
                                  int delay) {
    connection->SourceNeuron.push_back(source);
    connection->TargetNeuron.push_back(target);
    connection->Type.push_back(type);
    connection->Weight.push_back(weight);
    connection->MaxWeight.push_back(weight);
    connection->Delay.push_back(delay);
    connection->SynapseRule.push_back(-1);
    connection->TriggerRule.push_back(-1);
}

void AppendOuterDynamicInterfaceNetwork(
    const Simulation* simulation,
    const std::list<OuterDynamicConnectionDescription>& outer_dynamic_connection_list,
    const std::vector<int>& original_to_main_neuron_id,
    std::list<NeuronLayerDescription>* main_layers,
    std::list<ConnectionDescription>* main_connections) {
    if (simulation == nullptr ||
        outer_dynamic_connection_list.empty() ||
        main_layers == nullptr ||
        main_connections == nullptr) {
        return;
    }

    std::vector<std::vector<int> > endpoint_main_id_by_outer;
    endpoint_main_id_by_outer.resize(simulation->OuterDynamicModelList.size());
    std::vector<int> endpoint_outer_ids;
    std::vector<int> endpoint_joint_ids;
    const int endpoint_start = CountLayerNeurons(*main_layers);
    int endpoint_count = 0;
    for (std::size_t outer_id = 0; outer_id < simulation->OuterDynamicModelList.size(); ++outer_id) {
        OuterDynamicModel* model = simulation->OuterDynamicModelList[outer_id];
        if (model == nullptr) {
            continue;
        }
        const int joint_count = model->GetJointCount();
        if (joint_count < 0) {
            throw std::runtime_error("OuterDynamic model returned a negative joint count.");
        }
        endpoint_main_id_by_outer[outer_id].assign(static_cast<std::size_t>(joint_count), -1);
        for (int joint = 0; joint < joint_count; ++joint) {
            endpoint_main_id_by_outer[outer_id][static_cast<std::size_t>(joint)] =
                endpoint_start + endpoint_count;
            endpoint_outer_ids.push_back(static_cast<int>(outer_id));
            endpoint_joint_ids.push_back(joint);
            ++endpoint_count;
        }
    }
    if (endpoint_count <= 0) {
        return;
    }

    NeuronLayerDescription endpoint_layer;
    endpoint_layer.numberofneuron = endpoint_count;
    endpoint_layer.ModelName = "OuterDynamicInterfaceNeuronModel";
    endpoint_layer.update_timestep = 1;
    endpoint_layer.isMonitored = false;
    endpoint_layer.isOutput = false;
    endpoint_layer.NeuronParameter["outer_dynamic_ids_by_neuron"] = endpoint_outer_ids;
    endpoint_layer.NeuronParameter["outer_dynamic_joint_ids_by_neuron"] = endpoint_joint_ids;
    main_layers->push_back(endpoint_layer);

    for (std::list<OuterDynamicConnectionDescription>::const_iterator desc_it =
             outer_dynamic_connection_list.begin();
         desc_it != outer_dynamic_connection_list.end();
         ++desc_it) {
        const std::size_t connection_count = desc_it->SourceNeuron.size();
        if (desc_it->TargetOuterDynamic.size() != connection_count ||
            desc_it->TargetJoint.size() != connection_count ||
            desc_it->Type.size() != connection_count ||
            desc_it->Weight.size() != connection_count ||
            desc_it->Delay.size() != connection_count) {
            throw std::runtime_error("OuterDynamicConnectionDescription arrays have mismatched sizes.");
        }
        ConnectionDescription translated;
        for (std::size_t index = 0; index < connection_count; ++index) {
            const int original_source = desc_it->SourceNeuron[index];
            int source = original_source;
            if (!original_to_main_neuron_id.empty()) {
                if (original_source < 0 ||
                    original_source >= static_cast<int>(original_to_main_neuron_id.size()) ||
                    original_to_main_neuron_id[static_cast<std::size_t>(original_source)] < 0) {
                    throw std::runtime_error("OuterDynamic connection source is not a main-network neuron.");
                }
                source = original_to_main_neuron_id[static_cast<std::size_t>(original_source)];
            }
            const int outer_id = desc_it->TargetOuterDynamic[index];
            const int joint_id = desc_it->TargetJoint[index];
            const int type = desc_it->Type[index];
            const int delay = desc_it->Delay[index];
            if (outer_id < 0 ||
                outer_id >= static_cast<int>(endpoint_main_id_by_outer.size()) ||
                joint_id < 0 ||
                joint_id >= static_cast<int>(endpoint_main_id_by_outer[static_cast<std::size_t>(outer_id)].size())) {
                throw std::runtime_error("OuterDynamic connection target outer id or joint id is invalid.");
            }
            if (type != 0 && type != 1) {
                throw std::runtime_error("OuterDynamic connection type must be 0(excitatory) or 1(inhibitory).");
            }
            if (delay < 0) {
                throw std::runtime_error("OuterDynamic connection delay must be non-negative.");
            }
            const int target =
                endpoint_main_id_by_outer[static_cast<std::size_t>(outer_id)][static_cast<std::size_t>(joint_id)];
            AppendOuterDynamicConnection(&translated, source, target, type, desc_it->Weight[index], delay);
        }
        if (!translated.SourceNeuron.empty()) {
            main_connections->push_back(translated);
        }
    }
}

Interconnections* CreateDenseToDenseInterfaceConnection(DenseSubnetworkModel* target_model,
                                                        int target_slot,
                                                        float weight,
                                                        int type,
                                                        int delay) {
    if (target_model == nullptr ||
        target_model->interface_spike_model() == nullptr ||
        target_slot < 0) {
        return nullptr;
    }
    Interconnections* synthetic_connection = new Interconnections();
    synthetic_connection->SetTargetNeuron(nullptr);
    synthetic_connection->SetTargetNeuronModel(target_model->interface_spike_model());
    synthetic_connection->SetTargetNeuronModelIndex(target_slot);
    synthetic_connection->SetWeight(weight);
    synthetic_connection->SetType(type);
    synthetic_connection->SetDelay(delay);
    return synthetic_connection;
}

int CountDenseSubnetworksByName(const Simulation* simulation, const std::string& name) {
    if (simulation == nullptr) {
        return 0;
    }
    int count = 0;
    for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
        if (simulation->dense_subnetworks[index] != nullptr &&
            simulation->dense_subnetworks[index]->name() == name) {
            ++count;
        }
    }
    return count;
}

int FindDenseSubnetworkIndexByName(const Simulation* simulation, const std::string& name) {
    if (simulation == nullptr) {
        return -1;
    }
    for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
        if (simulation->dense_subnetworks[index] != nullptr &&
            simulation->dense_subnetworks[index]->name() == name) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

bool BuildInputConvDenseCopySpecs(const InputConvDescription& description,
                                  int output_count,
                                  const DenseSubnetworkModel* dense_model,
                                  std::vector<DeviceToDeviceInterfaceCopySpec>* copy_specs,
                                  std::string* reason) {
    if (copy_specs == nullptr) {
        if (reason != nullptr) {
            *reason = "copy_specs output must not be null";
        }
        return false;
    }
    copy_specs->clear();
    if (description.output_source_indices.size() != description.output_target_neuron_ids.size()) {
        if (reason != nullptr) {
            *reason = "InputConv dense binding source and target route counts differ";
        }
        return false;
    }
    if (!description.output_scales.empty() &&
        description.output_scales.size() != description.output_source_indices.size()) {
        if (reason != nullptr) {
            *reason = "InputConv dense binding scale count must match route count";
        }
        return false;
    }
    if (description.output_source_indices.empty()) {
        if (reason != nullptr) {
            *reason = "InputConv dense binding requires at least one source->target route";
        }
        return false;
    }
    copy_specs->reserve(description.output_source_indices.size());
    for (std::size_t index = 0; index < description.output_source_indices.size(); ++index) {
        const int source_index = description.output_source_indices[index];
        if (source_index < 0 || source_index >= output_count) {
            if (reason != nullptr) {
                *reason = "InputConv dense binding source index is out of range";
            }
            return false;
        }
        DeviceToDeviceInterfaceCopySpec spec;
        // Direct target-neuron routing keeps D2D binding independent of the
        // dense interface-slot table used by main->dense spike delivery.
        spec.interface_slot = -1;
        const int requested_target = description.output_target_neuron_ids[index];
        const int compact_target =
            dense_model != nullptr
                ? dense_model->LocalNeuronIdForOriginalGlobal(requested_target)
                : -1;
        // Public InputConv dense bindings use original neuron ids. Direct
        // local ids are still accepted for explicitly constructed reference
        // dense specs that do not carry an original-id monitor map.
        spec.target_neuron = compact_target >= 0 ? compact_target : requested_target;
        spec.pending_channel = static_cast<std::uint8_t>(description.output_pending_channel);
        spec.source_index = source_index;
        spec.scale = description.output_scales.empty()
                         ? description.output_scale
                         : description.output_scales[index];
        spec.overwrite = description.output_overwrite;
        copy_specs->push_back(spec);
    }
    return true;
}

int CountNeurons(const std::list<NeuronLayerDescription>& layers) {
    int count = 0;
    for (std::list<NeuronLayerDescription>::const_iterator it = layers.begin();
         it != layers.end();
         ++it) {
        count += it->numberofneuron;
    }
    return count;
}

void AddInputConvMainCurrentRoutes(const std::list<InputConvDescription>& input_conv_list,
                                   const std::vector<int>& original_to_main_neuron_id,
                                   std::list<NeuronLayerDescription>* main_layers,
                                   std::list<ConnectionDescription>* main_connections,
                                   std::vector<std::vector<int> >* source_ids_by_input_conv) {
    if (main_layers == nullptr || main_connections == nullptr || source_ids_by_input_conv == nullptr) {
        return;
    }
    source_ids_by_input_conv->clear();
    source_ids_by_input_conv->resize(input_conv_list.size());
    int next_source_id = CountNeurons(*main_layers);
    std::size_t input_conv_index = 0;
    for (std::list<InputConvDescription>::const_iterator it = input_conv_list.begin();
         it != input_conv_list.end();
         ++it, ++input_conv_index) {
        const InputConvDescription& description = *it;
        if (description.output_target != InputConvOutputTarget::MainNetwork) {
            continue;
        }
        if (description.target_dense_subnetwork_name.empty() == false) {
            throw std::runtime_error("InputConv main-network binding must not name a dense subnetwork");
        }
        if (description.output_source_indices.size() != description.output_target_neuron_ids.size()) {
            throw std::runtime_error("InputConv main-network source and target route counts differ");
        }
        if (!description.output_scales.empty() &&
            description.output_scales.size() != description.output_source_indices.size()) {
            throw std::runtime_error("InputConv main-network scale count must match route count");
        }
        const int route_count = static_cast<int>(description.output_source_indices.size());
        if (route_count == 0) {
            continue;
        }
        NeuronLayerDescription input_current_layer;
        input_current_layer.ModelName = "InputCurrentNeuronModel";
        input_current_layer.numberofneuron = route_count;
        input_current_layer.update_timestep = 1;
        input_current_layer.isMonitored = false;
        input_current_layer.isOutput = false;
        main_layers->push_back(input_current_layer);

        ConnectionDescription current_connections;
        current_connections.SourceNeuron.reserve(static_cast<std::size_t>(route_count));
        current_connections.TargetNeuron.reserve(static_cast<std::size_t>(route_count));
        current_connections.Type.reserve(static_cast<std::size_t>(route_count));
        current_connections.Weight.reserve(static_cast<std::size_t>(route_count));
        current_connections.MaxWeight.reserve(static_cast<std::size_t>(route_count));
        current_connections.Delay.reserve(static_cast<std::size_t>(route_count));
        current_connections.SynapseRule.reserve(static_cast<std::size_t>(route_count));
        current_connections.TriggerRule.reserve(static_cast<std::size_t>(route_count));
        std::vector<int>& source_ids =
            (*source_ids_by_input_conv)[input_conv_index];
        source_ids.reserve(static_cast<std::size_t>(route_count));
        for (int route_index = 0; route_index < route_count; ++route_index) {
            const int source_neuron_id = next_source_id + route_index;
            const int original_target_neuron_id =
                description.output_target_neuron_ids[static_cast<std::size_t>(route_index)];
            if (original_target_neuron_id < 0 ||
                original_target_neuron_id >= static_cast<int>(original_to_main_neuron_id.size()) ||
                original_to_main_neuron_id[static_cast<std::size_t>(original_target_neuron_id)] < 0) {
                throw std::runtime_error("InputConv main-network target neuron is not in the main network");
            }
            const int target_neuron_id =
                original_to_main_neuron_id[static_cast<std::size_t>(original_target_neuron_id)];
            source_ids.push_back(source_neuron_id);
            current_connections.SourceNeuron.push_back(source_neuron_id);
            current_connections.TargetNeuron.push_back(target_neuron_id);
            // Type 3 is the legacy current-input connection type.
            current_connections.Type.push_back(3);
            current_connections.Weight.push_back(1.0f);
            current_connections.MaxWeight.push_back(1.0f);
            current_connections.Delay.push_back(0);
            current_connections.SynapseRule.push_back(-1);
            current_connections.TriggerRule.push_back(-1);
        }
        main_connections->push_back(current_connections);
        next_source_id += route_count;
    }
}

void BindInputConvOutputDescriptions(Simulation* simulation,
                                     const std::vector<InputConvDescription>& input_conv_list) {
    if (simulation == nullptr) {
        return;
    }
    if (input_conv_list.size() != simulation->InputConvModelList.size()) {
        throw std::runtime_error("InputConv model count does not match description count");
    }
    for (std::size_t model_index = 0;
         model_index < simulation->InputConvModelList.size();
         ++model_index) {
        const InputConvDescription& description = input_conv_list[model_index];
        InputConvModel* model = simulation->InputConvModelList[model_index];
        if (model == nullptr) {
            throw std::runtime_error("InputConv binding found a null model");
        }
        if (description.output_target == InputConvOutputTarget::MainNetwork) {
            if (!description.target_dense_subnetwork_name.empty()) {
                throw std::runtime_error("InputConv main-network binding must not name a dense subnetwork");
            }
            if (model->HasDeviceOutputBuffer() &&
                model->GetOutputCount() > 0 &&
                !description.output_source_indices.empty()) {
                if (description.output_source_indices.size() !=
                    simulation->InputConvMainCurrentSourceNeuronIds[model_index].size()) {
                    throw std::runtime_error("InputConv main-network current source route count mismatch");
                }
            }
            continue;
        }
        if (description.output_target != InputConvOutputTarget::DenseSubnetwork) {
            throw std::runtime_error("InputConv output target is not recognized");
        }
        if (description.target_dense_subnetwork_name.empty()) {
            throw std::runtime_error("InputConv dense binding requires target_dense_subnetwork_name");
        }
        const int matching_dense_count =
            CountDenseSubnetworksByName(simulation, description.target_dense_subnetwork_name);
        if (matching_dense_count != 1) {
            std::ostringstream oss;
            oss << "InputConv dense binding requires exactly one dense subnetwork named "
                << description.target_dense_subnetwork_name;
            throw std::runtime_error(oss.str());
        }
        const int dense_index =
            FindDenseSubnetworkIndexByName(simulation, description.target_dense_subnetwork_name);
        if (dense_index < 0 ||
            dense_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
            simulation->dense_subnetworks[static_cast<std::size_t>(dense_index)] == nullptr) {
            throw std::runtime_error("InputConv dense binding resolved an invalid dense subnetwork");
        }
        if (!model->HasDeviceOutputBuffer()) {
            throw std::runtime_error("InputConv dense binding requires a device output buffer");
        }
        std::string reason;
        std::vector<DeviceToDeviceInterfaceCopySpec> copy_specs;
        if (!BuildInputConvDenseCopySpecs(
                description,
                model->GetOutputCount(),
                simulation->dense_subnetworks[static_cast<std::size_t>(dense_index)],
                &copy_specs,
                &reason)) {
            throw std::runtime_error("InputConv dense binding is invalid: " + reason);
        }
        if (!simulation->dense_subnetworks[static_cast<std::size_t>(dense_index)]
                 ->BindInterfaceDeviceSource(
                     model->GetDeviceOutputBuffer(),
                     model->GetOutputCount(),
                     copy_specs,
                     &reason)) {
            throw std::runtime_error("InputConv dense binding failed: " + reason);
        }
    }
}

}  // namespace

namespace sim_support {

void AddExternalSpikeActivityTranslated(Simulation* simulation,
                                        const std::vector<int>& event_time,
                                        const std::vector<int>& event_neuron_id) {
    if (simulation == nullptr || simulation->inputSpikeDriver == nullptr ||
        simulation->EventHeap == nullptr || simulation->network == nullptr ||
        event_time.empty()) {
        return;
    }

    std::vector<int> translated_times;
    std::vector<int> translated_neurons;
    translated_times.reserve(event_time.size());
    translated_neurons.reserve(event_neuron_id.size());
    for (std::size_t index = 0; index < event_time.size() && index < event_neuron_id.size(); ++index) {
        const int translated_id = TranslateOriginalNeuronIdToMain(simulation, event_neuron_id[index]);
        if (translated_id < 0) {
            continue;
        }
        translated_times.push_back(event_time[index]);
        translated_neurons.push_back(translated_id);
    }
    if (!translated_times.empty()) {
        simulation->inputSpikeDriver->LoadInputSpike(
            simulation->EventHeap,
            simulation->network,
            translated_times.size(),
            translated_times.data(),
            translated_neurons.data());
    }
}

void AddExternalCurrentActivityTranslated(Simulation* simulation,
                                          const std::vector<int>& event_time,
                                          const std::vector<int>& event_neuron_id,
                                          const std::vector<float>& current) {
    if (simulation == nullptr || simulation->inputCurrentDriver == nullptr ||
        simulation->EventHeap == nullptr || simulation->network == nullptr ||
        event_time.empty()) {
        return;
    }

    std::vector<int> translated_times;
    std::vector<int> translated_neurons;
    std::vector<float> translated_currents;
    translated_times.reserve(event_time.size());
    translated_neurons.reserve(event_neuron_id.size());
    translated_currents.reserve(current.size());
    for (std::size_t index = 0;
         index < event_time.size() && index < event_neuron_id.size() && index < current.size();
         ++index) {
        const int translated_id = TranslateOriginalNeuronIdToMain(simulation, event_neuron_id[index]);
        if (translated_id < 0) {
            continue;
        }
        translated_times.push_back(event_time[index]);
        translated_neurons.push_back(translated_id);
        translated_currents.push_back(current[index]);
    }
    if (!translated_times.empty()) {
        simulation->inputCurrentDriver->LoadInputCurrent(
            simulation->EventHeap,
            simulation->network,
            translated_times.size(),
            translated_times.data(),
            translated_neurons.data(),
            translated_currents.data());
    }
}

void ResetDenseSubnetworks(Simulation* simulation) {
    if (simulation == nullptr) {
        return;
    }
    for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
        std::string reason;
        if (simulation->dense_subnetworks[index] != nullptr) {
            simulation->dense_subnetworks[index]->Reset(&reason);
        }
    }
}

bool IsDenseManagedModel(const Simulation* simulation, const NeuronModel* model) {
    if (simulation == nullptr || model == nullptr) {
        return false;
    }
    for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
        if (simulation->dense_subnetworks[index] != nullptr &&
            simulation->dense_subnetworks[index]->OwnsInternalModel(model)) {
            return true;
        }
    }
    return false;
}

int TranslateOriginalNeuronIdToMain(const Simulation* simulation, int original_neuron_id) {
    if (simulation == nullptr || original_neuron_id < 0 ||
        original_neuron_id >= static_cast<int>(simulation->original_to_main_neuron_id.size())) {
        return -1;
    }
    return simulation->original_to_main_neuron_id[static_cast<std::size_t>(original_neuron_id)];
}

int GetDenseSubnetworkCount(const Simulation* simulation) {
    return simulation == nullptr ? 0 : static_cast<int>(simulation->dense_subnetworks.size());
}

bool GetDenseSubnetworkName(const Simulation* simulation, int subnetwork_index, std::string& name) {
    if (simulation == nullptr || subnetwork_index < 0 ||
        subnetwork_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
        simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)] == nullptr) {
        return false;
    }
    name = simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)]->name();
    return true;
}

int FindDenseSubnetworkByName(const Simulation* simulation, const std::string& name) {
    if (simulation == nullptr) {
        return -1;
    }
    for (std::size_t index = 0; index < simulation->dense_subnetworks.size(); ++index) {
        if (simulation->dense_subnetworks[index] != nullptr &&
            simulation->dense_subnetworks[index]->name() == name) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

bool GetDenseSubnetworkDebugSnapshot(const Simulation* simulation,
                                     int subnetwork_index,
                                     DenseSubnetworkDebugSnapshot& snapshot) {
    if (simulation == nullptr || subnetwork_index < 0 ||
        subnetwork_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
        simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)] == nullptr) {
        return false;
    }
    std::string reason;
    snapshot = simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)]->BuildDebugSnapshot(&reason);
    return true;
}

bool GetDenseSubnetworkWeights(const Simulation* simulation,
                               int subnetwork_index,
                               std::vector<float>& weights) {
    if (simulation == nullptr || subnetwork_index < 0 ||
        subnetwork_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
        simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)] == nullptr) {
        return false;
    }
    weights = simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)]->GetSynapticWeights();
    return true;
}

bool ResetDenseSubnetwork(Simulation* simulation, int subnetwork_index) {
    if (simulation == nullptr || subnetwork_index < 0 ||
        subnetwork_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
        simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)] == nullptr) {
        return false;
    }
    std::string reason;
    return simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)]->Reset(&reason);
}

bool CaptureDenseSubnetworkDebugSnapshot(Simulation* simulation,
                                         const DenseSubnetworkModel* subnetwork,
                                         int time_step) {
    (void)simulation;
    (void)subnetwork;
    (void)time_step;
    return true;
}

bool GetConnectionWeight(const Simulation* simulation,
                         int original_connection_index,
                         float* weight,
                         std::string* reason) {
    if (simulation == nullptr || weight == nullptr) {
        if (reason != nullptr) {
            *reason = "GetConnectionWeight requires a simulation and output pointer";
        }
        return false;
    }
    if (original_connection_index < 0 ||
        original_connection_index >= static_cast<int>(simulation->original_connection_weight_refs.size())) {
        if (reason != nullptr) {
            *reason = "original connection index is out of range";
        }
        return false;
    }
    const RuntimeConnectionWeightRef& ref =
        simulation->original_connection_weight_refs[static_cast<std::size_t>(original_connection_index)];
    if (ref.owner == RuntimeConnectionWeightOwner::InitialOnly) {
        *weight = ref.initial_weight;
        return true;
    }
    if (ref.owner == RuntimeConnectionWeightOwner::MainNetwork) {
        if (simulation->network == nullptr ||
            simulation->network->wordination == nullptr ||
            ref.runtime_weight_index < 0 ||
            ref.runtime_weight_index >= simulation->network->intersNum ||
            simulation->network->wordination[ref.runtime_weight_index] == nullptr) {
            if (reason != nullptr) {
                *reason = "main-network connection weight reference is invalid";
            }
            return false;
        }
        *weight = simulation->network->wordination[ref.runtime_weight_index]->weight;
        return true;
    }
    if (ref.owner == RuntimeConnectionWeightOwner::DenseInternal) {
        if (ref.dense_subnetwork_index < 0 ||
            ref.dense_subnetwork_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
            simulation->dense_subnetworks[static_cast<std::size_t>(ref.dense_subnetwork_index)] == nullptr) {
            if (reason != nullptr) {
                *reason = "dense connection weight reference is invalid";
            }
            return false;
        }
        return simulation->dense_subnetworks[static_cast<std::size_t>(ref.dense_subnetwork_index)]
            ->GetLiveSynapseWeight(ref.runtime_weight_index, weight, reason);
    }
    if (reason != nullptr) {
        *reason = "connection weight owner is not recognized";
    }
    return false;
}

bool SetConnectionWeight(Simulation* simulation,
                         int original_connection_index,
                         float weight,
                         std::string* reason) {
    if (simulation == nullptr) {
        if (reason != nullptr) {
            *reason = "SetConnectionWeight requires a simulation";
        }
        return false;
    }
    if (original_connection_index < 0 ||
        original_connection_index >= static_cast<int>(simulation->original_connection_weight_refs.size())) {
        if (reason != nullptr) {
            *reason = "original connection index is out of range";
        }
        return false;
    }
    const RuntimeConnectionWeightRef& ref =
        simulation->original_connection_weight_refs[static_cast<std::size_t>(original_connection_index)];
    if (ref.owner == RuntimeConnectionWeightOwner::InitialOnly) {
        if (reason != nullptr) {
            *reason = "connection has no mutable runtime weight";
        }
        return false;
    }
    if (ref.owner == RuntimeConnectionWeightOwner::MainNetwork) {
        if (simulation->network == nullptr ||
            simulation->network->wordination == nullptr ||
            ref.runtime_weight_index < 0 ||
            ref.runtime_weight_index >= simulation->network->intersNum ||
            simulation->network->wordination[ref.runtime_weight_index] == nullptr) {
            if (reason != nullptr) {
                *reason = "main-network connection weight reference is invalid";
            }
            return false;
        }
        simulation->network->wordination[ref.runtime_weight_index]->weight = weight;
        return true;
    }
    if (ref.owner == RuntimeConnectionWeightOwner::DenseInternal) {
        if (ref.dense_subnetwork_index < 0 ||
            ref.dense_subnetwork_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
            simulation->dense_subnetworks[static_cast<std::size_t>(ref.dense_subnetwork_index)] == nullptr) {
            if (reason != nullptr) {
                *reason = "dense connection weight reference is invalid";
            }
            return false;
        }
        return simulation->dense_subnetworks[static_cast<std::size_t>(ref.dense_subnetwork_index)]
            ->SetLiveSynapseWeight(ref.runtime_weight_index, weight, reason);
    }
    if (reason != nullptr) {
        *reason = "connection weight owner is not recognized";
    }
    return false;
}

void FlushInputConvMainNetworkCurrent(Simulation* simulation, InputConvModel* model, int time_step) {
    if (simulation == nullptr || model == nullptr) {
        return;
    }
    std::size_t model_index = simulation->InputConvModelList.size();
    for (std::size_t index = 0; index < simulation->InputConvModelList.size(); ++index) {
        if (simulation->InputConvModelList[index] == model) {
            model_index = index;
            break;
        }
    }
    if (model_index >= simulation->InputConvDescriptionList.size() ||
        model_index >= simulation->InputConvMainCurrentSourceNeuronIds.size()) {
        return;
    }
    const InputConvDescription& description =
        simulation->InputConvDescriptionList[model_index];
    if (description.output_target != InputConvOutputTarget::MainNetwork ||
        description.output_source_indices.empty()) {
        return;
    }
    if (simulation->inputCurrentDriver == nullptr ||
        simulation->EventHeap == nullptr ||
        simulation->network == nullptr) {
        throw std::runtime_error("InputConv main-network current route requires initialized current driver");
    }
    const std::vector<int>& source_neuron_ids =
        simulation->InputConvMainCurrentSourceNeuronIds[model_index];
    if (source_neuron_ids.size() != description.output_source_indices.size()) {
        throw std::runtime_error("InputConv main-network current source route count mismatch");
    }
    if (!description.output_scales.empty() &&
        description.output_scales.size() != description.output_source_indices.size()) {
        throw std::runtime_error("InputConv main-network scale count must match route count");
    }
    if (!model->HasDeviceOutputBuffer()) {
        throw std::runtime_error("InputConv main-network current route requires a device output buffer");
    }
    const int output_count = model->GetOutputCount();
    if (output_count <= 0) {
        return;
    }
    std::vector<float> output_values(static_cast<std::size_t>(output_count), 0.0f);
#if NPGR_ENABLE_CUDA
    if (cudaMemcpy(output_values.data(),
                   model->GetDeviceOutputBuffer(),
                   sizeof(float) * static_cast<std::size_t>(output_count),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("InputConv main-network current route failed to download device output");
    }
#else
    throw std::runtime_error("InputConv main-network current route requires CUDA device output support");
#endif
    std::vector<int> times(description.output_source_indices.size(), time_step);
    std::vector<float> currents(description.output_source_indices.size(), 0.0f);
    for (std::size_t route_index = 0;
         route_index < description.output_source_indices.size();
         ++route_index) {
        const int source_index = description.output_source_indices[route_index];
        if (source_index < 0 || source_index >= output_count) {
            throw std::runtime_error("InputConv main-network source index is out of range");
        }
        const float scale = description.output_scales.empty()
                                ? description.output_scale
                                : description.output_scales[route_index];
        currents[route_index] = output_values[static_cast<std::size_t>(source_index)] * scale;
    }
    // Reuse the legacy current driver so InputConv currents pass through the
    // same InputCurrent -> PropogatedCurrent -> ProcessCurrent path as user
    // supplied external currents.
    simulation->inputCurrentDriver->LoadInputCurrent(
        simulation->EventHeap,
        simulation->network,
        static_cast<int>(source_neuron_ids.size()),
        times.data(),
        source_neuron_ids.data(),
        currents.data());
}

bool BindDenseSubnetworkInterfaceCurrentDeviceSource(Simulation* simulation,
                                                     int subnetwork_index,
                                                     const float* device_current,
                                                     int interface_count,
                                                     std::string* reason) {
    if (simulation == nullptr || subnetwork_index < 0 ||
        subnetwork_index >= static_cast<int>(simulation->dense_subnetworks.size()) ||
        simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)] == nullptr) {
        if (reason != nullptr) {
            *reason = "dense subnetwork index is out of range";
        }
        return false;
    }
    return simulation->dense_subnetworks[static_cast<std::size_t>(subnetwork_index)]
        ->BindInterfaceCurrentDeviceSource(device_current, interface_count, reason);
}

void RecreateEventQueue(Simulation* simulation) {
    if (simulation == nullptr) {
        return;
    }
    if (simulation->EventHeap != NULL) {
        delete simulation->EventHeap;
        simulation->EventHeap = NULL;
    }

    if (simulation->eventQueueType == EVENT_QUEUE_TIMING_WHEEL) {
        const int wheelSize =
            simulation->timingWheelSize > 0 ? simulation->timingWheelSize : simulation->totaltimesteps + 1;
        simulation->EventHeap = new TimingWheelEventQueue(simulation->NumberOfQueue, wheelSize);
    } else {
        simulation->EventHeap = new EventQueue(simulation->NumberOfQueue);
    }
}

void BindDriversOnce(Simulation* simulation) {
    if (simulation == nullptr || simulation->drivers_initialized) {
        return;
    }

    simulation->AddInputSpikeDriver(simulation->inputSpikeDriver);
    simulation->AddInputCurrentDriver(simulation->inputCurrentDriver);
    simulation->AddOutputSpikeDriver(simulation->output_spike_driver);
    if (simulation->file_output_spike_driver != 0) {
        simulation->AddOutputSpikeDriver(simulation->file_output_spike_driver);
    }
    if (simulation->file_output_weight_driver != 0) {
        simulation->AddOutputWeightDriver(simulation->file_output_weight_driver);
    }
    if (simulation->file_outer_dynamic_state_driver != 0) {
        simulation->AddOuterDynamicStateDriver(simulation->file_outer_dynamic_state_driver);
    }

    simulation->drivers_initialized = true;
}

void ScheduleInitialEvents(Simulation* simulation) {
    if (simulation == nullptr || simulation->EventHeap == nullptr || simulation->network == nullptr) {
        return;
    }

    for (int i = 0; i < simulation->NumberOfQueue; i++) {
        simulation->currenttime[i] = 0;
        simulation->SimulationEnd[i] = false;
        simulation->PauseThread[i] = false;
        simulation->SyncThread[i] = false;
        if (simulation->totaltimesteps > 0) {
            simulation->EventHeap->Insert_a_Event(new EndSimulationTime(simulation->totaltimesteps, i), i);
        }
        for (int z = 0; z < simulation->network->neurontypesNum; z++) {
            if (simulation->network->TimeDrivenNeuronsNum[z][i] > 0 &&
                !IsDenseManagedModel(simulation, simulation->network->neurontypes[z][i])) {
                simulation->EventHeap->Insert_a_Event(
                    new TimeEventUpdateNeuron(
                        0, i, simulation->network->neurontypes[z][i], simulation->network->TimeDrivenNeurons[z][i]),
                    i);
            }
        }
        for (int z = 0; z < simulation->network->neurontypesNum; z++) {
            if (simulation->network->TimeDrivenNeuronsNumGPU[z][i] > 0 &&
                !IsDenseManagedModel(simulation, simulation->network->neurontypes[z][i])) {
                simulation->EventHeap->Insert_a_Event(
                    new TimeEventUpdateNeuron(
                        0, i, simulation->network->neurontypes[z][i], simulation->network->TimeDrivenNeuronsGPU[z][i]),
                    i);
            }
        }
        for (std::size_t index = 0; index < simulation->OuterDynamicModelList.size(); ++index) {
            if (simulation->OuterDynamicModelList[index]->getQueueIndex() == i) {
                simulation->EventHeap->Insert_a_Event(
                    new OuterUpdateEvent(0, i, simulation->OuterDynamicModelList[index]), i);
            }
        }
        for (std::size_t index = 0; index < simulation->InputConvModelList.size(); ++index) {
            if (simulation->InputConvModelList[index] != nullptr &&
                simulation->InputConvModelList[index]->getQueueIndex() == i) {
                simulation->EventHeap->Insert_a_Event(
                    new UpdateInputConvEvent(0, i, simulation->InputConvModelList[index]), i);
            }
        }
        for (std::size_t dense_index = 0; dense_index < simulation->dense_subnetworks.size(); ++dense_index) {
            if (simulation->dense_subnetworks[dense_index] != nullptr &&
                simulation->dense_subnetworks[dense_index]->queue_index() == i) {
                simulation->EventHeap->Insert_a_Event(
                    new DenseSubnetworkUpdateEvent(0, i, simulation->dense_subnetworks[dense_index]),
                    i);
            }
        }
    }
    if (simulation->DelayMin >= 0) {
        simulation->EventHeap->Insert_a_syn_Event(new SynchronizeActivityEvent(0, simulation));
    }
    if (simulation->WeightSaveInterval > 0) {
        simulation->EventHeap->Insert_a_syn_Event(new SaveWeightEvent(simulation->WeightSaveInterval, simulation));
    }
    if (simulation->CommunivationInterval > 0) {
        CommunicationEvent* communication_event =
            new CommunicationEvent(simulation->CommunivationInterval, simulation->CommunivationInterval, simulation);
        // CommunicationEvent creates matching SynchronizeSimulationEvent items
        // for every queue, so the communication payload itself must stay on the
        // synchronized queue even for a single-queue simulation.
        simulation->EventHeap->Insert_a_syn_Event(communication_event);
    }
}

void ResetForNextRound(Simulation* simulation, bool preserve_weights) {
    if (simulation == nullptr || simulation->network == nullptr) {
        return;
    }
    simulation->network->ResetDynamicState(preserve_weights);
    ResetDenseSubnetworks(simulation);
    simulation->ResetInputConvModels();
    for (std::size_t index = 0; index < simulation->InputConvFrameSourceList.size(); ++index) {
        if (simulation->InputConvFrameSourceList[index] != nullptr) {
            simulation->InputConvFrameSourceList[index]->Clear();
        }
    }
    BindInputConvOutputDescriptions(simulation, simulation->InputConvDescriptionList);
    RecreateEventQueue(simulation);

    if (simulation->output_spike_driver != NULL) {
        simulation->output_spike_driver->ClearBuffer();
    }
    if (simulation->outer_dynamic_spike_buffer != NULL) {
        simulation->outer_dynamic_spike_buffer->Clear();
    }
    {
        std::lock_guard<std::mutex> lock(simulation->outer_dynamic_input_mutex);
        simulation->outer_dynamic_input_times.clear();
        simulation->outer_dynamic_input_neurons.clear();
    }
    {
        std::lock_guard<std::mutex> lock(simulation->outer_dynamic_state_mutex);
        simulation->latest_outer_dynamic_state = OuterDynamicJointState();
        simulation->latest_outer_dynamic_time_step = 0;
        simulation->has_latest_outer_dynamic_state = false;
        simulation->latest_outer_dynamic_states.clear();
    }

    ScheduleInitialEvents(simulation);
}

void ConstructDenseAwareSimulation(Simulation* simulation,
                                   const std::list<NeuronLayerDescription>& neuron_layer_list,
                                   const std::list<ConnectionDescription>& connection_list,
                                   const std::list<LearningRuleDescription>& learning_rule_list,
                                   const std::list<OuterDynamicDescription>& outer_dynamic_list,
                                   const std::list<OuterDynamicConnectionDescription>& outer_dynamic_connection_list,
                                   const std::list<InputConvDescription>& input_conv_list,
                                   int requested_queue_count) {
    if (simulation == nullptr) {
        return;
    }
    setNumberOfOpenMPQueues(requested_queue_count);
    simulation->NumberOfQueue = NumberOfOpenMPQueues;
    simulation->currenttime = new int[simulation->NumberOfQueue]();
    
    simulation->SimulationEnd = new bool[simulation->NumberOfQueue]();
    
    simulation->PauseThread = new bool[simulation->NumberOfQueue]();
    
    simulation->SyncThread = new bool[simulation->NumberOfQueue]();
    
    RecreateEventQueue(simulation);
    simulation->outer_dynamic_spike_buffer = new OuterDynamicSpikeBuffer();
    simulation->CreateOuterDynamic(outer_dynamic_list);
    std::vector<std::vector<int> > input_conv_main_current_source_ids;
    {
        // Split the original build description before constructing any Network.
        // Dense-tagged layers are removed from the main Network and represented
        // by DenseSubnetworkBuildSpec records instead.
        PreparedSimulationBuild prepared_build =
            sim_support::PrepareBlackBoxDenseBuild(
                neuron_layer_list,
                connection_list,
                learning_rule_list,
                simulation->basetimesteps);
        if (!prepared_build.build_error.empty()) {
            throw std::runtime_error(prepared_build.build_error);
        }
        simulation->original_connection_weight_refs =
            prepared_build.original_connection_weight_refs;
        std::list<NeuronLayerDescription> main_layers_with_input_conv =
            prepared_build.dense_specs.empty() ? neuron_layer_list : prepared_build.main_layers;
        std::list<ConnectionDescription> main_connections_with_input_conv =
            prepared_build.dense_specs.empty() ? connection_list : prepared_build.main_connections;
        AddInputConvMainCurrentRoutes(input_conv_list,
                                      prepared_build.original_to_main_neuron_id,
                                      &main_layers_with_input_conv,
                                      &main_connections_with_input_conv,
                                      &input_conv_main_current_source_ids);
        AppendOuterDynamicInterfaceNetwork(simulation,
                                           outer_dynamic_connection_list,
                                           prepared_build.original_to_main_neuron_id,
                                           &main_layers_with_input_conv,
                                           &main_connections_with_input_conv);
        // if the network does not use the dense subnetwork, just build the main network directly
        if (prepared_build.dense_specs.empty()) {
            simulation->original_to_main_neuron_id = prepared_build.original_to_main_neuron_id;
            simulation->network = new Network(
                main_layers_with_input_conv,
                main_connections_with_input_conv,
                learning_rule_list,
                simulation->NumberOfQueue,
                simulation->basetimesteps,
                simulation);
        } else {
            simulation->original_to_main_neuron_id = prepared_build.original_to_main_neuron_id;
            // The main legacy Network contains only non-dense layers plus
            // synthetic interface neurons for main->dense boundary delivery.
            simulation->network = new Network(
                main_layers_with_input_conv,
                main_connections_with_input_conv,
                learning_rule_list,
                simulation->NumberOfQueue,
                simulation->basetimesteps,
                simulation);
        }
        // Build dense runtimes first. Output binding is delayed until every
        // dense model exposes its interface neuron model, which is required for
        // dense-to-dense boundary routes.
        std::vector<sim_support::DenseSubnetworkBuildSpec> finalized_specs(
            prepared_build.dense_specs.size());
        for (std::size_t dense_index = 0; dense_index < prepared_build.dense_specs.size(); ++dense_index) {
            sim_support::DenseSubnetworkBuildSpec spec = prepared_build.dense_specs[dense_index];
            LegacyNetworkBinding binding;
            std::string reason;
            // merge all the same type model layer in to a single merged layer
            if (!sim_support::CompactDenseSubnetworkByModel(&spec, nullptr, &reason)) {
                throw std::runtime_error("dense subnetwork compaction failed: " + reason);
            }
            // build the pre_slice and the buffer table
            if (!sim_support::FinalizeDenseSubnetworkBuildSpec(&spec, &reason)) {
                throw std::runtime_error("dense subnetwork finalization failed: " + reason);
            }
            // scan all the input connections from the main network
            for (std::size_t input_index = 0;
                 input_index < spec.input_connection_source_main_ids.size();
                 ++input_index) {
                // find the interface slot index
                const int interface_slot_index = spec.input_connection_slot_indices[input_index];
                if (interface_slot_index < 0 ||
                    interface_slot_index >= static_cast<int>(spec.input_interface_main_ids.size()) ||
                    spec.input_interface_main_ids[static_cast<std::size_t>(interface_slot_index)] < 0) {
                    continue;
                }
                // find the interface connection
                binding.input_connections.push_back(
                    sim_support::FindMainConnection(
                        simulation->network,
                        spec.input_connection_source_main_ids[input_index],
                        spec.input_interface_main_ids[static_cast<std::size_t>(interface_slot_index)]));
                binding.input_slot_indices.push_back(interface_slot_index);
            }
            // check the dense queue index
            const int dense_queue_index =
                (spec.queue_index >= 0 && spec.queue_index < simulation->NumberOfQueue)
                    ? spec.queue_index
                    : 0;
            DenseSubnetworkModel* dense_model = new DenseSubnetworkModel();
            if (!dense_model->InitializeBlackBox(
                    simulation,
                    spec,
                    binding,
                    spec.config,
                    dense_queue_index,
                    spec.update_timestep,
                    &reason)) {
                delete dense_model;
                throw std::runtime_error("dense subnetwork initialization failed: " + reason);
            }
            finalized_specs[dense_index] = spec;
            // Dense subnetworks are now fail-fast: every spec must initialize
            // successfully, so runtime storage preserves the dense_specs order
            // and can be indexed directly by target_spec_id.
            simulation->dense_subnetworks.push_back(dense_model);
        }
        // scan the subnetwork lists
        for (std::size_t dense_index = 0; dense_index < simulation->dense_subnetworks.size(); ++dense_index) {
            
            DenseSubnetworkModel* dense_model = simulation->dense_subnetworks[dense_index];
            const sim_support::DenseSubnetworkBuildSpec& spec = finalized_specs[dense_index];
            LegacyNetworkBinding binding;
            std::string reason;
            std::vector<std::vector<int> > output_routes_tmp(
                static_cast<std::size_t>(spec.layout.stats.neuron_count));
            // scan the output marker to the network
            for (std::size_t output_index = 0;
                 output_index < spec.output_source_local_ids.size();
                 ++output_index) {
                // find the output target in the main network
                const int output_target_main_id = spec.output_target_main_ids[output_index];
                // create a connection to the main network
                Interconnections* synthetic_connection =
                    CreateDenseToMainConnection(simulation->network,
                                                output_target_main_id,
                                                spec.output_weights[output_index],
                                                spec.output_types[output_index],
                                                spec.output_delays[output_index]);
                if (synthetic_connection == nullptr) {
                    continue;
                }
                // get the connection index
                const int binding_index = static_cast<int>(binding.output_connections.size());
                binding.output_connections.push_back(synthetic_connection);
                binding.output_queue_indices.push_back(
                    simulation->network->neurons[output_target_main_id].Queue_index);
                // get the source local neuron id
                const int output_source_local_id = spec.output_source_local_ids[output_index];
                // build the outputs route map according to the source local neuron id
                if (output_source_local_id >= 0 &&
                    output_source_local_id < static_cast<int>(output_routes_tmp.size())) {
                    output_routes_tmp[static_cast<std::size_t>(output_source_local_id)].push_back(binding_index);
                }
            }
            // scan all the dense output marker to the dense subnetwork
            for (std::size_t output_index = 0;
                 output_index < spec.output_dense_source_local_ids.size();
                 ++output_index) {
                // find the target subnetwork id
                const int target_spec_id = spec.output_dense_target_spec_ids[output_index];
                if (target_spec_id < 0 ||
                    target_spec_id >= static_cast<int>(simulation->dense_subnetworks.size())) {
                    throw std::runtime_error("dense-to-dense output route references an invalid target spec");
                }
                // create a connection to the interface slot
                Interconnections* synthetic_connection =
                    CreateDenseToDenseInterfaceConnection(
                        simulation->dense_subnetworks[static_cast<std::size_t>(target_spec_id)],
                        spec.output_dense_target_slot_indices[output_index],
                        spec.output_dense_weights[output_index],
                        spec.output_dense_types[output_index],
                        spec.output_dense_delays[output_index]);
                if (synthetic_connection == nullptr) {
                    continue;
                }
                // add the new connection
                const int binding_index = static_cast<int>(binding.output_connections.size());
                binding.output_connections.push_back(synthetic_connection);
                binding.output_queue_indices.push_back(
                    simulation->dense_subnetworks[static_cast<std::size_t>(target_spec_id)]->queue_index());
                const int output_source_local_id = spec.output_dense_source_local_ids[output_index];
                if (output_source_local_id >= 0 &&
                    output_source_local_id < static_cast<int>(output_routes_tmp.size())) {
                    output_routes_tmp[static_cast<std::size_t>(output_source_local_id)].push_back(binding_index);
                }
            }
            //init the output route table
            binding.output_route_start_by_source_neuron.assign(
                static_cast<std::size_t>(spec.layout.stats.neuron_count + 1), 0);
            binding.output_route_binding_indices.clear();
            // scan all the neurons to build the output neuron
            for (int source_neuron = 0; source_neuron < spec.layout.stats.neuron_count; ++source_neuron) {
                // get the total output routes numbers of the specific neuron(CSR form)
                binding.output_route_start_by_source_neuron[static_cast<std::size_t>(source_neuron)] =
                    static_cast<int>(binding.output_route_binding_indices.size());
                // get the local route list(out put connections of the neuron)
                const std::vector<int>& local_routes =
                    output_routes_tmp[static_cast<std::size_t>(source_neuron)];
                // insert the local list to the total index list
                binding.output_route_binding_indices.insert(binding.output_route_binding_indices.end(),
                                                            local_routes.begin(),
                                                            local_routes.end());
            }
            // define the last element's number
            binding.output_route_start_by_source_neuron[static_cast<std::size_t>(spec.layout.stats.neuron_count)] =
                static_cast<int>(binding.output_route_binding_indices.size());
            dense_model->BindOutputConnections(binding, &reason);
        }
    }

    simulation->neuronMonitorExist = simulation->network->isMonitor;
    simulation->CreateInputConv(input_conv_list);
    if (!input_conv_main_current_source_ids.empty()) {
        simulation->InputConvMainCurrentSourceNeuronIds = input_conv_main_current_source_ids;
    }

    simulation->inputSpikeDriver = new ArrayInputSpikeDriver();
    simulation->inputCurrentDriver = new ArrayInputCurrentDriver();
    simulation->output_spike_driver = new ArrayOutputSpikeDriver();
    simulation->RealTimeRestrictionObject = new RealTimeRestriction();
    simulation->DelayMin = simulation->network->GetMinInterpropagationTime();
}

}  // namespace sim_support
}  // namespace npgr
