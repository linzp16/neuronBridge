#ifndef NPGR_SIMULATION_COMMON_HOST_H
#define NPGR_SIMULATION_COMMON_HOST_H

#include "source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "source_file_realtime_v1_async/Simulation/inc/InputConvDescription.h"

#include <list>
#include <string>
#include <vector>

class NeuronModel;
class Simulation;
class InputConvModel;

namespace npgr {
class DenseSubnetworkModel;
struct DenseSubnetworkDebugSnapshot;

namespace sim_support {

void AddExternalSpikeActivityTranslated(Simulation* simulation,
                                        const std::vector<int>& event_time,
                                        const std::vector<int>& event_neuron_id);
void AddExternalCurrentActivityTranslated(Simulation* simulation,
                                          const std::vector<int>& event_time,
                                          const std::vector<int>& event_neuron_id,
                                          const std::vector<float>& current);

void ResetDenseSubnetworks(Simulation* simulation);
bool IsDenseManagedModel(const Simulation* simulation, const NeuronModel* model);
int TranslateOriginalNeuronIdToMain(const Simulation* simulation, int original_neuron_id);
int GetDenseSubnetworkCount(const Simulation* simulation);
bool GetDenseSubnetworkName(const Simulation* simulation, int subnetwork_index, std::string& name);
int FindDenseSubnetworkByName(const Simulation* simulation, const std::string& name);
bool GetDenseSubnetworkDebugSnapshot(const Simulation* simulation,
                                     int subnetwork_index,
                                     DenseSubnetworkDebugSnapshot& snapshot);
bool GetDenseSubnetworkWeights(const Simulation* simulation,
                               int subnetwork_index,
                               std::vector<float>& weights);
bool ResetDenseSubnetwork(Simulation* simulation, int subnetwork_index);
bool CaptureDenseSubnetworkDebugSnapshot(Simulation* simulation,
                                         const DenseSubnetworkModel* subnetwork,
                                         int time_step);
bool GetConnectionWeight(const Simulation* simulation,
                         int original_connection_index,
                         float* weight,
                         std::string* reason);
bool SetConnectionWeight(Simulation* simulation,
                         int original_connection_index,
                         float weight,
                         std::string* reason);
void FlushInputConvMainNetworkCurrent(Simulation* simulation, InputConvModel* model, int time_step);
bool BindDenseSubnetworkInterfaceCurrentDeviceSource(Simulation* simulation,
                                                     int subnetwork_index,
                                                     const float* device_current,
                                                     int interface_count,
                                                     std::string* reason);

void RecreateEventQueue(Simulation* simulation);
void BindDriversOnce(Simulation* simulation);
void ScheduleInitialEvents(Simulation* simulation);
void ResetForNextRound(Simulation* simulation, bool preserve_weights);
void ConstructDenseAwareSimulation(Simulation* simulation,
                                   const std::list<NeuronLayerDescription>& neuron_layer_list,
                                   const std::list<ConnectionDescription>& connection_list,
                                   const std::list<LearningRuleDescription>& learning_rule_list,
                                   const std::list<OuterDynamicDescription>& outer_dynamic_list,
                                   const std::list<OuterDynamicConnectionDescription>& outer_dynamic_connection_list,
                                   const std::list<InputConvDescription>& input_conv_list,
                                   int requested_queue_count);

}  // namespace sim_support
}  // namespace npgr

#endif
