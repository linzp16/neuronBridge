#ifndef OUTER_DYNAMIC_INTERFACE_NEURON_MODEL_H
#define OUTER_DYNAMIC_INTERFACE_NEURON_MODEL_H

#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/EventDrivenInputDevice.h"

#include <vector>

class Simulation;

/* Hidden endpoint model used to receive ordinary network spikes for
 * OuterDynamic joints. It is created by Simulation, not declared by users as a
 * normal layer. Each endpoint neuron maps to one (outer_dynamic, joint) pair. */
class OuterDynamicInterfaceNeuronModel : public EventDrivenInputDevice {
public:
    OuterDynamicInterfaceNeuronModel();
    explicit OuterDynamicInterfaceNeuronModel(int timestep);
    virtual ~OuterDynamicInterfaceNeuronModel();

    void BindSimulation(Simulation* simulation);
    void ConfigureEndpointBindings(const std::vector<int>& outer_dynamic_ids,
                                   const std::vector<int>& joint_ids);
    void ConfigureFromParameters(const std::map<std::string, boost::any>& parameters);

    virtual Neuron_State_Vector* InitState() { return NULL; }
    virtual void InitStateVector(int NumberOfNeurons, int GPUIndex) {
        (void)NumberOfNeurons;
        (void)GPUIndex;
    }
    virtual InternalSpike* ProcessSpike(Interconnections* inter, int time);
    virtual void ProcessCurrent(Interconnections* inter, Neuron* Target, float current) {
        (void)inter;
        (void)Target;
        (void)current;
    }
    virtual void UpdateState(int index, int time, Simulation* simulation) {
        (void)index;
        (void)time;
        (void)simulation;
    }
    virtual void InitializeInputCurrentSynapseStructure() {}
    virtual void CheckType(Interconnections* inter) { (void)inter; }
    virtual int getV_index() { return -1; }
    virtual int get_NumberOfState() { return -1; }
    virtual enum NeuronModelType getNeuronModelType() { return INPUT_DEVICE; }
    virtual bool compare(NeuronModel* neuronModel);
    virtual std::map<std::string, boost::any> getParameters();

private:
    Simulation* simulation_;
    std::vector<int> outer_dynamic_id_by_neuron_;
    std::vector<int> joint_id_by_neuron_;
    int endpoint_global_start_;
};

#endif
