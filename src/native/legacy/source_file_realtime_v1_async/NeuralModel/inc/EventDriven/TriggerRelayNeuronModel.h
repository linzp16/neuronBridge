#ifndef TRIGGER_RELAY_NEURON_MODEL_H
#define TRIGGER_RELAY_NEURON_MODEL_H

#include "../source_file_realtime_v1_async/Event/inc/Spike/TriggerRelayInternalSpike.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/EventDrivenInputDevice.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "../source_file_realtime_v1_async/Neuron/inc/Neuron.h"

// Event-driven relay neuron shared by legacy CPU/GPU and dense declarations.
// Any input spike immediately schedules an internal spike from the target relay
// neuron, allowing relay neurons to mediate trigger-like signals through normal
// spike propagation.
class TriggerRelayNeuronModel : public EventDrivenInputDevice {
public:
    TriggerRelayNeuronModel();
    TriggerRelayNeuronModel(int timestep);
    ~TriggerRelayNeuronModel();

    virtual Neuron_State_Vector* InitState() { return NULL; }
    virtual void InitStateVector(int NumberOfNeurons, int GPUIndex) {
        (void)NumberOfNeurons;
        (void)GPUIndex;
    }

    virtual InternalSpike* ProcessSpike(Interconnections* inter, int time) {
        if (inter == NULL || inter->TargetNeuron == NULL) {
            return NULL;
        }
        return new TriggerRelayInternalSpike(
            inter->TargetNeuron,
            time,
            inter->TargetNeuron->Queue_index);
    }

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

    virtual bool compare(NeuronModel* neuronModel) {
        if (!EventDrivenInputDevice::compare(neuronModel)) {
            return false;
        }
        return dynamic_cast<TriggerRelayNeuronModel*>(neuronModel) != NULL;
    }

    virtual std::map<std::string, boost::any> getParameters() {
        return std::map<std::string, boost::any>();
    }
};

#endif
