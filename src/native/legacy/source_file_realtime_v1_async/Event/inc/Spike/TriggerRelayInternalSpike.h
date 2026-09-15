#ifndef TRIGGER_RELAY_INTERNAL_SPIKE_H
#define TRIGGER_RELAY_INTERNAL_SPIKE_H

#include "../source_file_realtime_v1_async/Event/inc/Spike/InternalSpike.h"

// Internal spike emitted by TriggerRelayNeuronModel immediately after it
// receives an input spike. It reuses the normal legacy propagation path so a
// relay neuron can be marked as output or connected to downstream neurons.
class TriggerRelayInternalSpike : public InternalSpike {
public:
    TriggerRelayInternalSpike();
    TriggerRelayInternalSpike(Neuron* neuron, int time, int queue_index);
    ~TriggerRelayInternalSpike();

    virtual void ProcessEvent(Simulation* simulation);
    virtual void ProcessEvent(Simulation* simulation, RealTimeRestrictionLevel level);
    virtual enum EventPriority getPriority();
};

#endif
