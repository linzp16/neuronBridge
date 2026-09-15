#ifndef NPGR_DENSE_SUBNETWORK_UPDATE_EVENT_H
#define NPGR_DENSE_SUBNETWORK_UPDATE_EVENT_H

#include "source_file_realtime_v1_async/Event/inc/Event.h"

namespace npgr {

class DenseSubnetworkModel;

class DenseSubnetworkUpdateEvent : public Event {
public:
    DenseSubnetworkUpdateEvent(int time, int queue_index, DenseSubnetworkModel* model);
    ~DenseSubnetworkUpdateEvent();

    virtual void ProcessEvent(Simulation* simulation);
    virtual void ProcessEvent(Simulation* simulation, RealTimeRestrictionLevel level);
    virtual enum EventPriority getPriority();

private:
    DenseSubnetworkModel* model_;
};

}  // namespace npgr

#endif
