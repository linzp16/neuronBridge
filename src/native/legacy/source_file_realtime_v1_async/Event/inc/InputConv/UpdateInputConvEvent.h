#ifndef UPDATE_INPUT_CONV_EVENT_H
#define UPDATE_INPUT_CONV_EVENT_H

#include "../source_file_realtime_v1_async/Event/inc/Event.h"

class InputConvModel;

class UpdateInputConvEvent : public Event {
public:
    UpdateInputConvEvent(int time, int queue_index, InputConvModel* model);
    ~UpdateInputConvEvent();

    virtual void ProcessEvent(Simulation* simulation);
    virtual void ProcessEvent(Simulation* simulation, RealTimeRestrictionLevel level);
    virtual enum EventPriority getPriority();

private:
    InputConvModel* model_;
};

#endif
