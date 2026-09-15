#include "../source_file_realtime_v1_async/InputConv/inc/InputConvV1Stub.h"

InputConvV1Stub::InputConvV1Stub() : InputConvModel() {}

InputConvV1Stub::InputConvV1Stub(int timestep_size) : InputConvModel(timestep_size) {}

InputConvV1Stub::~InputConvV1Stub() {}

void InputConvV1Stub::Initialize(const InputConvDescription& description, Simulation* simulation) {
    (void)simulation;
    this->description_ = description;
    this->setTimestepSize(description.update_timestep > 0 ? description.update_timestep : 1);
    this->setQueueIndex(description.queue_index);
}

void InputConvV1Stub::Update(int time, Simulation* simulation) {
    (void)time;
    (void)simulation;
}

void InputConvV1Stub::Reset(Simulation* simulation) {
    (void)simulation;
}
