#ifndef INPUT_CONV_V1_STUB_H
#define INPUT_CONV_V1_STUB_H

#include "../source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"

class InputConvV1Stub : public InputConvModel {
public:
    InputConvV1Stub();
    explicit InputConvV1Stub(int timestep_size);
    virtual ~InputConvV1Stub();

    virtual void Initialize(const InputConvDescription& description, Simulation* simulation);
    virtual void Update(int time, Simulation* simulation);
    virtual void Reset(Simulation* simulation);

private:
    InputConvDescription description_;
};

#endif
