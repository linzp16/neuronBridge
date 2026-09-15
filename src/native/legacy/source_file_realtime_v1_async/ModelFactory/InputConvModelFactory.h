#ifndef INPUT_CONV_MODEL_FACTORY_H
#define INPUT_CONV_MODEL_FACTORY_H

#include "../source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"

class InputConvModelFactory {
public:
    static InputConvModel* createInputConvModel(const InputConvDescription& description);
};

#endif
