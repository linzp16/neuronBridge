#ifndef NPGR_CUSTOM_EQUATION_DESCRIPTOR_H
#define NPGR_CUSTOM_EQUATION_DESCRIPTOR_H

namespace npgr {

// A generated descriptor contains only static layout and event-routing data.
// Equation text and parsed ASTs deliberately remain build-time concerns.
enum class CustomInputDelivery : unsigned char {
    AddToState,
    AddToCurrentAccumulator,
};

struct CustomInputBinding {
    int connection_type;
    int target_state_slot;
    float scale;
    CustomInputDelivery delivery;
};

struct CustomEquationDescriptor {
    const char* implementation_name;
    int state_count;
    int differential_state_count;
    int voltage_state_slot;
    const CustomInputBinding* input_bindings;
    int input_binding_count;
};

}  // namespace npgr

#endif
