#ifndef NPGR_DENSE_NEURON_FIELD_SCHEMA_H
#define NPGR_DENSE_NEURON_FIELD_SCHEMA_H

#include <cstdint>
#include <string>

namespace npgr {

// Storage type for a model-declared flat neuron field.
enum class DenseFieldStorage : std::uint8_t {
    Float32 = 0,
    Int32 = 1,
    UInt8 = 2,
};

// Logical role used by reset/debug/export code.
enum class DenseFieldRole : std::uint8_t {
    Parameter = 0,
    State = 1,
    Derived = 2,
    Debug = 3,
};

// Model-owned field declaration. Runtime allocates fields automatically from
// these schemas so adding a model does not require editing runtime buffers.
struct DenseFieldSchema {
    std::string name;
    DenseFieldStorage storage = DenseFieldStorage::Float32;
    DenseFieldRole role = DenseFieldRole::Parameter;
    float default_float = 0.0f;
    int default_int = 0;
    unsigned char default_byte = 0;
};

}  // namespace npgr

#endif
