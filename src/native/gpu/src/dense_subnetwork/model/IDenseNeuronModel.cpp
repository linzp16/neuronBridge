#include "dense_subnetwork/model/IDenseNeuronModel.h"

#include "dense_subnetwork/model/DenseNeuronModelFactory.h"

#include <sstream>
#include <unordered_set>

namespace npgr {

bool IDenseNeuronModel::Validate(std::string* reason) const {
    if (FactoryModelId() <= DenseNeuronModelFactory::kUnknownModelId) {
        if (reason != nullptr) {
            *reason = "dense neuron model has an invalid factory model id";
        }
        return false;
    }
    if (CanonicalName() == nullptr || CanonicalName()[0] == '\0') {
        if (reason != nullptr) {
            *reason = "dense neuron model has an empty canonical name";
        }
        return false;
    }
    const int slot_count = FieldSlotCount();
    if (slot_count < 0) {
        if (reason != nullptr) {
            *reason = std::string(CanonicalName()) + " has a negative field slot count";
        }
        return false;
    }
    const std::vector<DenseFieldSchema> fields = Fields();
    std::unordered_set<std::string> field_names;
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (fields[index].name.empty()) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " declares an empty field name";
            }
            return false;
        }
        if (!field_names.insert(fields[index].name).second) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " declares duplicate field: " + fields[index].name;
            }
            return false;
        }
    }

    std::vector<unsigned char> seen_slots(static_cast<std::size_t>(slot_count), 0);
    const std::vector<DenseFieldSlotBinding> slots = FieldSlots();
    for (std::size_t index = 0; index < slots.size(); ++index) {
        const DenseFieldSlotBinding& slot = slots[index];
        if (slot.slot_id < 0 || slot.slot_id >= slot_count) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " declares an invalid field slot";
            }
            return false;
        }
        if (seen_slots[static_cast<std::size_t>(slot.slot_id)] != 0) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " declares a duplicate field slot";
            }
            return false;
        }
        seen_slots[static_cast<std::size_t>(slot.slot_id)] = 1;
        if (field_names.find(slot.field_name) == field_names.end()) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " binds slot to unknown field: " + slot.field_name;
            }
            return false;
        }
    }
    for (int slot = 0; slot < slot_count; ++slot) {
        if (seen_slots[static_cast<std::size_t>(slot)] == 0) {
            if (reason != nullptr) {
                std::ostringstream oss;
                oss << CanonicalName() << " leaves field slot " << slot << " unbound";
                *reason = oss.str();
            }
            return false;
        }
    }

    std::unordered_set<int> input_channels;
    const std::vector<DenseInputChannelBinding> channels = InputChannels();
    for (std::size_t index = 0; index < channels.size(); ++index) {
        const int channel = static_cast<int>(channels[index].channel);
        if (channel < 0 || channel >= DenseNeuronModelFactory::kPendingChannelKeyStride) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) + " declares an invalid pending channel";
            }
            return false;
        }
        input_channels.insert(channel);
    }
    const std::vector<DenseSpikeEffectBinding> effects = SpikeEffects();
    for (std::size_t index = 0; index < effects.size(); ++index) {
        const int channel = static_cast<int>(effects[index].channel);
        if (input_channels.find(channel) == input_channels.end()) {
            if (reason != nullptr) {
                *reason = std::string(CanonicalName()) +
                    " maps a spike effect to a channel the model does not consume";
            }
            return false;
        }
    }
    return true;
}

bool IDenseNeuronModel::FillInitialFieldValues(DenseNeuronHostFieldTable*,
                                               const DenseNeuronModelSpec&,
                                               std::string*) const {
    return true;
}

bool IDenseNeuronModel::BuildDerivedFields(DenseNeuronHostFieldTable*,
                                           const DenseNeuronModelSpec&,
                                           float,
                                           std::string*) const {
    return true;
}

bool IDenseNeuronModel::ResetStateFields(DenseNeuronHostFieldTable*,
                                         const DenseNeuronModelSpec&,
                                         std::string*) const {
    return true;
}

}  // namespace npgr
