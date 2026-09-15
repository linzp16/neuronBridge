#ifndef NPGR_DENSETRIGGERRELAYMODEL_H
#define NPGR_DENSETRIGGERRELAYMODEL_H

#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this model and define the compact field-id order
// consumed by the CUDA update entry.
enum TriggerRelayFieldSlot {
    kTriggerRelayThreshold,
    kTriggerRelayLastExcInput,
    kTriggerRelayFired,
    kTriggerRelaySlotCount,
};

class TriggerRelayModel : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override { return DenseNeuronModelFactory::kTriggerRelayModelId; }
    const char* CanonicalName() const override { return "TriggerRelayNeuronModel"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "TriggerRelayNeuronModel" ||
               name == "TriggerRelayNeuronModel_GPU" ||
               name == "DenseTriggerRelayNeuron" ||
               name == "DenseTriggerRelayNeuronModel" ||
               name == "TriggerRelayNeuron";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"threshold", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"last_exc_input", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
        };
    }
    int FieldSlotCount() const override { return kTriggerRelaySlotCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kTriggerRelayThreshold, "threshold"},
            {kTriggerRelayLastExcInput, "last_exc_input"},
            {kTriggerRelayFired, "fired"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        // Main-network trigger mediation uses the ordinary excitatory pending
        // channel, so no propagation kernel or Iset bitmap path is specialized.
        return {
            {PendingChannel::ExcitatoryConductance, "exc_trigger", false},
        };
    }
    std::vector<DenseSpikeEffectBinding> SpikeEffects() const override {
        // Every incoming legacy connection type is normalized to ordinary
        // excitatory input for this relay. The relay's only purpose is to turn
        // a boundary arrival into a dense-local spike.
        return {
            {0, PendingChannel::ExcitatoryConductance, 1.0f},
            {1, PendingChannel::ExcitatoryConductance, 1.0f},
            {2, PendingChannel::ExcitatoryConductance, 1.0f},
            {3, PendingChannel::ExcitatoryConductance, 1.0f},
        };
    }
    bool FillInitialFieldValues(DenseNeuronHostFieldTable* table,
                                const DenseNeuronModelSpec& spec,
                                std::string*) const override {
        FillFloatField(table, spec, "threshold", {"threshold", "trigger_threshold"}, 0.0f);
        return ResetStateFields(table, spec, nullptr);
    }
    bool ResetStateFields(DenseNeuronHostFieldTable* table,
                          const DenseNeuronModelSpec& spec,
                          std::string*) const override {
        FillFloatConstant(table, spec, "last_exc_input", 0.0f);
        FillByteConstant(table, spec, "fired", 0);
        return true;
    }

protected:
    std::vector<const char*> ParameterKeys() const override {
        return {"threshold", "trigger_threshold"};
    }
};


}  // namespace
}  // namespace npgr

#endif
