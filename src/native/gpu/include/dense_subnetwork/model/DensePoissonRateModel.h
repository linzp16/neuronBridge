#ifndef NPGR_DENSEPOISSONRATEMODEL_H
#define NPGR_DENSEPOISSONRATEMODEL_H

#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this model and define the compact field-id order
// consumed by the CUDA update entry.
enum PoissonFieldSlot {
    kPoissonRateHz,
    kPoissonFired,
    kPoissonRateBiasHz,
    kPoissonRateGainHzPerCurrent,
    kPoissonSlotCount,
};

class PoissonRateModel : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override { return DenseNeuronModelFactory::kPoissonRateModelId; }
    const char* CanonicalName() const override { return "PoissonRate"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "PoissonRate";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"rate_hz", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
            {"rate_bias_hz", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"rate_gain_hz_per_current", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
        };
    }
    int FieldSlotCount() const override { return kPoissonSlotCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kPoissonRateHz, "rate_hz"},
            {kPoissonFired, "fired"},
            {kPoissonRateBiasHz, "rate_bias_hz"},
            {kPoissonRateGainHzPerCurrent, "rate_gain_hz_per_current"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        return {
            {PendingChannel::ExcitatoryConductance, "rate_exc", false},
            {PendingChannel::InhibitoryConductance, "rate_inh", false},
            {PendingChannel::Current, "rate_current", false},
        };
    }
    std::vector<DenseSpikeEffectBinding> SpikeEffects() const override {
        return {
            {0, PendingChannel::ExcitatoryConductance, 1.0f},
            {1, PendingChannel::InhibitoryConductance, 1.0f},
            {3, PendingChannel::Current, 1.0f},
        };
    }
    bool FillInitialFieldValues(DenseNeuronHostFieldTable* table,
                                const DenseNeuronModelSpec& spec,
                                std::string*) const override {
        FillFloatField(table, spec, "rate_bias_hz", {"poisson_rate_bias_hz", "rate_bias_hz"}, 0.0f);
        FillFloatField(table, spec, "rate_gain_hz_per_current",
                       {"poisson_rate_gain_hz_per_current", "rate_gain_hz_per_current"}, 1.0f);
        return ResetStateFields(table, spec, nullptr);
    }
    bool ResetStateFields(DenseNeuronHostFieldTable* table,
                          const DenseNeuronModelSpec& spec,
                          std::string*) const override {
        FillFloatConstant(table, spec, "rate_hz", 0.0f);
        FillByteConstant(table, spec, "fired", 0);
        return true;
    }

protected:
    std::vector<const char*> ParameterKeys() const override {
        return {"poisson_rate_bias_hz", "poisson_rate_gain_hz_per_current"};
    }
};

}  // namespace
}  // namespace npgr

#endif
