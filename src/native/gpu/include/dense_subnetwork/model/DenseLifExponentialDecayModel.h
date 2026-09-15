#ifndef NPGR_DENSELIFEXPONENTIALDECAYMODEL_H
#define NPGR_DENSELIFEXPONENTIALDECAYMODEL_H

#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this model and define the compact field-id order
// consumed by the CUDA update entry.
enum LifDecayFieldSlot {
    kLifDecayV,
    kLifDecayG,
    kLifDecayFired,
    kLifDecayStepsSinceLastSpike,
    kLifDecayVRest,
    kLifDecayVReset,
    kLifDecayVThreshold,
    kLifDecayR,
    kLifDecayE,
    kLifDecayGAlpha,
    kLifDecayDtInvTauM,
    kLifDecayRefractorySteps,
    kLifDecaySlotCount,
};

class LifExponentialDecayModel : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override { return DenseNeuronModelFactory::kLifExponentialDecayModelId; }
    const char* CanonicalName() const override { return "TimeDrivenLIF_Exponential_Decay"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "TimeDrivenLIF_Exponential_Decay";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"v", DenseFieldStorage::Float32, DenseFieldRole::State, -65.0f},
            {"decay_g", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
            {"steps_since_last_spike", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 10000},
            {"tau_m_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 20.0f},
            {"g_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 12.0f},
            {"v_rest", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"v_reset", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"v_threshold", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -20.0f},
            {"r", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"e", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"t_ref", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 50.0f},
            {"decay_g_alpha", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"dt_inv_tau_m", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"refractory_steps", DenseFieldStorage::Int32, DenseFieldRole::Derived, 0.0f, 0},
        };
    }
    int FieldSlotCount() const override { return kLifDecaySlotCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kLifDecayV, "v"},
            {kLifDecayG, "decay_g"},
            {kLifDecayFired, "fired"},
            {kLifDecayStepsSinceLastSpike, "steps_since_last_spike"},
            {kLifDecayVRest, "v_rest"},
            {kLifDecayVReset, "v_reset"},
            {kLifDecayVThreshold, "v_threshold"},
            {kLifDecayR, "r"},
            {kLifDecayE, "e"},
            {kLifDecayGAlpha, "decay_g_alpha"},
            {kLifDecayDtInvTauM, "dt_inv_tau_m"},
            {kLifDecayRefractorySteps, "refractory_steps"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        return {
            {PendingChannel::ExcitatoryConductance, "exc_conductance", true},
            {PendingChannel::InhibitoryConductance, "inh_conductance", true},
            {PendingChannel::Current, "current", false},
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
        FillFloatField(table, spec, "tau_m_ms", {"tau", "tau_m_ms"}, 20.0f);
        FillFloatField(table, spec, "g_tau", {"g_tau"}, 12.0f);
        FillFloatField(table, spec, "v_rest", {"V_rest", "v_rest"}, -65.0f);
        FillFloatField(table, spec, "v_reset", {"V_reset", "v_reset"}, -65.0f);
        FillFloatField(table, spec, "v_threshold", {"V_th", "v_threshold"}, -20.0f);
        FillFloatField(table, spec, "r", {"R", "r"}, 1.0f);
        FillFloatField(table, spec, "e", {"E"}, 0.0f);
        FillFloatField(table, spec, "t_ref", {"t_ref"}, 50.0f);
        return ResetStateFields(table, spec, nullptr);
    }
    bool BuildDerivedFields(DenseNeuronHostFieldTable* table,
                            const DenseNeuronModelSpec& spec,
                            float dt_ms,
                            std::string* reason) const override {
        const float* tau_m = FloatField(*table, "tau_m_ms");
        const float* g_tau = FloatField(*table, "g_tau");
        const float* t_ref = FloatField(*table, "t_ref");
        float* decay_g_alpha = FloatField(table, "decay_g_alpha");
        float* dt_inv_tau_m = FloatField(table, "dt_inv_tau_m");
        int* refractory_steps = IntField(table, "refractory_steps");
        const DenseNeuronRange& range = spec.range;
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            if (tau_m[neuron_id] <= 0.0f || g_tau[neuron_id] <= 0.0f) {
                if (reason != nullptr) {
                    *reason = "LIF decay time constants must be positive";
                }
                return false;
            }
            decay_g_alpha[neuron_id] = std::exp(-dt_ms / g_tau[neuron_id]);
            dt_inv_tau_m[neuron_id] = dt_ms / tau_m[neuron_id];
            refractory_steps[neuron_id] = static_cast<int>(t_ref[neuron_id]);
        }
        return true;
    }
    bool ResetStateFields(DenseNeuronHostFieldTable* table,
                          const DenseNeuronModelSpec& spec,
                          std::string*) const override {
        const float* v_rest = FloatField(*table, "v_rest");
        float* v = FloatField(table, "v");
        const DenseNeuronRange& range = spec.range;
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            v[neuron_id] = v_rest[neuron_id];
        }
        FillFloatConstant(table, spec, "decay_g", 0.0f);
        FillByteConstant(table, spec, "fired", 0);
        FillIntConstant(table, spec, "steps_since_last_spike", 10000);
        return true;
    }

protected:
    std::vector<const char*> ParameterKeys() const override {
        return {"tau", "tau_m_ms", "V_rest", "v_rest", "V_th", "v_threshold",
                "R", "r", "V_reset", "v_reset", "g_tau", "E", "t_ref"};
    }
};

}  // namespace
}  // namespace npgr

#endif
