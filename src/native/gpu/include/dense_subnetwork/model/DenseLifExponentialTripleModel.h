#ifndef NPGR_DENSELIFEXPONENTIALTRIPLEMODEL_H
#define NPGR_DENSELIFEXPONENTIALTRIPLEMODEL_H

#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this model and define the compact field-id order
// consumed by the CUDA update entry.
enum LifTripleFieldSlot {
    kLifTripleV,
    kLifTripleGampa,
    kLifTripleGgaba,
    kLifTripleGnmda,
    kLifTripleFired,
    kLifTripleStepsSinceLastSpike,
    kLifTripleVRest,
    kLifTripleVReset,
    kLifTripleVThreshold,
    kLifTripleR,
    kLifTripleEAmpa,
    kLifTripleEGaba,
    kLifTripleAmpaDecay,
    kLifTripleGabaDecay,
    kLifTripleNmdaDecay,
    kLifTripleMembraneAlpha,
    kLifTripleRefractorySteps,
    kLifTripleSlotCount,
};

class LifExponentialTripleModel : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override { return DenseNeuronModelFactory::kLifExponentialTripleModelId; }
    const char* CanonicalName() const override { return "TimeDrivenLIF_Exponential_triple"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "TimeDrivenLIF_Exponential_triple" ||
               name == "TimeDrivenLIF_Exponential_triple_GPU";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"v", DenseFieldStorage::Float32, DenseFieldRole::State, -60.0f},
            {"gampa", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"ggaba", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"gnmda", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
            {"steps_since_last_spike", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 10000},
            {"tau_m_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 20.0f},
            {"v_rest", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -60.0f},
            {"v_reset", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -60.0f},
            {"v_threshold", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -50.0f},
            {"r", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"E_ampa", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"E_gaba", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -80.0f},
            {"ampa_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 10.0f},
            {"gaba_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 10.0f},
            {"nmda_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 10.0f},
            {"t_ref", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 50.0f},
            {"ampa_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"gaba_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"nmda_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"membrane_alpha", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"refractory_steps", DenseFieldStorage::Int32, DenseFieldRole::Derived, 0.0f, 0},
        };
    }
    int FieldSlotCount() const override { return kLifTripleSlotCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kLifTripleV, "v"},
            {kLifTripleGampa, "gampa"},
            {kLifTripleGgaba, "ggaba"},
            {kLifTripleGnmda, "gnmda"},
            {kLifTripleFired, "fired"},
            {kLifTripleStepsSinceLastSpike, "steps_since_last_spike"},
            {kLifTripleVRest, "v_rest"},
            {kLifTripleVReset, "v_reset"},
            {kLifTripleVThreshold, "v_threshold"},
            {kLifTripleR, "r"},
            {kLifTripleEAmpa, "E_ampa"},
            {kLifTripleEGaba, "E_gaba"},
            {kLifTripleAmpaDecay, "ampa_decay"},
            {kLifTripleGabaDecay, "gaba_decay"},
            {kLifTripleNmdaDecay, "nmda_decay"},
            {kLifTripleMembraneAlpha, "membrane_alpha"},
            {kLifTripleRefractorySteps, "refractory_steps"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        return {
            {PendingChannel::ExcitatoryConductance, "ampa", true},
            {PendingChannel::InhibitoryConductance, "gaba", true},
            {PendingChannel::NmdaConductance, "nmda", true},
            {PendingChannel::Current, "current", false},
        };
    }
    std::vector<DenseSpikeEffectBinding> SpikeEffects() const override {
        return {
            {0, PendingChannel::ExcitatoryConductance, 1.0f},
            {1, PendingChannel::InhibitoryConductance, 1.0f},
            {2, PendingChannel::NmdaConductance, 1.0f},
            {3, PendingChannel::Current, 1.0f},
        };
    }
    bool FillInitialFieldValues(DenseNeuronHostFieldTable* table,
                                const DenseNeuronModelSpec& spec,
                                std::string*) const override {
        FillFloatField(table, spec, "tau_m_ms", {"tau", "tau_m_ms"}, 20.0f);
        FillFloatField(table, spec, "v_rest", {"V_rest", "v_rest"}, -60.0f);
        FillFloatField(table, spec, "v_reset", {"V_reset", "v_reset"}, -60.0f);
        FillFloatField(table, spec, "v_threshold", {"V_th", "v_threshold"}, -50.0f);
        FillFloatField(table, spec, "r", {"R", "r"}, 1.0f);
        FillFloatField(table, spec, "E_ampa", {"E_ampa", "Eexc", "e_exc"}, 0.0f);
        FillFloatField(table, spec, "E_gaba", {"E_gaba", "Einh", "e_inh"}, -80.0f);
        FillFloatField(table, spec, "ampa_tau", {"ampa_tau", "gexc_tau", "tau_exc_ms"}, 10.0f);
        FillFloatField(table, spec, "gaba_tau", {"gaba_tau", "ginh_tau", "tau_inh_ms"}, 10.0f);
        FillFloatField(table, spec, "nmda_tau", {"nmda_tau"}, 10.0f);
        FillFloatField(table, spec, "t_ref", {"t_ref"}, 50.0f);
        return ResetStateFields(table, spec, nullptr);
    }
    bool BuildDerivedFields(DenseNeuronHostFieldTable* table,
                            const DenseNeuronModelSpec& spec,
                            float dt_ms,
                            std::string* reason) const override {
        const float* tau_m = FloatField(*table, "tau_m_ms");
        const float* ampa_tau = FloatField(*table, "ampa_tau");
        const float* gaba_tau = FloatField(*table, "gaba_tau");
        const float* nmda_tau = FloatField(*table, "nmda_tau");
        const float* t_ref = FloatField(*table, "t_ref");
        float* ampa_decay = FloatField(table, "ampa_decay");
        float* gaba_decay = FloatField(table, "gaba_decay");
        float* nmda_decay = FloatField(table, "nmda_decay");
        float* membrane_alpha = FloatField(table, "membrane_alpha");
        int* refractory_steps = IntField(table, "refractory_steps");
        const DenseNeuronRange& range = spec.range;
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            if (tau_m[neuron_id] <= 0.0f ||
                ampa_tau[neuron_id] <= 0.0f ||
                gaba_tau[neuron_id] <= 0.0f ||
                nmda_tau[neuron_id] <= 0.0f) {
                if (reason != nullptr) {
                    *reason = "LIF triple time constants must be positive";
                }
                return false;
            }
            ampa_decay[neuron_id] = std::exp(-dt_ms / ampa_tau[neuron_id]);
            gaba_decay[neuron_id] = std::exp(-dt_ms / gaba_tau[neuron_id]);
            nmda_decay[neuron_id] = std::exp(-dt_ms / nmda_tau[neuron_id]);
            membrane_alpha[neuron_id] = dt_ms / tau_m[neuron_id];
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
        FillFloatConstant(table, spec, "gampa", 0.0f);
        FillFloatConstant(table, spec, "ggaba", 0.0f);
        FillFloatConstant(table, spec, "gnmda", 0.0f);
        FillByteConstant(table, spec, "fired", 0);
        FillIntConstant(table, spec, "steps_since_last_spike", 10000);
        return true;
    }

protected:
    std::vector<const char*> ParameterKeys() const override {
        return {"tau", "tau_m_ms", "V_rest", "v_rest", "V_reset", "v_reset",
                "V_th", "v_threshold", "R", "r", "E_ampa", "E_gaba",
                "ampa_tau", "gaba_tau", "nmda_tau", "t_ref"};
    }
};

}  // namespace
}  // namespace npgr

#endif
