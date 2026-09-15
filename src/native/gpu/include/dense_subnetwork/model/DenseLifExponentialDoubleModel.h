#ifndef NPGR_DENSELIFEXPONENTIALDOUBLEMODEL_H
#define NPGR_DENSELIFEXPONENTIALDOUBLEMODEL_H

#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this model and define the compact field-id order
// consumed by the CUDA update entry.
enum LifDoubleFieldSlot {
    kLifDoubleV,
    kLifDoubleGexc,
    kLifDoubleGinh,
    kLifDoubleFired,
    kLifDoubleStepsSinceLastSpike,
    kLifDoubleVRest,
    kLifDoubleVReset,
    kLifDoubleVThreshold,
    kLifDoubleEExc,
    kLifDoubleEInh,
    kLifDoubleExcDecay,
    kLifDoubleInhDecay,
    kLifDoubleMembraneAlpha,
    kLifDoubleRefractorySteps,
    kLifDoubleSlotCount,
};

class LifExponentialDoubleModel : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override { return DenseNeuronModelFactory::kLifExponentialDoubleModelId; }
    const char* CanonicalName() const override { return "TimeDrivenLIF_Exponential_double"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "TimeDrivenLIF_Exponential_double";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"v", DenseFieldStorage::Float32, DenseFieldRole::State, -65.0f},
            {"gexc", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"ginh", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
            {"steps_since_last_spike", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 10000},
            {"tau_m_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 20.0f},
            {"tau_exc_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 5.0f},
            {"tau_inh_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 10.0f},
            {"v_rest", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"v_reset", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"v_threshold", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -50.0f},
            {"e_exc", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"e_inh", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -80.0f},
            {"t_ref", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 50.0f},
            {"exc_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"inh_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"membrane_alpha", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"refractory_steps", DenseFieldStorage::Int32, DenseFieldRole::Derived, 0.0f, 0},
        };
    }
    int FieldSlotCount() const override { return kLifDoubleSlotCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kLifDoubleV, "v"},
            {kLifDoubleGexc, "gexc"},
            {kLifDoubleGinh, "ginh"},
            {kLifDoubleFired, "fired"},
            {kLifDoubleStepsSinceLastSpike, "steps_since_last_spike"},
            {kLifDoubleVRest, "v_rest"},
            {kLifDoubleVReset, "v_reset"},
            {kLifDoubleVThreshold, "v_threshold"},
            {kLifDoubleEExc, "e_exc"},
            {kLifDoubleEInh, "e_inh"},
            {kLifDoubleExcDecay, "exc_decay"},
            {kLifDoubleInhDecay, "inh_decay"},
            {kLifDoubleMembraneAlpha, "membrane_alpha"},
            {kLifDoubleRefractorySteps, "refractory_steps"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        return {
            {PendingChannel::ExcitatoryConductance, "gexc", true},
            {PendingChannel::InhibitoryConductance, "ginh", true},
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
        FillFloatField(table, spec, "tau_exc_ms", {"gexc_tau", "tau_exc_ms"}, 5.0f);
        FillFloatField(table, spec, "tau_inh_ms", {"ginh_tau", "tau_inh_ms"}, 10.0f);
        FillFloatField(table, spec, "v_rest", {"V_rest", "v_rest"}, -65.0f);
        FillFloatField(table, spec, "v_reset", {"V_reset", "v_reset"}, -65.0f);
        FillFloatField(table, spec, "v_threshold", {"V_th", "v_threshold"}, -50.0f);
        FillFloatField(table, spec, "e_exc", {"Eexc", "e_exc"}, 0.0f);
        FillFloatField(table, spec, "e_inh", {"Einh", "Einhibitory", "e_inh"}, -80.0f);
        FillFloatField(table, spec, "t_ref", {"t_ref"}, 50.0f);
        return ResetStateFields(table, spec, nullptr);
    }
    bool BuildDerivedFields(DenseNeuronHostFieldTable* table,
                            const DenseNeuronModelSpec& spec,
                            float dt_ms,
                            std::string* reason) const override {
        const float* tau_m = FloatField(*table, "tau_m_ms");
        const float* tau_exc = FloatField(*table, "tau_exc_ms");
        const float* tau_inh = FloatField(*table, "tau_inh_ms");
        const float* t_ref = FloatField(*table, "t_ref");
        float* exc_decay = FloatField(table, "exc_decay");
        float* inh_decay = FloatField(table, "inh_decay");
        float* membrane_alpha = FloatField(table, "membrane_alpha");
        int* refractory_steps = IntField(table, "refractory_steps");
        const DenseNeuronRange& range = spec.range;
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            if (tau_m[neuron_id] <= 0.0f || tau_exc[neuron_id] <= 0.0f || tau_inh[neuron_id] <= 0.0f) {
                if (reason != nullptr) {
                    *reason = "LIF double time constants must be positive";
                }
                return false;
            }
            exc_decay[neuron_id] = std::exp(-dt_ms / tau_exc[neuron_id]);
            inh_decay[neuron_id] = std::exp(-dt_ms / tau_inh[neuron_id]);
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
        FillFloatConstant(table, spec, "gexc", 0.0f);
        FillFloatConstant(table, spec, "ginh", 0.0f);
        FillByteConstant(table, spec, "fired", 0);
        FillIntConstant(table, spec, "steps_since_last_spike", 10000);
        return true;
    }

protected:
    std::vector<const char*> ParameterKeys() const override {
        return {"tau", "tau_m_ms", "gexc_tau", "tau_exc_ms", "ginh_tau", "tau_inh_ms",
                "V_rest", "v_rest", "V_reset", "v_reset", "V_th", "v_threshold",
                "Eexc", "e_exc", "Einh", "Einhibitory", "e_inh", "R", "r", "t_ref"};
    }
};

}  // namespace
}  // namespace npgr

#endif
