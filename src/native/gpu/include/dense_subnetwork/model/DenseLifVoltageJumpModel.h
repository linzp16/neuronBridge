#ifndef NPGR_DENSELIFVOLTAGEJUMPMODEL_H
#define NPGR_DENSELIFVOLTAGEJUMPMODEL_H

#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this model and define the compact field-id order
// consumed by the CUDA update entry.
enum LifVoltageJumpFieldSlot {
    kLifVoltageJumpV,
    kLifVoltageJumpFired,
    kLifVoltageJumpStepsSinceLastSpike,
    kLifVoltageJumpVRest,
    kLifVoltageJumpVReset,
    kLifVoltageJumpVThreshold,
    kLifVoltageJumpR,
    kLifVoltageJumpMembraneAlpha,
    kLifVoltageJumpRefractorySteps,
    kLifVoltageJumpSlotCount,
};

class LifVoltageJumpModel : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override { return DenseNeuronModelFactory::kLifVoltageJumpModelId; }
    const char* CanonicalName() const override { return "TimeDrivenLIF_Voltage_jump"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "TimeDrivenLIF_Voltage_jump" ||
               name == "TimeDrivenLIF_Voltage_jump_GPU";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"v", DenseFieldStorage::Float32, DenseFieldRole::State, -65.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
            {"steps_since_last_spike", DenseFieldStorage::Int32, DenseFieldRole::State, 0.0f, 10000},
            {"tau_m_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 20.0f},
            {"v_rest", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"v_reset", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"v_threshold", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -50.0f},
            {"r", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"t_ref", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 50.0f},
            {"membrane_alpha", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"refractory_steps", DenseFieldStorage::Int32, DenseFieldRole::Derived, 0.0f, 0},
        };
    }
    int FieldSlotCount() const override { return kLifVoltageJumpSlotCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kLifVoltageJumpV, "v"},
            {kLifVoltageJumpFired, "fired"},
            {kLifVoltageJumpStepsSinceLastSpike, "steps_since_last_spike"},
            {kLifVoltageJumpVRest, "v_rest"},
            {kLifVoltageJumpVReset, "v_reset"},
            {kLifVoltageJumpVThreshold, "v_threshold"},
            {kLifVoltageJumpR, "r"},
            {kLifVoltageJumpMembraneAlpha, "membrane_alpha"},
            {kLifVoltageJumpRefractorySteps, "refractory_steps"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        // Voltage-jump LIF consumes pending channels as direct membrane/current
        // increments, so no extra conductance state is needed.
        return {
            {PendingChannel::ExcitatoryConductance, "voltage_jump_exc", false},
            {PendingChannel::InhibitoryConductance, "voltage_jump_inh", false},
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
        FillFloatField(table, spec, "v_rest", {"V_rest", "v_rest"}, -65.0f);
        FillFloatField(table, spec, "v_reset", {"V_reset", "v_reset"}, -65.0f);
        FillFloatField(table, spec, "v_threshold", {"V_th", "v_threshold"}, -50.0f);
        FillFloatField(table, spec, "r", {"R", "r"}, 1.0f);
        FillFloatField(table, spec, "t_ref", {"t_ref"}, 50.0f);
        return ResetStateFields(table, spec, nullptr);
    }
    bool BuildDerivedFields(DenseNeuronHostFieldTable* table,
                            const DenseNeuronModelSpec& spec,
                            float dt_ms,
                            std::string* reason) const override {
        const float* tau_m = FloatField(*table, "tau_m_ms");
        const float* t_ref = FloatField(*table, "t_ref");
        float* membrane_alpha = FloatField(table, "membrane_alpha");
        int* refractory_steps = IntField(table, "refractory_steps");
        const DenseNeuronRange& range = spec.range;
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            if (tau_m[neuron_id] <= 0.0f) {
                if (reason != nullptr) {
                    *reason = "Voltage-jump LIF membrane time constant must be positive";
                }
                return false;
            }
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
        FillByteConstant(table, spec, "fired", 0);
        FillIntConstant(table, spec, "steps_since_last_spike", 10000);
        return true;
    }

protected:
    std::vector<const char*> ParameterKeys() const override {
        return {"tau", "tau_m_ms", "V_rest", "v_rest", "V_reset", "v_reset",
                "V_th", "v_threshold", "R", "r", "t_ref"};
    }
};

}  // namespace
}  // namespace npgr

#endif
