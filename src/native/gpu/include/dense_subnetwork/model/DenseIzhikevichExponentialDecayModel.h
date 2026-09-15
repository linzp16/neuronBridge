#ifndef NPGR_DENSEIZHIKEVICHEXPONENTIALDECAYMODEL_H
#define NPGR_DENSEIZHIKEVICHEXPONENTIALDECAYMODEL_H

#include "dense_subnetwork/model/DenseBuiltinNeuronModelCommon.h"

namespace npgr {
namespace {

// Field slots are local to this model and define the compact field-id order
// consumed by the CUDA update entry.
enum IzhikevichFieldSlot {
    kIzhV,
    kIzhU,
    kIzhGexc,
    kIzhGinh,
    kIzhNmdaG,
    kIzhFired,
    kIzhA,
    kIzhB,
    kIzhC,
    kIzhD,
    kIzhVThreshold,
    kIzhR,
    kIzhEExc,
    kIzhEInh,
    kIzhExcDecay,
    kIzhInhDecay,
    kIzhNmdaDecay,
    kIzhSlotCount,
};

class IzhikevichExponentialDecayModel : public DenseNeuronModelBase {
public:
    int FactoryModelId() const override { return DenseNeuronModelFactory::kIzhikevichExponentialDecayModelId; }
    const char* CanonicalName() const override { return "TimeDrivenIzhikevic_Exponential_Decay"; }
    bool MatchesLegacyName(const std::string& name) const override {
        return name == "TimeDrivenIzhikevic_I_Exponential_Decay" ||
               name == "TimeDrivenIzhikevic_Exponential_Decay";
    }
    std::vector<DenseFieldSchema> Fields() const override {
        return {
            {"v", DenseFieldStorage::Float32, DenseFieldRole::State, -65.0f},
            {"u", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"gexc", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"ginh", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"nmda_g", DenseFieldStorage::Float32, DenseFieldRole::State, 0.0f},
            {"fired", DenseFieldStorage::UInt8, DenseFieldRole::State, 0.0f, 0, 0},
            {"a", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.02f},
            {"b", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.2f},
            {"v_rest", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"c", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -65.0f},
            {"d", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 8.0f},
            {"v_threshold", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -50.0f},
            {"r", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 1.0f},
            {"E_ampa", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 0.0f},
            {"E_gaba", DenseFieldStorage::Float32, DenseFieldRole::Parameter, -80.0f},
            {"ampa_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 5.0f},
            {"gaba_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 10.0f},
            {"nmda_tau", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 100.0f},
            {"exc_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"inh_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
            {"nmda_decay", DenseFieldStorage::Float32, DenseFieldRole::Derived, 0.0f},
        };
    }
    int FieldSlotCount() const override { return kIzhSlotCount; }
    std::vector<DenseFieldSlotBinding> FieldSlots() const override {
        return {
            {kIzhV, "v"},
            {kIzhU, "u"},
            {kIzhGexc, "gexc"},
            {kIzhGinh, "ginh"},
            {kIzhNmdaG, "nmda_g"},
            {kIzhFired, "fired"},
            {kIzhA, "a"},
            {kIzhB, "b"},
            {kIzhC, "c"},
            {kIzhD, "d"},
            {kIzhVThreshold, "v_threshold"},
            {kIzhR, "r"},
            {kIzhEExc, "E_ampa"},
            {kIzhEInh, "E_gaba"},
            {kIzhExcDecay, "exc_decay"},
            {kIzhInhDecay, "inh_decay"},
            {kIzhNmdaDecay, "nmda_decay"},
        };
    }
    std::vector<DenseInputChannelBinding> InputChannels() const override {
        return {
            {PendingChannel::ExcitatoryConductance, "ampa", true},
            {PendingChannel::InhibitoryConductance, "gaba", true},
            {PendingChannel::NmdaConductance, "nmda", false},
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
        FillFloatField(table, spec, "a", {"a"}, 0.02f);
        FillFloatField(table, spec, "b", {"b"}, 0.2f);
        FillFloatField(table, spec, "v_rest", {"V_rest", "v_rest"}, -65.0f);
        FillFloatField(table, spec, "c", {"c", "V_reset", "v_reset"}, -65.0f);
        FillFloatField(table, spec, "d", {"d"}, 8.0f);
        FillFloatField(table, spec, "v_threshold", {"V_th", "v_threshold"}, -50.0f);
        FillFloatField(table, spec, "r", {"R", "r"}, 1.0f);
        FillFloatField(table, spec, "E_ampa", {"E_ampa", "Eexc", "e_exc"}, 0.0f);
        FillFloatField(table, spec, "E_gaba", {"E_gaba", "Einh", "e_inh"}, -80.0f);
        FillFloatField(table, spec, "ampa_tau", {"ampa_tau", "tau_exc_ms"}, 5.0f);
        FillFloatField(table, spec, "gaba_tau", {"gaba_tau", "tau_inh_ms"}, 10.0f);
        FillFloatField(table, spec, "nmda_tau", {"nmda_tau"}, 100.0f);
        return ResetStateFields(table, spec, nullptr);
    }
    bool BuildDerivedFields(DenseNeuronHostFieldTable* table,
                            const DenseNeuronModelSpec& spec,
                            float dt_ms,
                            std::string* reason) const override {
        const float* ampa_tau = FloatField(*table, "ampa_tau");
        const float* gaba_tau = FloatField(*table, "gaba_tau");
        const float* nmda_tau = FloatField(*table, "nmda_tau");
        float* exc_decay = FloatField(table, "exc_decay");
        float* inh_decay = FloatField(table, "inh_decay");
        float* nmda_decay = FloatField(table, "nmda_decay");
        const DenseNeuronRange& range = spec.range;
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            if (ampa_tau[neuron_id] <= 0.0f || gaba_tau[neuron_id] <= 0.0f || nmda_tau[neuron_id] <= 0.0f) {
                if (reason != nullptr) {
                    *reason = "Izhikevich conductance time constants must be positive";
                }
                return false;
            }
            exc_decay[neuron_id] = std::exp(-dt_ms / ampa_tau[neuron_id]);
            inh_decay[neuron_id] = std::exp(-dt_ms / gaba_tau[neuron_id]);
            nmda_decay[neuron_id] = std::exp(-dt_ms / nmda_tau[neuron_id]);
        }
        return true;
    }
    bool ResetStateFields(DenseNeuronHostFieldTable* table,
                          const DenseNeuronModelSpec& spec,
                          std::string*) const override {
        const float* v_rest = FloatField(*table, "v_rest");
        const float* b = FloatField(*table, "b");
        float* v = FloatField(table, "v");
        float* u = FloatField(table, "u");
        const DenseNeuronRange& range = spec.range;
        for (int offset = 0; offset < range.count; ++offset) {
            const int neuron_id = range.begin + offset;
            v[neuron_id] = v_rest[neuron_id];
            u[neuron_id] = b[neuron_id] * v_rest[neuron_id];
        }
        FillFloatConstant(table, spec, "gexc", 0.0f);
        FillFloatConstant(table, spec, "ginh", 0.0f);
        FillFloatConstant(table, spec, "nmda_g", 0.0f);
        FillByteConstant(table, spec, "fired", 0);
        return true;
    }

protected:
    std::vector<const char*> ParameterKeys() const override {
        return {"a", "b", "c", "d", "V_rest", "V_reset", "V_th", "R",
                "E_ampa", "ampa_tau", "E_gaba", "gaba_tau", "nmda_tau"};
    }
};

}  // namespace
}  // namespace npgr

#endif
