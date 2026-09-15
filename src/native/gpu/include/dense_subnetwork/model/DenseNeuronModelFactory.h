#ifndef NPGR_DENSE_NEURON_MODEL_FACTORY_H
#define NPGR_DENSE_NEURON_MODEL_FACTORY_H

#include "dense_subnetwork/model/DenseNeuronInputSchema.h"

#include <memory>
#include <string>
#include <vector>

struct NeuronLayerDescription;

namespace npgr {

class IDenseNeuronModel;

// Registry for dense neuron model metadata. Build code queries this factory
// once and compiles model-specific behavior into flat runtime arrays.
class DenseNeuronModelFactory {
public:
    // Stable factory-owned model ids used by the unified dense neuron dispatch path.
    static constexpr int kUnknownModelId = 0;
#define NPGR_DENSE_NEURON_MODEL(symbol, id, host_class, device_update_fn) \
    static constexpr int symbol = id;
#include "dense_subnetwork/model/DenseNeuronModelList.inc"
#undef NPGR_DENSE_NEURON_MODEL

    // Interface slot dedupe encodes (target_neuron, pending_channel) with one
    // byte reserved for the pending channel id.
    static constexpr int kPendingChannelKeyStride = 8;

    static DenseNeuronModelFactory& Instance();
    ~DenseNeuronModelFactory();

    const IDenseNeuronModel* FindByLegacyName(const std::string& name) const;
    const IDenseNeuronModel* FindByModelId(int factory_model_id) const;
    bool IsDenseManagedLayer(const NeuronLayerDescription& layer) const;
    bool ResolveSpikeEffect(int factory_model_id,
                            int synapse_type,
                            PendingChannel* channel,
                            float* scale) const;
    int RequiredPendingChannelCount() const;
    // Runs host-side metadata checks for every registered model. It is called
    // during dense build/init so extension mistakes fail before kernel launch.
    bool ValidateRegisteredModels(std::string* reason = nullptr) const;

private:
    DenseNeuronModelFactory();
    void Register(std::unique_ptr<IDenseNeuronModel> model);

    std::vector<std::unique_ptr<IDenseNeuronModel> > models_;
};

}  // namespace npgr

#endif
