#include "dense_subnetwork/model/DenseNeuronModelFactory.h"

#include "dense_subnetwork/model/DenseBuiltinNeuronModels.h"
#include "neuron_model/NeuronModelCatalog.h"

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <utility>

namespace npgr {

DenseNeuronModelFactory& DenseNeuronModelFactory::Instance() {
    static DenseNeuronModelFactory factory;
    return factory;
}

DenseNeuronModelFactory::DenseNeuronModelFactory() {
#define NPGR_DENSE_NEURON_MODEL(symbol, id, host_class, device_update_fn) \
    Register(std::unique_ptr<IDenseNeuronModel>(new host_class()));
#include "dense_subnetwork/model/DenseNeuronModelList.inc"
#undef NPGR_DENSE_NEURON_MODEL
}

DenseNeuronModelFactory::~DenseNeuronModelFactory() = default;

void DenseNeuronModelFactory::Register(std::unique_ptr<IDenseNeuronModel> model) {
    models_.push_back(std::move(model));
}

const IDenseNeuronModel* DenseNeuronModelFactory::FindByLegacyName(const std::string& name) const {
    const std::string canonical_name =
        NeuronModelCatalog::Instance().ResolveCanonicalName(name);
    for (std::size_t index = 0; index < models_.size(); ++index) {
        if (models_[index]->CanonicalName() == canonical_name ||
            models_[index]->MatchesLegacyName(name)) {
            return models_[index].get();
        }
    }
    return nullptr;
}

const IDenseNeuronModel* DenseNeuronModelFactory::FindByModelId(int factory_model_id) const {
    for (std::size_t index = 0; index < models_.size(); ++index) {
        if (models_[index]->FactoryModelId() == factory_model_id) {
            return models_[index].get();
        }
    }
    return nullptr;
}

bool DenseNeuronModelFactory::IsDenseManagedLayer(const NeuronLayerDescription& layer) const {
    return NeuronModelCatalog::Instance().IsSupported(layer.ModelName, NeuronBackend::DenseGpu) &&
           FindByLegacyName(layer.ModelName) != nullptr;
}

bool DenseNeuronModelFactory::ResolveSpikeEffect(int factory_model_id,
                                                 int synapse_type,
                                                 PendingChannel* channel,
                                                 float* scale) const {
    const IDenseNeuronModel* model = FindByModelId(factory_model_id);
    if (model == nullptr) {
        return false;
    }
    const std::vector<DenseSpikeEffectBinding> effects = model->SpikeEffects();
    for (std::size_t index = 0; index < effects.size(); ++index) {
        if (effects[index].synapse_type == synapse_type) {
            if (channel != nullptr) {
                *channel = effects[index].channel;
            }
            if (scale != nullptr) {
                *scale = effects[index].scale;
            }
            return true;
        }
    }
    return false;
}

int DenseNeuronModelFactory::RequiredPendingChannelCount() const {
    int channel_count = 0;
    for (std::size_t model_index = 0; model_index < models_.size(); ++model_index) {
        const std::vector<DenseInputChannelBinding> channels = models_[model_index]->InputChannels();
        for (std::size_t channel_index = 0; channel_index < channels.size(); ++channel_index) {
            const int id = static_cast<int>(channels[channel_index].channel);
            channel_count = std::max(channel_count, id + 1);
        }
    }
    return channel_count;
}

bool DenseNeuronModelFactory::ValidateRegisteredModels(std::string* reason) const {
    std::set<int> ids;
    std::set<std::string> canonical_names;
    for (std::size_t index = 0; index < models_.size(); ++index) {
        const IDenseNeuronModel* model = models_[index].get();
        if (model == nullptr) {
            if (reason != nullptr) {
                *reason = "dense neuron factory contains a null model";
            }
            return false;
        }
        if (!model->Validate(reason)) {
            return false;
        }
        if (!ids.insert(model->FactoryModelId()).second) {
            if (reason != nullptr) {
                std::ostringstream oss;
                oss << "duplicate dense neuron model id: " << model->FactoryModelId();
                *reason = oss.str();
            }
            return false;
        }
        const std::string canonical_name = model->CanonicalName();
        if (!canonical_names.insert(canonical_name).second) {
            if (reason != nullptr) {
                *reason = "duplicate dense neuron canonical name: " + canonical_name;
            }
            return false;
        }
        if (!NeuronModelCatalog::Instance().IsSupported(canonical_name, NeuronBackend::DenseGpu)) {
            if (reason != nullptr) {
                *reason = "dense neuron model is missing from NeuronModelCatalog: " + canonical_name;
            }
            return false;
        }
    }
    const std::vector<std::string> dense_names =
        NeuronModelCatalog::Instance().SupportedModelNames(NeuronBackend::DenseGpu);
    for (std::size_t index = 0; index < dense_names.size(); ++index) {
        if (FindByLegacyName(dense_names[index]) == nullptr) {
            if (reason != nullptr) {
                *reason = "NeuronModelCatalog marks an unregistered dense model as supported: " +
                    dense_names[index];
            }
            return false;
        }
    }
    return true;
}

}  // namespace npgr
