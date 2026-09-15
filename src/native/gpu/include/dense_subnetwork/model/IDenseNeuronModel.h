#ifndef NPGR_I_DENSE_NEURON_MODEL_H
#define NPGR_I_DENSE_NEURON_MODEL_H

#include "dense_subnetwork/DenseNeuronModelSpec.h"
#include "dense_subnetwork/model/DenseNeuronFieldSchema.h"
#include "dense_subnetwork/model/DenseNeuronFieldTable.h"
#include "dense_subnetwork/model/DenseNeuronInputSchema.h"
#include "source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"

#include <string>
#include <vector>

namespace npgr {

struct DenseFieldSlotBinding {
    int slot_id = -1;
    std::string field_name;
};

// Model extension point for dense subnetworks. The propagation runtime stays
// unified; each model only declares metadata and its own future update equation.
class IDenseNeuronModel {
public:
    virtual ~IDenseNeuronModel() {}

    virtual int FactoryModelId() const = 0;
    virtual const char* CanonicalName() const = 0;
    virtual bool MatchesLegacyName(const std::string& name) const = 0;

    virtual std::vector<DenseFieldSchema> Fields() const = 0;
    virtual int FieldSlotCount() const = 0;
    virtual std::vector<DenseFieldSlotBinding> FieldSlots() const = 0;
    virtual std::vector<DenseInputChannelBinding> InputChannels() const = 0;
    virtual std::vector<DenseSpikeEffectBinding> SpikeEffects() const = 0;
    // Validate model-owned metadata during host initialization. This catches
    // extension mistakes before any CUDA kernel consumes the compact field ids.
    virtual bool Validate(std::string* reason) const;

    virtual bool FillLayerParams(DenseNeuronModelSpec* spec,
                                 const NeuronLayerDescription& layer,
                                 const DenseNeuronRange& range,
                                 int neuron_count,
                                 std::string* reason) const = 0;

    virtual bool FillInitialFieldValues(DenseNeuronHostFieldTable* table,
                                        const DenseNeuronModelSpec& spec,
                                        std::string* reason) const;
    virtual bool BuildDerivedFields(DenseNeuronHostFieldTable* table,
                                    const DenseNeuronModelSpec& spec,
                                    float dt_ms,
                                    std::string* reason) const;
    virtual bool ResetStateFields(DenseNeuronHostFieldTable* table,
                                  const DenseNeuronModelSpec& spec,
                                  std::string* reason) const;
};

}  // namespace npgr

#endif
