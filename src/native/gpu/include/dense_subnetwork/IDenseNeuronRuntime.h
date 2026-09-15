#ifndef NPGR_I_DENSE_NEURON_RUNTIME_H
#define NPGR_I_DENSE_NEURON_RUNTIME_H

#include "dense_subnetwork/DenseNeuronDeviceViews.h"
#include "dense_subnetwork/DenseNeuronModelSpec.h"

#include <string>
#include <vector>

namespace npgr {

class IDenseNeuronRuntime {
public:
    virtual ~IDenseNeuronRuntime() {}

    virtual bool Initialize(const DenseNeuronModelSpec& spec,
                            int neuron_count,
                            float dt_ms,
                            std::string* reason) = 0;
    virtual bool AllocateDeviceBuffers(std::string* reason) = 0;
    virtual bool UploadHostToDevice(std::string* reason) = 0;
    virtual bool Step(const DeviceCommonBuffersView& common,
                      int current_time_step,
                      std::vector<int>* firing_ids,
                      std::string* reason) = 0;
    virtual bool ResetState(std::string* reason) = 0;
    virtual bool ExportDebugState(DenseNeuronDebugSnapshot* out,
                                  std::string* reason) const = 0;

    virtual int factory_model_id() const = 0;
    virtual int model_id() const = 0;
};

}  // namespace npgr

#endif
