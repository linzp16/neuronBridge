#ifndef NPGR_DENSE_NEURON_DEVICE_UPDATE_CUH
#define NPGR_DENSE_NEURON_DEVICE_UPDATE_CUH

#include "dense_subnetwork/model/DenseBuiltinNeuronDeviceUpdates.cuh"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"

namespace npgr {

__device__ inline DenseDeviceModelUpdateFn ResolveDenseDeviceModelUpdate(int factory_model_id) {
    switch (factory_model_id) {
#define NPGR_DENSE_NEURON_MODEL(symbol, id, host_class, device_update_fn) \
        case DenseNeuronModelFactory::symbol:                             \
            return device_update_fn;
#include "dense_subnetwork/model/DenseNeuronModelList.inc"
#undef NPGR_DENSE_NEURON_MODEL
        default:
            return nullptr;
    }
}

}  // namespace npgr

#endif
