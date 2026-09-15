#ifndef NPGR_DENSE_BUILTIN_NEURON_DEVICE_UPDATES_CUH
#define NPGR_DENSE_BUILTIN_NEURON_DEVICE_UPDATES_CUH

#include "dense_subnetwork/model/DenseIzhikevichExponentialDecayDeviceUpdate.cuh"
#include "dense_subnetwork/model/DenseLifExponentialDecayDeviceUpdate.cuh"
#include "dense_subnetwork/model/DenseLifExponentialDoubleDeviceUpdate.cuh"
#include "dense_subnetwork/model/DenseLifExponentialTripleDeviceUpdate.cuh"
#include "dense_subnetwork/model/DenseLifVoltageJumpDeviceUpdate.cuh"
#include "dense_subnetwork/model/DensePoissonRateDeviceUpdate.cuh"
#include "dense_subnetwork/model/DenseTriggerRelayDeviceUpdate.cuh"

#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomGeneratedDenseNeuronDeviceUpdates.cuh"
#endif

#endif
