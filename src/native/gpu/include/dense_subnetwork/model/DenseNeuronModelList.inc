// Central dense neuron model registry list.
//
// This file is intentionally macro-based so host registration, stable model
// ids, and CUDA dispatch are generated from the same source of truth without
// adding runtime indirection.
//
// The including file must define:
//   NPGR_DENSE_NEURON_MODEL(symbol, id, host_class, device_update_fn)
NPGR_DENSE_NEURON_MODEL(kLifExponentialDoubleModelId,
                        1,
                        LifExponentialDoubleModel,
                        UpdateLifExponentialDoubleDeviceEntry)
NPGR_DENSE_NEURON_MODEL(kLifExponentialDecayModelId,
                        2,
                        LifExponentialDecayModel,
                        UpdateLifExponentialDecayDeviceEntry)
NPGR_DENSE_NEURON_MODEL(kPoissonRateModelId,
                        3,
                        PoissonRateModel,
                        UpdatePoissonRateDeviceEntry)
NPGR_DENSE_NEURON_MODEL(kIzhikevichExponentialDecayModelId,
                        4,
                        IzhikevichExponentialDecayModel,
                        UpdateIzhikevichExponentialDecayDeviceEntry)
NPGR_DENSE_NEURON_MODEL(kTriggerRelayModelId,
                        5,
                        TriggerRelayModel,
                        UpdateTriggerRelayDeviceEntry)
NPGR_DENSE_NEURON_MODEL(kLifVoltageJumpModelId,
                        6,
                        LifVoltageJumpModel,
                        UpdateLifVoltageJumpDeviceEntry)
NPGR_DENSE_NEURON_MODEL(kLifExponentialTripleModelId,
                        7,
                        LifExponentialTripleModel,
                        UpdateLifExponentialTripleDeviceEntry)
#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomDenseNeuronModelList.inc"
#endif
