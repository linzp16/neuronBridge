// Central dense learning model registry list.
//
// The including file must define:
//   NPGR_DENSE_LEARNING_MODEL(symbol, id, host_class, pre_fn, post_fn, trigger_fn)
//
// pre_fn, post_fn, and trigger_fn are small device-entry wrappers with unified
// signatures. They keep CUDA dispatch generated from this list while preserving
// the existing specialized learning implementations.
NPGR_DENSE_LEARNING_MODEL(kDenseLearningStdpModelId,
                          0,
                          DenseStdpLearningModel,
                          ApplyStdpPreDeviceEntry,
                          ApplyStdpPostDeviceEntry,
                          ApplyDenseLearningNoTriggerDeviceEntry)
NPGR_DENSE_LEARNING_MODEL(kDenseLearningRStdpModelId,
                          1,
                          DenseRStdpLearningModel,
                          ApplyRStdpPreDeviceEntry,
                          ApplyRStdpPostDeviceEntry,
                          ApplyRStdpTriggerDeviceEntry)
NPGR_DENSE_LEARNING_MODEL(kDenseLearningAdditiveKernelModelId,
                          2,
                          DenseAdditiveKernelLearningModel,
                          ApplyAdditiveKernelPreDeviceEntry,
                          ApplyDenseLearningNoPostDeviceEntry,
                          ApplyAdditiveKernelTriggerDeviceEntry)
NPGR_DENSE_LEARNING_MODEL(kDenseLearningCerebellarModelId,
                          3,
                          DenseCerebellarLearningModel,
                          ApplyCerebellarPreDeviceEntry,
                          ApplyDenseLearningNoPostDeviceEntry,
                          ApplyCerebellarTriggerDeviceEntry)
#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomDenseLearningModelList.inc"
#endif
