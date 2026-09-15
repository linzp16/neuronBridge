#ifndef NPGR_DENSE_BUILTIN_LEARNING_RULE_MODELS_H
#define NPGR_DENSE_BUILTIN_LEARNING_RULE_MODELS_H

#include "dense_subnetwork/learning/DenseAdditiveKernelLearningModel.h"
#include "dense_subnetwork/learning/DenseCerebellarLearningModel.h"
#include "dense_subnetwork/learning/DenseRStdpLearningModel.h"
#include "dense_subnetwork/learning/DenseStdpLearningModel.h"
#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomGeneratedDenseLearningRules.h"
#endif

#endif
