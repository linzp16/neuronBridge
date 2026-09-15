#ifndef NPGR_DENSE_BUILTIN_NEURON_MODELS_H
#define NPGR_DENSE_BUILTIN_NEURON_MODELS_H

#include "dense_subnetwork/model/DenseIzhikevichExponentialDecayModel.h"
#include "dense_subnetwork/model/DenseLifExponentialDecayModel.h"
#include "dense_subnetwork/model/DenseLifExponentialDoubleModel.h"
#include "dense_subnetwork/model/DenseLifExponentialTripleModel.h"
#include "dense_subnetwork/model/DenseLifVoltageJumpModel.h"
#include "dense_subnetwork/model/DensePoissonRateModel.h"
#include "dense_subnetwork/model/DenseTriggerRelayModel.h"

#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomGeneratedDenseNeuronModels.h"
#endif

#endif
