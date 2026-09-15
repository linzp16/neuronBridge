#ifndef NPGR_DENSE_NEURON_MODEL_SPEC_H
#define NPGR_DENSE_NEURON_MODEL_SPEC_H

#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include <map>
#include <string>
#include <vector>

namespace npgr {

struct DenseNeuronRange {
    int begin = 0;
    int count = 0;
};

struct DenseNeuronModelSpec {
    int factory_model_id = DenseNeuronModelFactory::kUnknownModelId;
    std::string legacy_model_name;
    int model_id = -1;
    DenseNeuronRange range;
    std::map<std::string, std::vector<float> > per_neuron_params;
};

struct DenseNeuronDebugSnapshot {
    int factory_model_id = DenseNeuronModelFactory::kUnknownModelId;
    int model_id = -1;
    std::string legacy_model_name;
    std::map<std::string, std::vector<float> > float_state_vectors;
    std::map<std::string, std::vector<unsigned char> > byte_state_vectors;
};

}  // namespace npgr

#endif
