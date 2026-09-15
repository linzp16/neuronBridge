#ifndef NPGR_DENSE_NEURON_FIELD_TABLE_BUILDER_H
#define NPGR_DENSE_NEURON_FIELD_TABLE_BUILDER_H

#include "dense_subnetwork/DenseNeuronModelSpec.h"
#include "dense_subnetwork/model/DenseNeuronFieldTable.h"

#include <string>
#include <vector>

namespace npgr {

bool BuildDenseNeuronHostFieldTable(const std::vector<DenseNeuronModelSpec>& specs,
                                    int neuron_count,
                                    DenseNeuronHostFieldTable* table,
                                    std::string* reason = nullptr);

}  // namespace npgr

#endif
