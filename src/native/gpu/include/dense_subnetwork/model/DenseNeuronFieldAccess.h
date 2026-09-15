#ifndef NPGR_DENSE_NEURON_FIELD_ACCESS_H
#define NPGR_DENSE_NEURON_FIELD_ACCESS_H

#include "dense_subnetwork/model/DenseNeuronFieldTable.h"

#include <string>

namespace npgr {

const DenseFieldSpan* FindField(const DenseNeuronHostFieldTable& table,
                                const std::string& name);
DenseFieldSpan* FindField(DenseNeuronHostFieldTable* table,
                          const std::string& name);

float* FloatField(DenseNeuronHostFieldTable* table, const std::string& name);
const float* FloatField(const DenseNeuronHostFieldTable& table, const std::string& name);
int* IntField(DenseNeuronHostFieldTable* table, const std::string& name);
const int* IntField(const DenseNeuronHostFieldTable& table, const std::string& name);
unsigned char* ByteField(DenseNeuronHostFieldTable* table, const std::string& name);
const unsigned char* ByteField(const DenseNeuronHostFieldTable& table, const std::string& name);

}  // namespace npgr

#endif
