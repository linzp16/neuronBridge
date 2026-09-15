#include "dense_subnetwork/model/DenseNeuronFieldAccess.h"

namespace npgr {

const DenseFieldSpan* FindField(const DenseNeuronHostFieldTable& table,
                                const std::string& name) {
    std::unordered_map<std::string, int>::const_iterator found = table.field_id_by_name.find(name);
    if (found == table.field_id_by_name.end()) {
        return nullptr;
    }
    const int field_id = found->second;
    if (field_id < 0 || field_id >= static_cast<int>(table.fields.size())) {
        return nullptr;
    }
    return &table.fields[static_cast<std::size_t>(field_id)];
}

DenseFieldSpan* FindField(DenseNeuronHostFieldTable* table,
                          const std::string& name) {
    if (table == nullptr) {
        return nullptr;
    }
    std::unordered_map<std::string, int>::const_iterator found = table->field_id_by_name.find(name);
    if (found == table->field_id_by_name.end()) {
        return nullptr;
    }
    const int field_id = found->second;
    if (field_id < 0 || field_id >= static_cast<int>(table->fields.size())) {
        return nullptr;
    }
    return &table->fields[static_cast<std::size_t>(field_id)];
}

float* FloatField(DenseNeuronHostFieldTable* table, const std::string& name) {
    DenseFieldSpan* span = FindField(table, name);
    if (table == nullptr || span == nullptr || span->storage != DenseFieldStorage::Float32) {
        return nullptr;
    }
    return table->float_pool.data() + span->offset;
}

const float* FloatField(const DenseNeuronHostFieldTable& table, const std::string& name) {
    const DenseFieldSpan* span = FindField(table, name);
    if (span == nullptr || span->storage != DenseFieldStorage::Float32) {
        return nullptr;
    }
    return table.float_pool.data() + span->offset;
}

int* IntField(DenseNeuronHostFieldTable* table, const std::string& name) {
    DenseFieldSpan* span = FindField(table, name);
    if (table == nullptr || span == nullptr || span->storage != DenseFieldStorage::Int32) {
        return nullptr;
    }
    return table->int_pool.data() + span->offset;
}

const int* IntField(const DenseNeuronHostFieldTable& table, const std::string& name) {
    const DenseFieldSpan* span = FindField(table, name);
    if (span == nullptr || span->storage != DenseFieldStorage::Int32) {
        return nullptr;
    }
    return table.int_pool.data() + span->offset;
}

unsigned char* ByteField(DenseNeuronHostFieldTable* table, const std::string& name) {
    DenseFieldSpan* span = FindField(table, name);
    if (table == nullptr || span == nullptr || span->storage != DenseFieldStorage::UInt8) {
        return nullptr;
    }
    return table->byte_pool.data() + span->offset;
}

const unsigned char* ByteField(const DenseNeuronHostFieldTable& table, const std::string& name) {
    const DenseFieldSpan* span = FindField(table, name);
    if (span == nullptr || span->storage != DenseFieldStorage::UInt8) {
        return nullptr;
    }
    return table.byte_pool.data() + span->offset;
}

}  // namespace npgr
