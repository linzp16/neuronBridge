#ifndef NPGR_DENSE_UNIFIED_NEURON_RUNTIME_H
#define NPGR_DENSE_UNIFIED_NEURON_RUNTIME_H

#include "dense_subnetwork/DenseNeuronDeviceViews.h"
#include "dense_subnetwork/DenseNeuronModelSpec.h"
#include "dense_subnetwork/model/DenseNeuronFieldTable.h"
#include "dense_subnetwork/model/DenseNeuronModelFieldIndex.h"

#include <cstddef>
#include <string>
#include <vector>

namespace npgr {

class DenseUnifiedNeuronRuntime {
public:
    DenseUnifiedNeuronRuntime();
    ~DenseUnifiedNeuronRuntime();

    bool Initialize(const std::vector<DenseNeuronModelSpec>& specs,
                    int neuron_count,
                    float dt_ms,
                    std::string* reason = nullptr);
    bool AllocateDeviceBuffers(std::string* reason = nullptr);
    bool UploadHostToDevice(std::string* reason = nullptr);
    bool ConfigureOutputNeuronMask(const std::vector<unsigned char>& output_neuron_mask,
                                   std::string* reason = nullptr);
    bool Step(const DeviceCommonBuffersView& common,
              int current_time_step,
              std::vector<int>* output_firing_ids,
              std::vector<int>* full_firing_ids,
              bool export_full_firings_to_host,
              std::string* reason = nullptr);
    bool ResetState(std::string* reason = nullptr);
    bool SyncHostStateFromDevice(std::string* reason = nullptr) const;
    bool BuildDebugSnapshots(std::vector<DenseNeuronDebugSnapshot>* out,
                             std::string* reason = nullptr) const;

    const int* device_full_firing_ids() const;
    const int* device_full_firing_count() const;
    const unsigned char* device_current_did_fire() const;

private:
    void DestroyDeviceBuffers();
    bool ResetStateFieldsGpu(std::string* reason);
    bool DownloadFieldsToHost(DenseNeuronHostFieldTable* out, std::string* reason) const;
    void ClearHostBuildCaches();

    struct DenseRuntimeModelDebugInfo {
        int factory_model_id = -1;
        int model_id = -1;
        std::string legacy_model_name;
    };

    DenseNeuronHostFieldTable initial_host_fields_;
    std::vector<DenseFieldSpan> host_field_spans_;
    std::vector<DenseRuntimeModelDebugInfo> debug_model_infos_;
    DenseModelFieldIndexTable model_field_indices_;
    std::size_t float_pool_size_;
    std::size_t int_pool_size_;
    std::size_t byte_pool_size_;
    int neuron_count_;
    float dt_ms_;
    bool initialized_;
    bool device_ready_;
    mutable bool host_state_dirty_;

    std::vector<int> model_id_by_neuron_;
    std::vector<unsigned char> active_mask_;
    std::vector<unsigned char> output_neuron_mask_;
    int* d_model_id_by_neuron_;
    unsigned char* d_active_mask_;
    unsigned char* d_output_neuron_mask_;
    DenseNeuronDeviceFieldTable device_fields_;
    DenseDeviceFieldSpan* d_field_spans_;
    float* d_float_field_pool_;
    int* d_int_field_pool_;
    unsigned char* d_byte_field_pool_;
    float* d_reset_float_field_pool_;
    int* d_reset_int_field_pool_;
    unsigned char* d_reset_byte_field_pool_;
    DenseDeviceModelFieldIndexTable device_model_field_indices_;
    DenseModelFieldIndexSpan* d_model_field_spans_;
    int* d_model_field_indices_;
    int* d_model_field_span_by_neuron_;
    int* d_current_firing_ids_;
    int* d_current_firing_count_;
    unsigned char* d_current_did_fire_;
    int* d_output_firing_ids_;
    int* d_output_firing_count_;
    std::size_t device_firing_capacity_;
    std::size_t device_output_firing_capacity_;
};

}  // namespace npgr

#endif
