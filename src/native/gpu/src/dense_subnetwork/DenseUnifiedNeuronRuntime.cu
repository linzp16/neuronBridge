#include "dense_subnetwork/DenseUnifiedNeuronRuntime.h"
#include "dense_subnetwork/model/DenseNeuronDeviceUpdate.cuh"
#include "gpu_runtime/GpuLaunchConfig.h"
#include "gpu_runtime/GpuPropagationLayout.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace npgr {

namespace {

// Shared staging reduces global atomic pressure when many neurons fire in the
// same update kernel. The sparse global list is still kept for history/output.
constexpr int kBlockFiringTableCapacity = 256;

bool CheckCudaStatus(cudaError_t status, const char* action, std::string* reason) {
    if (status == cudaSuccess) {
        return true;
    }
    if (reason != nullptr) {
        std::ostringstream oss;
        oss << action << " failed: " << cudaGetErrorString(status);
        *reason = oss.str();
    }
    return false;
}

bool CopyIfNonEmpty(void* dst,
                    const void* src,
                    std::size_t bytes,
                    cudaMemcpyKind kind,
                    const char* action,
                    std::string* reason) {
    if (bytes == 0) {
        return true;
    }
    return CheckCudaStatus(cudaMemcpy(dst, src, bytes, kind), action, reason);
}

__global__ void UpdateDenseUnifiedKernel(const unsigned char* active_mask,
                                         const unsigned char* output_neuron_mask,
                                         DenseNeuronDeviceFieldTable fields,
                                         DenseDeviceModelFieldIndexTable model_field_indices,
                                         float dt_ms,
                                         int current_time_step,
                                         float* pending_channels,
                                         int pending_channel_count,
                                         int pending_channel_stride,
                                         int neuron_count,
                                         int* full_firing_ids,
                                         int* full_firing_count,
                                         unsigned char* current_did_fire,
                                         int* output_firing_ids,
                                         int* output_firing_count) {
    // build the shared memory
    extern __shared__ int block_firing_ids[];
    __shared__ int block_firing_count;
    __shared__ int block_global_offset;

    // The kernel is neuron-parallel. Each thread updates one neuron per
    // grid-stride round, writes its did-fire byte, and stages fired ids locally.
    for (int block_begin = blockIdx.x * blockDim.x;
         block_begin < neuron_count;
         block_begin += blockDim.x * gridDim.x) {
        // thread 0 init the firing count and global offset
        if (threadIdx.x == 0) {
            block_firing_count = 0;
            block_global_offset = 0;
        }
        __syncthreads();

        const int index = block_begin + threadIdx.x;
        unsigned char did_fire = 0;
        if (index < neuron_count) {
            if (active_mask != nullptr && active_mask[index] == 0) {
                // if the neuron is not active
                current_did_fire[index] = 0;
            } else {
                int fired_field_id = -1;
                // get the neuron span index
                const int span_index = model_field_indices.span_index_by_neuron[index];
                if (span_index < 0 || span_index >= model_field_indices.span_count) {
                    current_did_fire[index] = 0;
                } else {
                    // get the neuron span field
                    const DenseModelFieldIndexSpan span = model_field_indices.spans[span_index];
                    const int* field_ids = model_field_indices.field_indices + span.field_index_offset;
                    // The unified kernel only provides a pending-channel view.
                    // Each model update owns the meaning and draining policy of
                    // its channels, so new models do not modify this main path.
                    DensePendingChannelDeviceView pending_view;
                    pending_view.pending_channels = pending_channels;
                    pending_view.channel_count = pending_channel_count;
                    pending_view.channel_stride = pending_channel_stride;
                    pending_view.neuron_index = index;
                    // update the neuron
                    const DenseDeviceModelUpdateFn update_fn =
                        ResolveDenseDeviceModelUpdate(span.factory_model_id);
                    if (update_fn != nullptr) {
                        did_fire = update_fn(
                            fields,
                            field_ids,
                            index,
                            current_time_step,
                            pending_view,
                            dt_ms,
                            &fired_field_id);
                    }
                    if (fired_field_id >= 0) {
                        // mark the fired neuron
                        DeviceByteField(fields, fired_field_id)[index] = did_fire;
                    }
                    current_did_fire[index] = did_fire;
                    if (did_fire != 0) {
                        if (output_neuron_mask != nullptr && output_neuron_mask[index] != 0) {
                            const int output_index = atomicAdd(output_firing_count, 1);
                            output_firing_ids[output_index] = index;
                        }
                        // Full firing ids are staged in shared memory first so
                        // the block performs one global atomic for the batch.
                        const int local_index = atomicAdd(&block_firing_count, 1);
                        if (local_index < kBlockFiringTableCapacity) {
                            block_firing_ids[local_index] = index;
                        }
                    }
                }
            }
        }
        __syncthreads();
        const int staged_count = min(block_firing_count, kBlockFiringTableCapacity);
        if (threadIdx.x == 0 && staged_count > 0) {
            block_global_offset = atomicAdd(full_firing_count, staged_count);
        }
        __syncthreads();
        // Copy the block-local firing batch into the global sparse list. The
        // list order is intentionally unspecified, matching the old atomic path.
        for (int local_index = threadIdx.x;
             local_index < staged_count;
             local_index += blockDim.x) {
            full_firing_ids[block_global_offset + local_index] = block_firing_ids[local_index];
        }
        __syncthreads();
    }
}

}  // namespace

bool DenseUnifiedNeuronRuntime::ConfigureOutputNeuronMask(const std::vector<unsigned char>& output_neuron_mask,
                                                          std::string* reason) {
    if (!output_neuron_mask.empty() &&
        static_cast<int>(output_neuron_mask.size()) != neuron_count_) {
        if (reason != nullptr) {
            *reason = "output_neuron_mask size must match neuron_count";
        }
        return false;
    }
    output_neuron_mask_.assign(static_cast<std::size_t>(neuron_count_), 0);
    for (std::size_t index = 0; index < output_neuron_mask.size(); ++index) {
        output_neuron_mask_[index] = output_neuron_mask[index] != 0 ? 1 : 0;
    }
    // copy the output_neuron_mask to device
    if (device_ready_ && d_output_neuron_mask_ != nullptr) {
        const std::size_t byte_bytes = sizeof(unsigned char) * static_cast<std::size_t>(neuron_count_);
        if (!CheckCudaStatus(cudaMemcpy(d_output_neuron_mask_,
                                        output_neuron_mask_.data(),
                                        byte_bytes,
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy(d_output_neuron_mask_)", reason)) {
            return false;
        }
    }
    return true;
}

bool DenseUnifiedNeuronRuntime::AllocateDeviceBuffers(std::string* reason) {
    DestroyDeviceBuffers();
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime is not initialized";
        }
        return false;
    }
    const std::size_t byte_bytes = sizeof(unsigned char) * static_cast<std::size_t>(neuron_count_);
    const std::size_t int_bytes = sizeof(int) * static_cast<std::size_t>(neuron_count_);
    const std::size_t field_span_bytes = sizeof(DenseDeviceFieldSpan) * host_field_spans_.size();
    const std::size_t float_pool_bytes = sizeof(float) * float_pool_size_;
    const std::size_t int_pool_bytes = sizeof(int) * int_pool_size_;
    const std::size_t byte_pool_bytes = sizeof(unsigned char) * byte_pool_size_;

    if (!CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_model_id_by_neuron_), int_bytes), "cudaMalloc(d_model_id_by_neuron_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_active_mask_), byte_bytes), "cudaMalloc(d_active_mask_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_output_neuron_mask_), byte_bytes), "cudaMalloc(d_output_neuron_mask_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_field_spans_), std::max<std::size_t>(1, field_span_bytes)), "cudaMalloc(d_field_spans_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_float_field_pool_), std::max<std::size_t>(1, float_pool_bytes)), "cudaMalloc(d_float_field_pool_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_int_field_pool_), std::max<std::size_t>(1, int_pool_bytes)), "cudaMalloc(d_int_field_pool_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_byte_field_pool_), std::max<std::size_t>(1, byte_pool_bytes)), "cudaMalloc(d_byte_field_pool_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_reset_float_field_pool_), std::max<std::size_t>(1, float_pool_bytes)), "cudaMalloc(d_reset_float_field_pool_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_reset_int_field_pool_), std::max<std::size_t>(1, int_pool_bytes)), "cudaMalloc(d_reset_int_field_pool_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_reset_byte_field_pool_), std::max<std::size_t>(1, byte_pool_bytes)), "cudaMalloc(d_reset_byte_field_pool_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_model_field_spans_), sizeof(DenseModelFieldIndexSpan) * std::max<std::size_t>(1, model_field_indices_.spans.size())), "cudaMalloc(d_model_field_spans_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_model_field_indices_), sizeof(int) * std::max<std::size_t>(1, model_field_indices_.field_indices.size())), "cudaMalloc(d_model_field_indices_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_model_field_span_by_neuron_), int_bytes), "cudaMalloc(d_model_field_span_by_neuron_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_current_firing_ids_), int_bytes), "cudaMalloc(d_current_firing_ids_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_current_firing_count_), sizeof(int)), "cudaMalloc(d_current_firing_count_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_current_did_fire_), byte_bytes), "cudaMalloc(d_current_did_fire_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_output_firing_ids_), int_bytes), "cudaMalloc(d_output_firing_ids_)", reason) ||
        !CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&d_output_firing_count_), sizeof(int)), "cudaMalloc(d_output_firing_count_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }

    device_fields_.field_spans = d_field_spans_;
    device_fields_.field_count = static_cast<int>(host_field_spans_.size());
    device_fields_.float_pool = d_float_field_pool_;
    device_fields_.int_pool = d_int_field_pool_;
    device_fields_.byte_pool = d_byte_field_pool_;
    device_model_field_indices_.spans = d_model_field_spans_;
    device_model_field_indices_.span_count = static_cast<int>(model_field_indices_.spans.size());
    device_model_field_indices_.field_indices = d_model_field_indices_;
    device_model_field_indices_.field_index_count = static_cast<int>(model_field_indices_.field_indices.size());
    device_model_field_indices_.span_index_by_neuron = d_model_field_span_by_neuron_;
    device_model_field_indices_.neuron_count = neuron_count_;

    device_firing_capacity_ = static_cast<std::size_t>(neuron_count_);
    device_output_firing_capacity_ = static_cast<std::size_t>(neuron_count_);
    device_ready_ = true;
    return true;
}

bool DenseUnifiedNeuronRuntime::UploadHostToDevice(std::string* reason) {
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime device buffers are not allocated";
        }
        return false;
    }
    const std::size_t byte_bytes = sizeof(unsigned char) * static_cast<std::size_t>(neuron_count_);
    const std::size_t int_bytes = sizeof(int) * static_cast<std::size_t>(neuron_count_);

    if (initial_host_fields_.fields.empty() && !host_field_spans_.empty()) {
        if (reason != nullptr) {
            *reason = "initial dense neuron host fields have already been released";
        }
        return false;
    }

    std::vector<DenseDeviceFieldSpan> device_spans(host_field_spans_.size());
    for (std::size_t index = 0; index < host_field_spans_.size(); ++index) {
        const DenseFieldSpan& host_span = host_field_spans_[index];
        DenseDeviceFieldSpan span;
        span.field_id = host_span.field_id;
        span.storage = host_span.storage;
        span.role = host_span.role;
        span.offset = host_span.offset;
        span.count = host_span.count;
        device_spans[index] = span;
    }

    if (!CheckCudaStatus(cudaMemcpy(d_model_id_by_neuron_, model_id_by_neuron_.data(), int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(d_model_id_by_neuron_)", reason) ||
        !CheckCudaStatus(cudaMemcpy(d_active_mask_, active_mask_.data(), byte_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(d_active_mask_)", reason) ||
        !CheckCudaStatus(cudaMemcpy(d_output_neuron_mask_, output_neuron_mask_.data(), byte_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(d_output_neuron_mask_)", reason) ||
        !CopyIfNonEmpty(d_field_spans_, device_spans.data(), sizeof(DenseDeviceFieldSpan) * device_spans.size(), cudaMemcpyHostToDevice, "cudaMemcpy(d_field_spans_)", reason) ||
        !CopyIfNonEmpty(d_float_field_pool_, initial_host_fields_.float_pool.data(), sizeof(float) * float_pool_size_, cudaMemcpyHostToDevice, "cudaMemcpy(d_float_field_pool_)", reason) ||
        !CopyIfNonEmpty(d_int_field_pool_, initial_host_fields_.int_pool.data(), sizeof(int) * int_pool_size_, cudaMemcpyHostToDevice, "cudaMemcpy(d_int_field_pool_)", reason) ||
        !CopyIfNonEmpty(d_byte_field_pool_, initial_host_fields_.byte_pool.data(), sizeof(unsigned char) * byte_pool_size_, cudaMemcpyHostToDevice, "cudaMemcpy(d_byte_field_pool_)", reason) ||
        !CopyIfNonEmpty(d_reset_float_field_pool_, initial_host_fields_.float_pool.data(), sizeof(float) * float_pool_size_, cudaMemcpyHostToDevice, "cudaMemcpy(d_reset_float_field_pool_)", reason) ||
        !CopyIfNonEmpty(d_reset_int_field_pool_, initial_host_fields_.int_pool.data(), sizeof(int) * int_pool_size_, cudaMemcpyHostToDevice, "cudaMemcpy(d_reset_int_field_pool_)", reason) ||
        !CopyIfNonEmpty(d_reset_byte_field_pool_, initial_host_fields_.byte_pool.data(), sizeof(unsigned char) * byte_pool_size_, cudaMemcpyHostToDevice, "cudaMemcpy(d_reset_byte_field_pool_)", reason) ||
        !CopyIfNonEmpty(d_model_field_spans_, model_field_indices_.spans.data(), sizeof(DenseModelFieldIndexSpan) * model_field_indices_.spans.size(), cudaMemcpyHostToDevice, "cudaMemcpy(d_model_field_spans_)", reason) ||
        !CopyIfNonEmpty(d_model_field_indices_, model_field_indices_.field_indices.data(), sizeof(int) * model_field_indices_.field_indices.size(), cudaMemcpyHostToDevice, "cudaMemcpy(d_model_field_indices_)", reason) ||
        !CheckCudaStatus(cudaMemcpy(d_model_field_span_by_neuron_, model_field_indices_.span_index_by_neuron.data(), int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(d_model_field_span_by_neuron_)", reason)) {
        return false;
    }
    this->ClearHostBuildCaches();
    host_state_dirty_ = false;
    return true;
}

bool DenseUnifiedNeuronRuntime::Step(const DeviceCommonBuffersView& common,
                                     int current_time_step,
                                     std::vector<int>* output_firing_ids,
                                     std::vector<int>* full_firing_ids,
                                     bool export_full_firings_to_host,
                                     std::string* reason) {
    if (output_firing_ids == nullptr) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime output_firing_ids output must not be null";
        }
        return false;
    }
    output_firing_ids->clear();
    if (full_firing_ids != nullptr) {
        full_firing_ids->clear();
    }
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime device buffers are not allocated";
        }
        return false;
    }
    const int zero = 0;
    if (!CheckCudaStatus(cudaMemcpy(d_current_firing_count_, &zero, sizeof(int), cudaMemcpyHostToDevice),
                         "cudaMemcpy(reset d_current_firing_count_)", reason) ||
        !CheckCudaStatus(cudaMemcpy(d_output_firing_count_, &zero, sizeof(int), cudaMemcpyHostToDevice),
                         "cudaMemcpy(reset d_output_firing_count_)", reason)) {
        return false;
    }
    const LaunchConfig1D launch = MakeLaunchConfig1D(neuron_count_, 256, kNeuronLaunchBlocksPerSm);
    const std::size_t shared_bytes = sizeof(int) * kBlockFiringTableCapacity;
    // Dynamic shared memory backs the block-local full firing table used by the
    // kernel to reduce global atomic contention.
    UpdateDenseUnifiedKernel<<<launch.blocks, launch.threads, shared_bytes>>>(
        d_active_mask_,
        d_output_neuron_mask_,
        device_fields_,
        device_model_field_indices_,
        dt_ms_,
        current_time_step,
        common.d_pending_channels,
        common.pending_channel_count,
        common.pending_channel_stride,
        neuron_count_,
        d_current_firing_ids_,
        d_current_firing_count_,
        d_current_did_fire_,
        d_output_firing_ids_,
        d_output_firing_count_);
    if (!CheckCudaStatus(cudaGetLastError(), "launch UpdateDenseUnifiedKernel", reason) ||
        !CheckCudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize(UpdateDenseUnifiedKernel)", reason)) {
        return false;
    }

    int output_firing_count = 0;
    if (!CheckCudaStatus(cudaMemcpy(&output_firing_count, d_output_firing_count_, sizeof(int), cudaMemcpyDeviceToHost),
                         "cudaMemcpy(d_output_firing_count_ -> host)", reason)) {
        return false;
    }
    output_firing_ids->resize(static_cast<std::size_t>(output_firing_count));
    if (output_firing_count > 0) {
        if (!CheckCudaStatus(cudaMemcpy(output_firing_ids->data(),
                                        d_output_firing_ids_,
                                        sizeof(int) * static_cast<std::size_t>(output_firing_count),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy(d_output_firing_ids_ -> host)", reason)) {
            return false;
        }
    }
    if (export_full_firings_to_host && full_firing_ids != nullptr) {
        int current_firing_count = 0;
        if (!CheckCudaStatus(cudaMemcpy(&current_firing_count, d_current_firing_count_, sizeof(int), cudaMemcpyDeviceToHost),
                             "cudaMemcpy(d_current_firing_count_ -> host)", reason)) {
            return false;
        }
        full_firing_ids->resize(static_cast<std::size_t>(current_firing_count));
        if (current_firing_count > 0) {
            if (!CheckCudaStatus(cudaMemcpy(full_firing_ids->data(),
                                            d_current_firing_ids_,
                                            sizeof(int) * static_cast<std::size_t>(current_firing_count),
                                            cudaMemcpyDeviceToHost),
                                 "cudaMemcpy(d_current_firing_ids_ -> host)", reason)) {
                return false;
            }
        }
    }
    host_state_dirty_ = true;
    return true;
}

bool DenseUnifiedNeuronRuntime::SyncHostStateFromDevice(std::string* reason) const {
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime device buffers are not allocated";
        }
        return false;
    }
    // Host field pools are intentionally not retained after initialization.
    // Debug callers should use BuildDebugSnapshots(), which downloads into a
    // temporary host table and releases it before returning.
    host_state_dirty_ = false;
    return true;
}

bool DenseUnifiedNeuronRuntime::DownloadFieldsToHost(DenseNeuronHostFieldTable* out,
                                                     std::string* reason) const {
    if (out == nullptr) {
        if (reason != nullptr) {
            *reason = "dense neuron debug field output must not be null";
        }
        return false;
    }
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime device buffers are not allocated";
        }
        return false;
    }
    out->fields = host_field_spans_;
    out->field_id_by_name.clear();
    for (std::size_t index = 0; index < out->fields.size(); ++index) {
        out->field_id_by_name[out->fields[index].name] = out->fields[index].field_id;
    }
    out->float_pool.assign(float_pool_size_, 0.0f);
    out->int_pool.assign(int_pool_size_, 0);
    out->byte_pool.assign(byte_pool_size_, 0);
    if (!CopyIfNonEmpty(out->float_pool.data(),
                        d_float_field_pool_,
                        sizeof(float) * float_pool_size_,
                        cudaMemcpyDeviceToHost,
                        "cudaMemcpy(d_float_field_pool_ -> debug host)",
                        reason) ||
        !CopyIfNonEmpty(out->int_pool.data(),
                        d_int_field_pool_,
                        sizeof(int) * int_pool_size_,
                        cudaMemcpyDeviceToHost,
                        "cudaMemcpy(d_int_field_pool_ -> debug host)",
                        reason) ||
        !CopyIfNonEmpty(out->byte_pool.data(),
                        d_byte_field_pool_,
                        sizeof(unsigned char) * byte_pool_size_,
                        cudaMemcpyDeviceToHost,
                        "cudaMemcpy(d_byte_field_pool_ -> debug host)",
                        reason)) {
        return false;
    }
    host_state_dirty_ = false;
    return true;
}

bool DenseUnifiedNeuronRuntime::ResetStateFieldsGpu(std::string* reason) {
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "DenseUnifiedNeuronRuntime device buffers are not allocated";
        }
        return false;
    }
    if (!CopyIfNonEmpty(d_float_field_pool_,
                        d_reset_float_field_pool_,
                        sizeof(float) * float_pool_size_,
                        cudaMemcpyDeviceToDevice,
                        "cudaMemcpyDeviceToDevice(reset float fields)",
                        reason) ||
        !CopyIfNonEmpty(d_int_field_pool_,
                        d_reset_int_field_pool_,
                        sizeof(int) * int_pool_size_,
                        cudaMemcpyDeviceToDevice,
                        "cudaMemcpyDeviceToDevice(reset int fields)",
                        reason) ||
        !CopyIfNonEmpty(d_byte_field_pool_,
                        d_reset_byte_field_pool_,
                        sizeof(unsigned char) * byte_pool_size_,
                        cudaMemcpyDeviceToDevice,
                        "cudaMemcpyDeviceToDevice(reset byte fields)",
                        reason)) {
        return false;
    }
    host_state_dirty_ = false;
    return true;
}

void DenseUnifiedNeuronRuntime::DestroyDeviceBuffers() {
    if (d_output_firing_count_ != nullptr) { cudaFree(d_output_firing_count_); d_output_firing_count_ = nullptr; }
    if (d_output_firing_ids_ != nullptr) { cudaFree(d_output_firing_ids_); d_output_firing_ids_ = nullptr; }
    if (d_current_did_fire_ != nullptr) { cudaFree(d_current_did_fire_); d_current_did_fire_ = nullptr; }
    if (d_current_firing_count_ != nullptr) { cudaFree(d_current_firing_count_); d_current_firing_count_ = nullptr; }
    if (d_current_firing_ids_ != nullptr) { cudaFree(d_current_firing_ids_); d_current_firing_ids_ = nullptr; }
    if (d_model_field_span_by_neuron_ != nullptr) { cudaFree(d_model_field_span_by_neuron_); d_model_field_span_by_neuron_ = nullptr; }
    if (d_model_field_indices_ != nullptr) { cudaFree(d_model_field_indices_); d_model_field_indices_ = nullptr; }
    if (d_model_field_spans_ != nullptr) { cudaFree(d_model_field_spans_); d_model_field_spans_ = nullptr; }
    if (d_reset_byte_field_pool_ != nullptr) { cudaFree(d_reset_byte_field_pool_); d_reset_byte_field_pool_ = nullptr; }
    if (d_reset_int_field_pool_ != nullptr) { cudaFree(d_reset_int_field_pool_); d_reset_int_field_pool_ = nullptr; }
    if (d_reset_float_field_pool_ != nullptr) { cudaFree(d_reset_float_field_pool_); d_reset_float_field_pool_ = nullptr; }
    if (d_byte_field_pool_ != nullptr) { cudaFree(d_byte_field_pool_); d_byte_field_pool_ = nullptr; }
    if (d_int_field_pool_ != nullptr) { cudaFree(d_int_field_pool_); d_int_field_pool_ = nullptr; }
    if (d_float_field_pool_ != nullptr) { cudaFree(d_float_field_pool_); d_float_field_pool_ = nullptr; }
    if (d_field_spans_ != nullptr) { cudaFree(d_field_spans_); d_field_spans_ = nullptr; }
    if (d_output_neuron_mask_ != nullptr) { cudaFree(d_output_neuron_mask_); d_output_neuron_mask_ = nullptr; }
    if (d_active_mask_ != nullptr) { cudaFree(d_active_mask_); d_active_mask_ = nullptr; }
    if (d_model_id_by_neuron_ != nullptr) { cudaFree(d_model_id_by_neuron_); d_model_id_by_neuron_ = nullptr; }
    device_fields_ = DenseNeuronDeviceFieldTable{};
    device_model_field_indices_ = DenseDeviceModelFieldIndexTable{};
    device_ready_ = false;
    device_firing_capacity_ = 0;
    device_output_firing_capacity_ = 0;
}

const int* DenseUnifiedNeuronRuntime::device_full_firing_ids() const {
    return d_current_firing_ids_;
}

const int* DenseUnifiedNeuronRuntime::device_full_firing_count() const {
    return d_current_firing_count_;
}

const unsigned char* DenseUnifiedNeuronRuntime::device_current_did_fire() const {
    return d_current_did_fire_;
}

}  // namespace npgr
