#include "input_conv/InputConvV1CudaHelpers.h"

#include <cuda_runtime.h>

#include "input_conv/Motion.cuh"

namespace {

struct InputConvV1CudaRuntime {
    Motion64* motion = 0;
    float* d_output_rates = 0;
    int output_count = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
};

}  // namespace

bool CreateInputConvV1CudaRuntime(const InputConvV1CudaConfig& config,
                                  void** out_handle,
                                  int* out_output_count,
                                  const float** out_device_rates) {
    if (out_handle == 0 || out_output_count == 0 || out_device_rates == 0) {
        return false;
    }
    *out_handle = 0;
    *out_output_count = 0;
    *out_device_rates = 0;

    if (config.width <= 0 || config.height <= 0 || config.channels <= 0) {
        return false;
    }

    InputConvV1CudaRuntime* runtime = new InputConvV1CudaRuntime();
    runtime->width = config.width;
    runtime->height = config.height;
    runtime->channels = config.channels;
    runtime->motion = new Motion64(config.width, config.height, config.channels);
    runtime->output_count = config.width * config.height * 8;
    if (cudaMalloc(reinterpret_cast<void**>(&runtime->d_output_rates),
                   sizeof(float) * static_cast<std::size_t>(runtime->output_count)) != cudaSuccess) {
        delete runtime->motion;
        runtime->motion = 0;
        delete runtime;
        return false;
    }
    cudaMemset(runtime->d_output_rates,
               0,
               sizeof(float) * static_cast<std::size_t>(runtime->output_count));
    *out_handle = runtime;
    *out_output_count = runtime->output_count;
    *out_device_rates = runtime->d_output_rates;
    return true;
}

const float* GetInputConvV1CudaDeviceRates(void* handle) {
    InputConvV1CudaRuntime* runtime = reinterpret_cast<InputConvV1CudaRuntime*>(handle);
    return runtime != 0 ? runtime->d_output_rates : 0;
}

int GetInputConvV1CudaOutputCount(void* handle) {
    InputConvV1CudaRuntime* runtime = reinterpret_cast<InputConvV1CudaRuntime*>(handle);
    return runtime != 0 ? runtime->output_count : 0;
}

void DestroyInputConvV1CudaRuntime(void* handle) {
    InputConvV1CudaRuntime* runtime = reinterpret_cast<InputConvV1CudaRuntime*>(handle);
    if (runtime == 0) {
        return;
    }
    if (runtime->d_output_rates != 0) {
        cudaFree(runtime->d_output_rates);
        runtime->d_output_rates = 0;
    }
    if (runtime->motion != 0) {
        delete runtime->motion;
        runtime->motion = 0;
    }
    delete runtime;
}

bool StepInputConvV1CudaRuntime(void* handle,
                                const unsigned char* host_stimulus,
                                float speed) {
    InputConvV1CudaRuntime* runtime = reinterpret_cast<InputConvV1CudaRuntime*>(handle);
    if (runtime == 0 || runtime->motion == 0 || runtime->d_output_rates == 0 || host_stimulus == 0) {
        return false;
    }
    runtime->motion->calcV1complex(const_cast<unsigned char*>(host_stimulus),
                                   runtime->d_output_rates,
                                   static_cast<double>(speed),
                                   true);
    return cudaDeviceSynchronize() == cudaSuccess;
}
