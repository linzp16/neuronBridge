#ifndef NPGR_INPUT_CONV_V1_CUDA_HELPERS_H
#define NPGR_INPUT_CONV_V1_CUDA_HELPERS_H

struct InputConvV1CudaConfig {
    int width = 0;
    int height = 0;
    int channels = 0;
    float speed = 1.5f;
};

bool CreateInputConvV1CudaRuntime(const InputConvV1CudaConfig& config,
                                  void** out_handle,
                                  int* out_output_count,
                                  const float** out_device_rates);
const float* GetInputConvV1CudaDeviceRates(void* handle);
int GetInputConvV1CudaOutputCount(void* handle);
void DestroyInputConvV1CudaRuntime(void* handle);
bool StepInputConvV1CudaRuntime(void* handle,
                                const unsigned char* host_stimulus,
                                float speed);

#endif
