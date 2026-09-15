#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

namespace {

__global__ void EvaluateCudaMath(float* output) {
    const float x = 0.25f;
    const float y = 1.5f;
    output[0] = expf(x) + expm1f(x) + logf(y) + log1pf(x) + sqrtf(y) +
        fabsf(-x) + fminf(x, y) + fmaxf(x, y) + powf(y, 2.0f);
}

bool CheckCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

}  // namespace

int main() {
    const float x = 0.25f;
    const float y = 1.5f;
    const float cpu = std::exp(x) + std::expm1(x) + std::log(y) +
        std::log1p(x) + std::sqrt(y) + std::fabs(-x) + std::fmin(x, y) +
        std::fmax(x, y) + std::pow(y, 2.0f);

    float* device = nullptr;
    if (!CheckCuda(cudaMalloc(&device, sizeof(float)), "cudaMalloc")) return EXIT_FAILURE;
    EvaluateCudaMath<<<1, 1>>>(device);
    if (!CheckCuda(cudaGetLastError(), "EvaluateCudaMath launch") ||
        !CheckCuda(cudaDeviceSynchronize(), "EvaluateCudaMath synchronize")) {
        cudaFree(device);
        return EXIT_FAILURE;
    }
    float gpu = 0.0f;
    const bool copied = CheckCuda(
        cudaMemcpy(&gpu, device, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(device);
    if (!copied || !std::isfinite(cpu) || !std::isfinite(gpu) ||
        std::fabs(cpu - gpu) > 1.0e-4f) {
        std::cerr << "math function CPU/CUDA mismatch: cpu=" << cpu << " gpu=" << gpu << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "math_functions=PASS cpu=" << cpu << " gpu=" << gpu << '\n';
    return EXIT_SUCCESS;
}
