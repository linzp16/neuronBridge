#ifndef NPGR_DENSE_INTEGRATION_KERNELS_CUH
#define NPGR_DENSE_INTEGRATION_KERNELS_CUH

namespace npgr {

// Dense integration helpers are compile-time CUDA utilities, not runtime
// integration models. Neuron update functions call them directly so the unified
// dense kernel keeps the same flat-memory, switch-dispatch execution path.

__device__ inline float DenseDecayAndAdd(float previous_value,
                                         float decay_factor,
                                         float arrival_value) {
    return previous_value * decay_factor + arrival_value;
}

__device__ inline float DenseConductanceCurrent(float conductance,
                                                float reversal_potential,
                                                float membrane_v) {
    return conductance * (reversal_potential - membrane_v);
}

__device__ inline float DenseNmdaVoltageGate(float membrane_v) {
    return 1.0f / (1.0f + expf(-0.062f * membrane_v) * (1.2f / 3.57f));
}

__device__ inline float DenseLifEulerStep(float membrane_v,
                                          float alpha,
                                          float drive) {
    return membrane_v + alpha * drive;
}

__device__ inline unsigned char DenseThresholdResetStep(float* membrane_v,
                                                        int* steps_since_last_spike,
                                                        float threshold,
                                                        float reset_value) {
    unsigned char did_fire = 0;
    *steps_since_last_spike += 1;
    if (*membrane_v >= threshold) {
        did_fire = 1;
        *membrane_v = reset_value;
        *steps_since_last_spike = 0;
    }
    return did_fire;
}

__device__ inline float DenseHashUniform01(int neuron_index, int time_step) {
    unsigned int x = static_cast<unsigned int>(neuron_index) * 747796405u +
                     static_cast<unsigned int>(time_step + 1) * 2891336453u +
                     0x9e3779b9u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

__device__ inline float DensePoissonFireProbability(float rate_hz,
                                                    float dt_ms) {
    return fminf(1.0f, rate_hz * dt_ms * 0.001f);
}

__device__ inline void DenseIzhikevichEulerStep(float* membrane_v,
                                                float* recovery_u,
                                                float a,
                                                float b,
                                                float r,
                                                float total_current,
                                                float dt_ms) {
    const float old_v = *membrane_v;
    *membrane_v += dt_ms *
                   (0.04f * old_v * old_v + 5.0f * old_v + 140.0f -
                    *recovery_u + r * total_current);
    *recovery_u += dt_ms * (a * (b * *membrane_v - *recovery_u));
}

}  // namespace npgr

#endif
