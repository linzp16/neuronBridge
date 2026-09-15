#include "../source_file_realtime_v1_async/MotionEnergy/inc/motion_energy.h"

#include <cstring>
#include <stdexcept>

namespace {

[[noreturn]] void ThrowMotionEnergyUnavailable() {
    throw std::runtime_error(
        "MotionEnergy requires the GPU MotionEnergy implementation in this GPU-only build.");
}

}

MotionEnergy::MotionEnergy(int nrX, int nrY, int nrC)
    : nrX_(nrX),
      nrY_(nrY),
      nrC_(nrC),
      nrScales_(0),
      min_nrX_(0),
      min_nrY_(0),
      scaleV1Linear_(0.0),
      scaleV1FullWaveRect_(0.0),
      scaleV1Blur_(0.0),
      scaleV1NormPopK_(0.0),
      scaleV1NormStrength_(0.0),
      scaleV1Complex_(0.0),
      scaleV1C50_(0.0),
      scaleV1ComplexFiring_(0.0),
      stimChannels(0),
      scalingFilt(NULL),
      v1Gaus(NULL),
      complexV1Filt(NULL),
      normV1filt(NULL),
      diff1filt(NULL),
      diff2filt(NULL),
      diff3filt(NULL),
      d_resp_(NULL),
      d_respV1c(NULL),
      d_stimBuf(NULL),
      d_scalingStimBuf(NULL),
      d_v1GausBuf(NULL),
      d_v1GausBuf2(NULL),
      d_diffV1GausBuf(NULL),
      diffV1GausBufT(NULL),
      d_stim(NULL),
      d_pop(NULL) {
}

MotionEnergy::~MotionEnergy() {
}

void MotionEnergy::setNumChannels(int nrC) {
    this->nrC_ = nrC;
}

void MotionEnergy::calcV1complex(unsigned char* stim, float* V1comp, double speed, bool GPUpointers) {
    (void)stim;
    (void)V1comp;
    (void)speed;
    (void)GPUpointers;
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::calcV1filters(unsigned char* stim, float* V1filt, bool GPUpointers) {
    (void)stim;
    (void)V1filt;
    (void)GPUpointers;
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::getMotionProj(double speed, float* hostWeights) {
    (void)speed;
    if (hostWeights != NULL) {
        std::memset(hostWeights, 0, sizeof(float) * MotionEnergy::NumDirs * MotionEnergy::NumFilters);
    }
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::initME() {
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::initParams() {
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::accumDiffStims(float* d_resp_tmp, float* diffV1GausBuf, dim3 sizes, int orderX, int orderY, int orderT) {
    (void)d_resp_tmp;
    (void)diffV1GausBuf;
    (void)sizes;
    (void)orderX;
    (void)orderY;
    (void)orderT;
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::conv2D(float* idata, float* odata, dim3 sizes, const float* filt, int filtlen) {
    (void)idata;
    (void)odata;
    (void)sizes;
    (void)filt;
    (void)filtlen;
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::conv3D(float* idata, float* odata, dim3 sizes, const float* filt, int filtlen) {
    (void)idata;
    (void)odata;
    (void)sizes;
    (void)filt;
    (void)filtlen;
    ThrowMotionEnergyUnavailable();
}

float* MotionEnergy::diff(float* idata, dim3 sizes, int order, int dim) {
    (void)idata;
    (void)sizes;
    (void)order;
    (void)dim;
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::loadInput(unsigned char* stim) {
    (void)stim;
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::calcV1linear() {
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::calcV1rect() {
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::calcV1blur() {
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::calcV1normalize() {
    ThrowMotionEnergyUnavailable();
}

void MotionEnergy::calcV1direction(double speed) {
    (void)speed;
    ThrowMotionEnergyUnavailable();
}
