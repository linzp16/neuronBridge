#ifndef MOTION_CUH
#define MOTION_CUH

#include <vector_types.h>

class Motion64 {
    public:
        /*
        * nrX - number of columns
        * nrY - number of rows
        * nrC - number of channels
        */
        Motion64(int nrX, int nrY, int nrC);

        ~Motion64();

        void setNumChannels(int nrC);

        void calcV1complex(unsigned char* stim, float* V1comp, double speed = 1.5, bool outputOnGPU = true);
        void calcV1complexReferenceBuffer(unsigned char* stim, float* V1comp, double speed = 1.5, bool outputOnGPU = true);

        double getScaleV1Linear() const { return scaleV1Linear_; }
        double getScaleV1FullWaveRect() const { return scaleV1FullWaveRect_; }
        double getScaleV1Blur() const { return scaleV1Blur_; }
        double getScaleV1NormPopK() const { return scaleV1NormPopK_; }
        double getScaleV1NormStrength() const { return scaleV1NormStrength_; }
        double getScaleV1Complex() const { return scaleV1Complex_; }
        double getScaleV1ComplexFiring() const { return scaleV1ComplexFiring_; }
        double getScaleV1C50() const { return scaleV1C50_; }

        void setScaleV1Linear(double s) { scaleV1Linear_ = s; }
        void setScaleV1FullWaveRect(double s) { scaleV1FullWaveRect_ = s; }
        void setScaleV1Blur(double s) { scaleV1Blur_ = s; }
        void setScaleV1NormPopK(double s) { scaleV1NormPopK_ = s; }
        void setScaleV1NormStrength(double s) { scaleV1NormStrength_ = s; }
        void setScaleV1Complex(double s) { scaleV1Complex_ = s; }
        void setScaleV1ComplexFiring(double s) { scaleV1ComplexFiring_ = s; }
        void setScaleV1C50(double s) { scaleV1C50_ = s; }


    private:
        // number of columns
        int nrX;
        // number of rows
        int nrY;
        // number of chanels
        int nrC;
        // number of scales
        int nrScales;
        // number of min columns
        int min_nrX;
        // number of min rows
        int min_nrY;

        double scaleV1Linear_;
        double scaleV1FullWaveRect_;
        double scaleV1Blur_;
        double scaleV1NormPopK_;
        double scaleV1NormStrength_;
        double scaleV1Complex_;
        double scaleV1C50_;
        double scaleV1ComplexFiring_;

        int stimChannels;
        // Host copies of the convolution filters used to seed constant memory.
        float* scalingFilt;
        float* v1Gaus;
        float* complexV1Filt;
        float* normV1filt;
        float* diff1filt;
        float* diff2filt;
        float* diff3filt;
        // Device buffers used by the V1 motion-energy pipeline.
        float* d_resp_;
        float* d_respV1c;
        float* d_stimBuf;
        float* d_scalingStimBuf;
        float* d_v1GausBuf;
        float* d_v1GausBuf2;
        float* d_diffV1GausBuf;
        float* diffV1GausBufT;
        unsigned char* d_stim;
        float* d_pop;

        void initME();

        void initParams();

        void accumDiffStims(float* d_res_tmp, float* diffV1GausBuf, dim3 sizes, int orderX, int orderY, int orderT);

        void conv2D(float* idata, float* odata, dim3 sizes,
            const float* filt, int filtlen);
        void conv3D(float* idata, float* odata, dim3 sizes,
            const float* filt, int filtlen);
        float* diff(float* idata, dim3 sizes, int order, int dim);

        void loadInput(unsigned char* stim);
        void loadInputReferenceBuffer(unsigned char* stim);
        void resetResponseBuffers();
        void calcV1linear();
        void calcV1rect();
        void calcV1blur();
        void calcV1normalize();
        void calcV1direction(double speed);

};

#endif
