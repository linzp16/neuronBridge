#define MOTION_CLASS_NAME Motion64
#define MOTION_FILTERBANK_HEADER "input_conv/MotionFilterBank64.h"

#include "input_conv/Motion.cuh"

#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm> 

#define iDivUp(a,b) ((a) + (b) - 1) / (b)
#define CONV1_THREAD_SIZE 256
#define CONVN_THREAD_SIZE1 16
#define CONVN_THREAD_SIZE2 31
#include MOTION_FILTERBANK_HEADER
// define the size 5 conv kernel
#define scalingFiltSize 5
__constant__ float d_scalingFilt[scalingFiltSize] = { 0.0884, 0.3536, 0.5303, 0.3536, 0.0884 };
// define the size 9 conv kernel
#define v1GaussFiltSize 9
__constant__ float d_v1GaussFilt[v1GaussFiltSize] = { 0.0007, 0.0155, 0.0903, 0.2345, 0.3179, 0.2345, 0.0903, 0.0155, 0.0007 };
// define the size 11 conv kernel
#define complexV1FiltSize 11
__constant__ float d_complexV1Filt[complexV1FiltSize] = { 0.0019, 0.0110, 0.0430, 0.1142, 0.2052, 0.2495, 0.2052, 0.1142, 0.0430, 0.0110, 0.0019 };
// define the size 25 conv kernel
#define normV1filtSize 25
__constant__ float d_normV1filt[normV1filtSize] = { 0.0045,0.0072,0.0109,0.0160,0.0225,0.0303,0.0393,0.0490,0.0587,0.0675,0.0746,0.0792,0.0808,0.0792,0.0746,0.0675,0.0587,0.0490,0.0393,0.0303,0.0225,0.0160,0.0109,0.0072,0.0045 };
// define the first order difference filter
#define diff1FiltSize 3
__constant__ float d_diff1filt[diff1FiltSize] = { -1 / 2.0, 0, 1 / 2.0 };
// define the second order difference filter
#define diff2FiltSize 3
__constant__ float d_diff2filt[diff2FiltSize] = { 1, -2, 1 };
// define the third order difference filter
#define diff3FiltSize 5
__constant__ float d_diff3filt[diff3FiltSize] = { -1 / 2.0, 1, 0, -1, 1 / 2.0 };

/*
* accumulate to the 28 directions
* @param d_resp_tmp
 *        
 *
 * @param diffV1GausBuf
 *        
 *
 * @param nrXnrY
 *        
 *
 * @param scale
 *        
 *
 * @param orderX
 *        X 
 *
 * @param orderY
 *        Y 
 *
 * @param orderT
 *        
*/

__global__ void dev_accumDiffStims(float* d_resp_tmp, float* diffV1GausBud, int nrXnrY, int scale, int orderX, int orderY, int orderT) {
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	// define the sharing memory to store the temp result
	__shared__ float dirorders[nrFilters];

	for (int filter = threadIdx.x; filter < nrFilters; filter += threadN) {
		const float dirXComponent = d_v1popDirs[0][filter];
		const float dirYComponent = d_v1popDirs[1][filter];
		const float dirTComponent = d_v1popDirs[2][filter];

		float dirXPower = 1.0f;
		for (int n = 0; n < orderX; n++) {
			dirXPower *= dirXComponent;
		}

		float dirYPower = 1.0f;
		for (int n = 0; n < orderY; n++) {
			dirYPower *= dirYComponent;
		}

		float dirTPower = 1.0f;
		for (int n = 0; n < orderT; n++) {
			dirTPower *= dirTComponent;
		}
		dirorders[filter] = dirXPower * dirYPower * dirTPower;

	}

	__syncthreads();

	for (int pixel = tid; pixel < nrXnrY; pixel += threadN) {
		const float derivateResponse = diffV1GausBud[pixel];

		const float scaleResponse = scale * derivateResponse;

		for (int filter = 0; filter < nrFilters; filter++) {
			const int outputIndex = filter * nrXnrY + pixel;

			d_resp_tmp[outputIndex] += scaleResponse * dirorders[filter];
		}
	}


}

/*
* caculate the average of two array per pixel
* @param i1data 
* @param i2data 
* @param odata 
* @param len    
*/
__global__ void dev_average2(
	const float* __restrict__ i1data,
	const float* __restrict__ i2data,
	float* __restrict__ odata,
	int len)
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const int threadCount = static_cast<int>(blockDim.x * gridDim.x);
	
	for (int i = tid; i < len; i += threadCount) {
		odata[i] = 0.5f * (i1data[i] + i2data[i]);
	}
}

/*
* making 1D conv caculation to a array
* * @param idata
 *        
 *
 * @param odata
 *        
 *        
 *
 * @param len
 *        
 *
 * @param filt
 *        
 *
 * @param filtlen
 *        
 *        1. filtlen > 0
 *        2. filtlen is odd
*/
__global__ void dev_conv1D(const float* __restrict__ idata,
	float* __restrict__ odata,
	int len,
	const float* __restrict__ filt,
	int filtlen) {
	// define the shared memory
	__shared__ float sharedData[CONV1_THREAD_SIZE];

	const int radius = (filtlen - 1) / 2;
	// valid convolution length 
	const int nrValidConv = CONV1_THREAD_SIZE - filtlen + 1;
	// each block deal with the conv of nrValidConv centered pixel
	const int inputX = static_cast<int>(blockIdx.x) * nrValidConv + static_cast<int>(threadIdx.x) - radius;
	// the row of the input
	const int row = static_cast<int>(blockIdx.y);

	if (inputX >= 0 && inputX < len) {
		sharedData[threadIdx.x] = idata[row * len + inputX];
	}
	else {
		sharedData[threadIdx.x] = 0;
	}

	__syncthreads();

	// only the first nrValidConv thread of each block generate output
	if (threadIdx.x >= nrValidConv) {
		return;
	}

	const int outputX = static_cast<int>(blockIdx.x) * nrValidConv + static_cast<int>(threadIdx.x);
	if (outputX >= len) {
		return;
	}

	float sum = 0;

	for (int tap = 0; tap < filtlen; tap++) {
		sum += sharedData[threadIdx.x + tap] * filt[tap];
	}

    odata[row * len + outputX] = sum;

}

/*
* making conv along y dim or z dim
*  * @param idata
 *        
 *
 * @param odata
 *       
 *        
 *
 * @param nrX
 *        
 *
 * @param nrN
 *        
 *
 * @param stride
 *        
 *
 * @param blockStride
 *        
 *
 * @param nrBlocks
 *        
 *
 * @param filt
 *       
 *
 * @param filtlen
 *        
*/
__global__ void dev_convn(
	const float* __restrict__ idata,
	float* __restrict__ odata,
	int nrX,
	int nrN,
	int stride,
	int blockStride,
	int nrBlocks,
	const float* __restrict__ filt,
	int filtlen) 
{
	__shared__ float shareData[CONVN_THREAD_SIZE1 * CONVN_THREAD_SIZE2];

	const int radius = (filtlen - 1) / 2;

	const int validOutputsPerBlock = CONVN_THREAD_SIZE2 - filtlen + 1;

	const int segmentIndex = static_cast<int>(blockIdx.y) / nrBlocks;
	
	const int dataBlockIndex = static_cast<int>(blockIdx.y) % nrBlocks;

	const int ind1 = static_cast<int>(blockIdx.x) * CONVN_THREAD_SIZE1 + static_cast<int>(threadIdx.x);

	const int inputInd2 = segmentIndex * validOutputsPerBlock + static_cast<int>(threadIdx.y) - radius;

	const int sharedIndex = static_cast<int>(threadIdx.x) * CONVN_THREAD_SIZE2 + static_cast<int>(threadIdx.y);

	if (ind1 < nrX && inputInd2 >= 0 && inputInd2 < nrN) {
		const int inputIndex = dataBlockIndex * blockStride + inputInd2 * stride + ind1;
		shareData[sharedIndex] = idata[inputIndex];
	}
	else {
		shareData[sharedIndex] = 0.0f;
	}

	__syncthreads();

	if (threadIdx.y >= validOutputsPerBlock) {
		return;
	}

	const int outputInd2 = segmentIndex * validOutputsPerBlock + static_cast<int>(threadIdx.y);

	if (ind1 >= nrX || outputInd2 >= nrN) {
		return;
	}

	float sum = 0;

	for (int tap = 0; tap < filtlen; tap++) {
		sum += shareData[sharedIndex + tap] * filt[tap];
	}

	const int outputIndex = dataBlockIndex * blockStride + outputInd2 * stride + ind1;

	odata[outputIndex] = sum;

}

/*
* Because convolution uses zero padding, positions near the image boundary will produce noticeable * artificial responses. This kernel scales the responses based on the distance of a pixel to the nearest image boundary.
*@param data
 *        
 *
 * @param len
 *        
 *
 * @param nrX
 *       
 *
 * @param nrY
 *       
*/
__global__ void dev_edges(
	float* data,
	int len,
	int nrX,
	int nrY)
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadCount = static_cast<int>(blockDim.x * gridDim.x);

	const int pixelsPerFilter = nrX * nrY;

	const int elementsPerScale = pixelsPerFilter * nrFilters;

	for (int index = tid; index < len; index += threadCount) {
		const int x = index % nrX;
		const int y = (index / nrX) % nrY;

		const int scale = index / elementsPerScale;

		const int distanceLeft = x;
		const int distanceRight = nrX - 1 - x;
		const int distanceTop = y;
		const int distanceBottom = nrY - 1 - y;

		const int nearestHorizintalEdge = min(distanceLeft, distanceRight);
		const int nearestVerticalEdge = min(distanceTop, distanceBottom);

		const float edgeDistance = static_cast<float>(min(nearestHorizintalEdge, nearestVerticalEdge));

		const float d2 = edgeDistance * edgeDistance;
		const float d3 = edgeDistance * d2;
		const float d4 = d2 * d2;
		const float d5 = d3 * d2;

		float edgeFactor = 1.0f;

		if (scale == 0) {
			edgeFactor = fminf(d3, 125.0f) / 125.0f;
		}
		else if (scale == 1) {
			edgeFactor = fminf(d4, 1296.0f) / 1296.0f;
		}
		else if (scale == 2) {
			edgeFactor = fminf(d5, 7776.0f) / 7776.0f;
		}

		data[index] *= edgeFactor;

	}
}

/*
* this kernel project the output of the spatial temprol output to the direction
* 	/*
	* this kernel project the output of the spatial temprol output to the direction
	*  *
 *
 *     d_respIn[scale][filter][position]
 *
 *
 *     inputIndex =
 *         scale  * nrFilters * nrXnrY
 *       + filter * nrXnrY
 *       + position
 *
 *
 *     d_respOut[direction][position]
 *
 *
 *     outputIndex =
 *         direction * nrXnrY + position
 *
*/

__global__ void devfilt2dir(const float* __restrict__ d_respIn,
	float* __restrict__ d_respOut,
	int nrXnrY,
	int nrScales,
	int nrDirections,
	int speedIndex)
{
	const int outputlength = nrDirections * nrXnrY;

	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadCount = static_cast<int>(blockDim.x * gridDim.x);

	for (int outputIndex = tid; outputIndex < outputlength; outputIndex += threadCount) {
		const int position = outputIndex % nrXnrY;

		const int direction = outputIndex / nrXnrY;

		float sum = 0.0f;

		for (int scale = 0; scale < nrScales; ++scale) {
			const int scaleOffset = scale * nrFilters * nrXnrY;

			for (int filter = 0; filter < nrFilters; ++filter) {
				const int inputIndex = scaleOffset + filter * nrXnrY + position;

				sum += d_respIn[inputIndex] * motionProj[speedIndex][filter][direction];
			}
		}

		d_respOut[outputIndex] = sum;
	}
}

/*
* this function does the full-wave rectification to the output
 *
 *     linearResponse =
 *         data[i] * scaleV1Linear
 *
 *     data[i] =
 *         linearResponse^2 * scaleV1FullWaveRect
 *
 *
 * @param data
 *        
 *
 * @param len
 *        
 *
 * @param scaleV1Linear
 *        
 *
 * @param scaleV1FullWaveRect
 *        
*/
__global__ void dev_fullRect2(
	float* data,
	int len,
	float scaleV1Linear,
	float scaleV1FullWaveRect) 
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	for (int i = tid; i < len; i += threadN) {

		const float linearResponse = data[i] * scaleV1Linear;

		data[i] = linearResponse * linearResponse * scaleV1FullWaveRect;

	}
}

/*
* this kernel take the average of the responses of nrZ filters at the same spatial location and scale
 *
 *     idata[scale][filter][position]
 *

 *
 *     inputIndex =
 *         scale  * nrFilters * nrXnrY
 *       + filter * nrXnrY
 *       + position
 *
 *
 *     odata[scale][position]
 *
 *
 *     outputIndex =
 *         scale * nrXnrY + position
 *
 * blockIdx.y 
 *
 * @param idata
 *        
 *
 * @param odata
 *       
 *
 * @param nrXnrY
 *        
 *
 * @param nrZ
 *        
*/
__global__ void dev_mean3(
	const float* __restrict__ idata,
	float* __restrict__ odata,
	int nrXnrY,
	int nrZ)
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	const int scale = static_cast<int>(blockIdx.y);

	const int elementPerScale = nrXnrY * nrZ;

	const int scaleInputOffset = scale * elementPerScale;

	const int scaleOutputOffset = nrXnrY * scale;

	for (int position = tid; position < nrXnrY; position += threadN) {
		float sum = 0.0;

		for (int filter = 0; filter < nrZ; ++filter) {
			const int inputIndex = scaleInputOffset + filter * nrXnrY + position;

			sum += idata[inputIndex];
		}

		odata[scaleOutputOffset + position] = sum / nrZ;
	}
}

/*
* this kernel is used for population division normalization of complex cellular responses
 *
 *     resp[scale][filter][position]
 *
 *
 *     pop[scale][position]
 *
 *
 *     denominator =
 *         pop[scale][position] + scaleV1C50^2
 *
 * @param resp
 *        
 *
 * @param pop
 *        
 *
 * @param nrXnrY
 *        
 *
 * @param scaleV1C50
 *        
*/
__global__ void dev_normalize(
	float* __restrict__ resp,
	const float* __restrict__ pop,
	int nrXnrY,
	float scaleV1C50)
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	const int scale = static_cast<int>(blockIdx.y);

	const int elementPerScale = nrXnrY * nrFilters;

	const int population_scale_offset = scale * nrXnrY;

	const float c50Squared = scaleV1C50 * scaleV1C50;

	for (int position = tid; position < nrXnrY; position += threadN) {
		const float populationResponse = pop[population_scale_offset + position];

		const float denominator = populationResponse + c50Squared;

		for (int filter = 0; filter < nrFilters; ++filter) {
			const int respIndex = scale * elementPerScale + filter * nrXnrY + position;

			resp[respIndex] = resp[respIndex] / denominator;
		}
	}
}


/*
* this kernel multiply the factor to the input array
*  * 
 *
 *     data[i] = data[i] * scale
 *
 *
 * @param data
 *        
 *
 * @param scale
 *        
 *
 * @param len
 *        
*/

__global__ void dev_scale(
	float* data,
	float scale,
	int len)
{
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadCount = static_cast<int>(blockDim.x * gridDim.x);

	for (int index = tid; index < len; index += threadCount) {
		data[index] *= scale;
	}
}


/*
* this kernel does the half-wave rectification + scaling + spontaneous activity lower limit replacement.
* 
 *
 *     
 *
 *         data[i] = data[i] * scale
 *
 *   
 *
 *         data[i] = spontaneousActivity
 *

 *
 *
 * @param data

 *
 * @param len

 *
 * @param scale
 *        
 *
 * @param spontaneousActivity
 *        
*/
__global__ void dev_scaleHalfRect(
	float* data,
	int len,
	float scale,
	float spontaneousActivity)
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	for (int index = tid; index < len; index += threadN) {
		const float response = data[index];

		data[index] = response > 0.0f ? response * scale : spontaneousActivity;


	}
}

/*
* this function does the Grayscale image normalization
* 
 *
 *     gray[i] = idata[i] / 255.0
 *
 *
 *     gray[i] = idata[i] * (1.0 / 255.0)
 *
 *
 *     idata[pixel]
 *
 *
 *     gray[pixel]
 *
 * @param idata
 *
 * @param gray
 *
 * @param len
*/
__global__ void dev_split_gray(
	const unsigned char* __restrict__ idata,
	float* __restrict__ gray,
	int len)
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	constexpr float inv255 = 1.0f / 255.0f;

	for (int index = tid; index < len; index += threadN) {
		gray[index] = static_cast<float>(idata[index]) * inv255;
	}
}

/*
* this kernel does the RGB image normalization
 *
 *     idata =
 *     [R0, G0, B0,
 *      R1, G1, B1,
 *      R2, G2, B2,
 *      ...]
 *
 *
 *     gray[i] =
 *         (R + G + B) / (3 * 255)
 *
 *
 * @param idata
 *        
 *
 * @param gray
 *        
 *
 * @param len
 *        
* 
*/
__global__ void dev_split_RGB(
	const unsigned char* __restrict__ idata,
	float* __restrict__ gray,
	int len)
{
	const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	constexpr float invThree255 = 1.0f / (3.0f * 255.0f);

	for (int pixel = tid; pixel < len; pixel += threadN) {
		const int base = 3 * pixel;

		const float r = static_cast<float>(idata[base]);

		const float g = static_cast<float>(idata[base + 1]);

		const float b = static_cast<float>(idata[base + 2]);

		gray[pixel] = (r + g + b) * invThree255;
	}
}

/*
* this kernel does the subtraction 
*  * @param input1
 *        
 *
 * @param input2
 *        
 *
 * @param output
 *        
 *
 * @param length
 *        
*/
__global__ void dev_sub(
	const float* input1,
	const float* input2,
	float* output,
	int length)
{
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	for (int index = tid; index < length; index += threadN) {
		output[index] = input1[index] - input2[index];
	}
}

/*
 * This kernel performs element-wise addition:
 *
 *     output[i] = input1[i] + input2[i]
 *
 * All buffers must contain at least length elements:
 *
 *     input1[0 ... length-1]
 *     input2[0 ... length-1]
 *     output[0 ... length-1]
 *
 * @param input1
 *        
 *
 * @param input2
 *        
 *
 * @param output
 *       
 *
 * @param length
 *       
*/
__global__ void dev_sum(
	const float* input1,
	const float* input2,
	float* output,
	int length)
{
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    const int threadN = static_cast<int>(blockDim.x * gridDim.x);

	for (int index = tid; index < length; index += threadN) {
        output[index] = input1[index] + input2[index];
	}
}


MOTION_CLASS_NAME::MOTION_CLASS_NAME(int _nrX, int _nrY, int _nrC) {
	nrX = _nrX;
	nrY = _nrY;
    nrC = _nrC;
	assert(nrC == 1 || nrC == 3);

	min_nrX = 19;
	min_nrY = 19;

	nrScales = 3;

	initParams();

	initME();

}

#define nrT 9
void MOTION_CLASS_NAME::initME() {

	const std::size_t pixelCount = static_cast<std::size_t>(nrX * nrY);

	const std::size_t stimulusByteCount = pixelCount * nrC * sizeof(unsigned char);

	const std::size_t temporalVolumeElements = pixelCount * static_cast<std::size_t>(nrT);

	const std::size_t responseElements = pixelCount * static_cast<std::size_t>(nrFilters) * static_cast<std::size_t>(nrScales);

	const std::size_t directionResponseElements = pixelCount * static_cast<std::size_t>(nrDirs);

	const std::size_t populationElements = pixelCount * static_cast<std::size_t>(nrScales);


	cudaMalloc(reinterpret_cast<void**>(&d_resp_), responseElements * sizeof(float));


	cudaMalloc(reinterpret_cast<void**>(&d_respV1c), directionResponseElements * sizeof(float));


	cudaMalloc(reinterpret_cast<void**>(&d_stim), stimulusByteCount);


	cudaMalloc(reinterpret_cast<void**>(&d_stimBuf), temporalVolumeElements * sizeof(float));
	cudaMemset(d_stimBuf, 0, temporalVolumeElements * sizeof(float));


	cudaMalloc(reinterpret_cast<void**>(&diffV1GausBufT), pixelCount * static_cast<std::size_t>(v1GaussFiltSize) * sizeof(float));


	cudaMalloc(reinterpret_cast<void**>(&d_scalingStimBuf), temporalVolumeElements * sizeof(float));

	cudaMalloc(reinterpret_cast<void**>(&d_v1GausBuf), temporalVolumeElements * sizeof(float));


	cudaMalloc(reinterpret_cast<void**>(&d_diffV1GausBuf), temporalVolumeElements * sizeof(float));


	cudaMalloc(reinterpret_cast<void**>(&d_pop), populationElements * sizeof(float));


	cudaGetSymbolAddress(
			reinterpret_cast<void**>(&scalingFilt),
			d_scalingFilt);

	cudaGetSymbolAddress(
			reinterpret_cast<void**>(&v1Gaus),
			d_v1GaussFilt);

	
	cudaGetSymbolAddress(
			reinterpret_cast<void**>(&complexV1Filt),
			d_complexV1Filt);

	
	cudaGetSymbolAddress(
			reinterpret_cast<void**>(&normV1filt),
			d_normV1filt);

	
	cudaGetSymbolAddress(
			reinterpret_cast<void**>(&diff1filt),
			d_diff1filt);

	
	cudaGetSymbolAddress(
			reinterpret_cast<void**>(&diff2filt),
			d_diff2filt);

	
	cudaGetSymbolAddress(
			reinterpret_cast<void**>(&diff3filt),
			d_diff3filt);


}

MOTION_CLASS_NAME::~MOTION_CLASS_NAME() {
	cudaFree(d_stimBuf);
	cudaFree(diffV1GausBufT);
	cudaFree(d_stim);
	cudaFree(d_scalingStimBuf);
	cudaFree(d_v1GausBuf);
	cudaFree(d_diffV1GausBuf);
	cudaFree(d_pop);
	cudaFree(d_resp_);
	cudaFree(d_respV1c);
}

void MOTION_CLASS_NAME::initParams() {
	scaleV1Linear_ = 6.6084;
	scaleV1FullWaveRect_ = 1.9263;
	scaleV1Blur_ = 1.0205;
	scaleV1NormPopK_ = 1.0;
	scaleV1NormStrength_ = 0.98;
	scaleV1Complex_ = 0.99;
	scaleV1C50_ = 0.1;
	scaleV1ComplexFiring_ = 10.0;
}

/*
* set the number of channnels
*/
void MOTION_CLASS_NAME::setNumChannels(int nrC) {

	assert(nrC == 1 || nrC == 3);

	if (this->nrC != nrC) {
		this->nrC = nrC;
		cudaFree(d_stim);
		cudaMalloc(reinterpret_cast<void**>(&d_stim), static_cast<std::size_t>(nrX) * nrY * nrC * sizeof(unsigned char));
	}
}

void MOTION_CLASS_NAME::calcV1complex(unsigned char* stim, float* V1comp, double speed, bool GPUpointers) {
	assert(speed == 1.5 || speed == 0.125 || speed == 9.0);
	loadInput(stim);
	calcV1linear();
	calcV1rect();
	calcV1blur();
	calcV1normalize();
	calcV1direction(speed);
	cudaMemcpy(V1comp, d_respV1c, sizeof(float) * nrX * nrY * nrDirs, GPUpointers ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);
}

void MOTION_CLASS_NAME::calcV1complexReferenceBuffer(unsigned char* stim, float* V1comp, double speed, bool GPUpointers) {
	assert(speed == 1.5 || speed == 0.125 || speed == 9.0);
	loadInputReferenceBuffer(stim);
	calcV1linear();
	calcV1rect();
	calcV1blur();
	calcV1normalize();
	calcV1direction(speed);
cudaMemcpy(V1comp, d_respV1c, sizeof(float) * nrX * nrY * nrDirs, GPUpointers ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);
}

void MOTION_CLASS_NAME::loadInput(unsigned char* stim) {

	const std::size_t pixelCount = static_cast<std::size_t>(nrX * nrY);
	const std::size_t frameBytes = pixelCount * sizeof(float);

	cudaMemcpy(d_stim, stim, nrC * nrX * nrY, cudaMemcpyHostToDevice);

	for (int timeIndex = 1; timeIndex < nrT; ++timeIndex) {
		const std::size_t sourceOffset = static_cast<std::size_t>(timeIndex) * pixelCount;
		const std::size_t destinationOffset = static_cast<std::size_t>(timeIndex - 1) * pixelCount;
		cudaMemcpy(d_stimBuf + destinationOffset, d_stimBuf + sourceOffset, frameBytes, cudaMemcpyDeviceToDevice);
	}

	float* latestGrayFrame = d_stimBuf + pixelCount * static_cast<std::size_t>(nrT - 1);
	constexpr int threadsPerBlock = 128;
	const int blocks = iDivUp(static_cast<int>(pixelCount), threadsPerBlock);

	if (nrC == 3) {
		dev_split_RGB<<<blocks, threadsPerBlock>>>(d_stim, latestGrayFrame, static_cast<int>(pixelCount));
	}
	else {
		dev_split_gray<<<blocks, threadsPerBlock>>>(d_stim, latestGrayFrame, static_cast<int>(pixelCount));
	}

	resetResponseBuffers();
}

void MOTION_CLASS_NAME::loadInputReferenceBuffer(unsigned char* stim) {

	const std::size_t pixelCount = static_cast<std::size_t>(nrX * nrY);

	const std::size_t frameBytes = pixelCount * sizeof(float);

	cudaMemcpy(d_stim, stim, nrC * nrX * nrY, cudaMemcpyHostToDevice);

	float* latestGrayFrame = d_stimBuf + pixelCount * static_cast<std::size_t>(nrT - 1);

	constexpr int threadsPerBlock = 128;

	const int blocks = iDivUp(static_cast<int>(pixelCount), threadsPerBlock);

	if (nrC == 3) {
		dev_split_RGB << <blocks, threadsPerBlock >> > (d_stim, latestGrayFrame, static_cast<int>(pixelCount));
	}
	else {
		dev_split_gray << <blocks, threadsPerBlock >> > (d_stim, latestGrayFrame, static_cast<int>(pixelCount));
	}

	// Match the reference project semantics: write the newest frame first,
	// then shift [1..nrT-1] to [0..nrT-2].
	for (int timeIndex = 1; timeIndex < nrT; ++timeIndex) {
		const std::size_t sourceOffset = static_cast<std::size_t>(timeIndex) * pixelCount;

		const std::size_t destinationOffset = static_cast<std::size_t>(timeIndex - 1) * pixelCount;

		cudaMemcpy(d_stimBuf + destinationOffset, d_stimBuf + sourceOffset, frameBytes, cudaMemcpyDeviceToDevice);
	}

	resetResponseBuffers();

}

void MOTION_CLASS_NAME::resetResponseBuffers() {

	const std::size_t pixelCount = static_cast<std::size_t>(nrX * nrY);

	const std::size_t temporalVolumeBytes = pixelCount * static_cast<std::size_t>(nrT) * sizeof(float);

	const std::size_t filterResponseElements =
		pixelCount
		* static_cast<std::size_t>(nrFilters)
		* static_cast<std::size_t>(nrScales);

	cudaMemset(d_resp_, 0, filterResponseElements * sizeof(float));
	cudaMemcpy(d_scalingStimBuf, d_stimBuf, temporalVolumeBytes, cudaMemcpyDeviceToDevice);

	const std::size_t directionResponseElements =
		pixelCount
		* static_cast<std::size_t>(nrDirs);

	cudaMemset(d_respV1c, 0, directionResponseElements * sizeof(float));

}

void MOTION_CLASS_NAME::calcV1linear() {
	const std::size_t pixelCount = static_cast<std::size_t>(nrX * nrY);

	const std::size_t fullTemporalElements = pixelCount * nrT;

	const std::size_t V1TemporalElements = pixelCount * v1GaussFiltSize;

	for (int scaleIndex = 0; scaleIndex < nrScales; ++scaleIndex) {
		if (scaleIndex > 0) {
			float* smoothedStimulus = nullptr;

			cudaMalloc(reinterpret_cast<void**>(&smoothedStimulus), fullTemporalElements * sizeof(float));

			const dim3 fullSizes(nrX, nrY, nrT);

			conv3D(d_scalingStimBuf, smoothedStimulus, fullSizes, scalingFilt, scalingFiltSize);

            cudaFree(d_scalingStimBuf);

			d_scalingStimBuf = smoothedStimulus;
		}

		const int temporalStart = (nrT - v1GaussFiltSize) / 2;

		const std::size_t temporalStartOffset = static_cast<std::size_t>(temporalStart) * pixelCount;

		cudaMemcpy(d_v1GausBuf, d_scalingStimBuf + temporalStartOffset, V1TemporalElements * sizeof(float), cudaMemcpyDeviceToDevice);

		float* smoothedV1 = nullptr;

		cudaMalloc(reinterpret_cast<void**>(&smoothedV1), V1TemporalElements * sizeof(float));
		const dim3 v1Sizes(nrX, nrY, v1GaussFiltSize);

		conv3D(d_v1GausBuf, smoothedV1, v1Sizes, v1Gaus, v1GaussFiltSize);
        cudaFree(d_v1GausBuf);

		d_v1GausBuf = smoothedV1;

		for (int orderT = 0; orderT <= 3; ++orderT) {
			cudaMemcpy(diffV1GausBufT, d_v1GausBuf, V1TemporalElements * sizeof(float), cudaMemcpyDeviceToDevice);

			if (orderT > 0) {
				diffV1GausBufT = diff(diffV1GausBufT, v1Sizes, orderT, 2);
			}
			for (int orderY = 0; orderY <= 3 - orderT; ++orderY) {
				const int orderX = 3 - orderY - orderT;

				cudaMemcpy(d_diffV1GausBuf, diffV1GausBufT, V1TemporalElements * sizeof(float), cudaMemcpyDeviceToDevice);

				if (orderX > 0) {
					d_diffV1GausBuf = diff(d_diffV1GausBuf, v1Sizes, orderX, 0);

				}
				if (orderY > 0) {
					d_diffV1GausBuf = diff(d_diffV1GausBuf, v1Sizes, orderY, 1);
				}

				const std::size_t centerFrameOffset = pixelCount * static_cast<std::size_t>(v1GaussFiltSize / 2);

				float* scaleResponse = d_resp_ + static_cast<std::size_t>(scaleIndex) * pixelCount * nrFilters;

				accumDiffStims(scaleResponse, d_diffV1GausBuf + centerFrameOffset, v1Sizes, orderX, orderY, orderT);

			}
		}
	}

	const int totalResponseElements = nrX * nrY * nrFilters * nrScales;

	const std::size_t total_blocks = iDivUp(totalResponseElements, 128);

	dev_edges << <total_blocks, 128 >> > (d_resp_, totalResponseElements, nrX, nrY);



}


void MOTION_CLASS_NAME::calcV1rect() {

	int len = nrX * nrY * nrFilters * nrScales;
	dev_fullRect2 << <iDivUp(nrX * nrY * nrFilters * nrScales, 128), 128 >> > (d_resp_, len, scaleV1Linear_, scaleV1FullWaveRect_);

}

/*
 *
 *     d_resp_[scale][filter][y][x]

 *

 *
 * conv2D(d_resp_, tmp, ...) 
*/
void MOTION_CLASS_NAME::calcV1blur() {

	const std::size_t responseElements = nrX * nrY * nrFilters * nrScales;

	float* tmp = nullptr;

	cudaMalloc(reinterpret_cast<void**>(&tmp), responseElements * sizeof(float));

	const dim3 sizes(nrX, nrY, nrFilters * nrScales);

	conv2D(d_resp_, tmp, sizes, complexV1Filt, complexV1FiltSize);

	cudaFree(tmp);

	if (scaleV1Blur_ != 1.0) {
		dev_scale << <
			iDivUp(
				static_cast<int>(responseElements),
				128),
			128 >> > (
				d_resp_,
				static_cast<float>(scaleV1Blur_),
				static_cast<int>(responseElements));
	}

}

/*
 *
 *
 *     d_resp_[scale][filter][position]
 *
 * population buffer d_pop 
 *
 *     d_pop[scale][position]
 *
 *
* 
*/

void MOTION_CLASS_NAME::calcV1normalize() {

	const int pixelCount = nrX * nrY;

	const int responseElements = pixelCount * nrFilters * nrScales;

	const int populationElements = pixelCount * nrScales;

	const dim3 meanGrid(iDivUp(pixelCount, 128), nrScales, 1);

	dev_mean3 << <meanGrid, 128 >> > (d_resp_, d_pop, pixelCount, nrFilters);

	if (scaleV1Complex_ != 1.0) {
		dev_scale << <iDivUp(responseElements, 128), 128 >> > (d_resp_, static_cast<float>(scaleV1Complex_), responseElements);

	}

	float* tmp = nullptr;

	cudaMalloc(reinterpret_cast<void**>(&tmp), populationElements * sizeof(float));

	const dim3 populationSizes(nrX, nrY, nrScales);

	conv2D(d_pop, tmp, populationSizes, normV1filt, normV1filtSize);

	cudaFree(tmp);

	const double normScale = scaleV1NormStrength_ * scaleV1NormPopK_;
	if (normScale != 1.0) {
		dev_scale << <iDivUp(populationElements, 128), 128 >> > (d_pop, static_cast<float>(normScale), populationElements);
	}

	dev_normalize << <meanGrid, 128 >> > (d_resp_, d_pop, pixelCount, static_cast<float>(scaleV1C50_));

}


/*
 *
 *
 *     d_resp_[scale][filter][position]
 *
 *
 *     d_respV1c[direction][position]
 *
 * Collapse directional filter responses into the population buffer.
 *
*/
void MOTION_CLASS_NAME::calcV1direction(double speed) {

	int speedIndex = -1;

	if (speed == 1.5) {
		speedIndex = 0;
	}
	else if (speed == 0.125) {
		speedIndex = 1;
	}
	else if (speed == 9.0) {
		speedIndex = 2;
	}

	const int pixelCount = nrX * nrY;

	const int directionElements = pixelCount * nrDirs;

	devfilt2dir<<<iDivUp(directionElements, 256) , 256>>>(d_resp_, d_respV1c, pixelCount, nrScales, nrDirs, speedIndex);

	constexpr float spontaneousFiring = 1.0f;

	dev_scaleHalfRect << <iDivUp(directionElements, 128), 128 >> > (d_respV1c, directionElements, static_cast<float>(scaleV1ComplexFiring_), spontaneousFiring);

}



void MOTION_CLASS_NAME::conv3D(float* idata, float* odata, dim3 sizes, const float* filt, int filtlen) {

	const unsigned int sizeX = sizes.x;

	const unsigned int sizeY = sizes.y;

	const unsigned int sizeZ = sizes.z;

	const int validConX = CONV1_THREAD_SIZE - filtlen + 1;

	const dim3 gridX(iDivUp(sizeX, validConX), sizeY * sizeZ, 1);

	const dim3 blockX(CONV1_THREAD_SIZE, 1, 1);
	dev_conv1D << <gridX, blockX >> > (idata, odata, sizeX, filt, filtlen);

	float* inputBuffer = odata;
	float* outputBuffer = idata;

	const int validConvN = CONVN_THREAD_SIZE2 - filtlen + 1;

	const dim3 gridY(iDivUp(sizeX, CONVN_THREAD_SIZE1), iDivUp(sizeY, validConvN) * sizeZ, 1);

	const dim3 blockN(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);

	dev_convn << <gridY, blockN >> > (inputBuffer, outputBuffer, sizeX, sizeY, sizeX, sizeX * sizeY, sizeZ, filt, filtlen);

	inputBuffer = idata;

	outputBuffer = odata;

	const dim3 grdZ(iDivUp(sizeX, CONVN_THREAD_SIZE1), iDivUp(sizeZ, validConvN) * sizeY, 1);

	dev_convn << <grdZ, blockN >> > (inputBuffer, outputBuffer, sizeX, sizeZ, sizeX * sizeY, sizeX, sizeY, filt, filtlen);
}

void MOTION_CLASS_NAME::conv2D(float* idata, float* odata, dim3 sizes, const float* filt, int filtlen)
{
	const unsigned int sizeX = sizes.x;

	const unsigned int sizeY = sizes.y;

	const unsigned int sizeZ = sizes.z;

	const int validOutputsX = CONV1_THREAD_SIZE - filtlen + 1;

	const dim3 gridX(iDivUp(sizeX, validOutputsX), sizeY * sizeZ, 1);

	const dim3 blockX(CONV1_THREAD_SIZE, 1, 1);

	dev_conv1D<<< gridX, blockX>>>(idata, odata, sizeX, filt, filtlen);

	float* inputBuffer = odata;
    float* outputBuffer = idata;

	const int validOutputsY = CONVN_THREAD_SIZE2 - filtlen + 1;
	const dim3 gridY(iDivUp(sizeX, CONVN_THREAD_SIZE1), iDivUp(sizeY, validOutputsY) * sizeZ, 1);
	const dim3 blockN(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);

	dev_convn << <gridY, blockN >> > (inputBuffer, outputBuffer, sizeX, sizeY, sizeX, sizeX * sizeY, sizeZ, filt, filtlen);

}

/*
* * @param idata
 *
 * @param sizes
 *
 * @param order
 *
 * @param dim

*/
float* MOTION_CLASS_NAME::diff(float* idata, dim3 sizes, int order, int dim) {

	const unsigned int sizeX = sizes.x;
	const unsigned int sizeY = sizes.y;
	const unsigned int sizeZ = sizes.z;

	const std::size_t elementCount =
		static_cast<std::size_t>(sizeX)
		* static_cast<std::size_t>(sizeY)
		* static_cast<std::size_t>(sizeZ);

	float* filter = nullptr;
	int filterLength = 0;

	switch (order) {
	case 1:
		filter = diff1filt;
		filterLength = diff1FiltSize;
		break;

	case 2:
		filter = diff2filt;
		filterLength = diff2FiltSize;
		break;

	case 3:
		filter = diff3filt;
		filterLength = diff3FiltSize;
		break;
	}

	float* odata = nullptr;

	cudaMalloc(reinterpret_cast<void**>(&odata), elementCount * sizeof(float));

	if (dim == 0) {

		const int validOutput = CONV1_THREAD_SIZE - filterLength + 1;

		const dim3 grid(iDivUp(sizeX, validOutput), sizeY * sizeZ, 1);

		const dim3 block(CONV1_THREAD_SIZE, 1, 1);

		dev_conv1D << <grid, block >> > (idata, odata, sizeX, filter, filterLength);

	}
	else if (dim == 1) {

		const int validOutput = CONVN_THREAD_SIZE2 - filterLength + 1;

		const dim3 grid(iDivUp(sizeX, CONVN_THREAD_SIZE1), iDivUp(sizeY, validOutput) * sizeZ, 1);

		const dim3 block(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);

		dev_convn << <grid, block >> > (idata, odata, sizeX, sizeY, sizeX, sizeX * sizeY, sizeZ, filter, filterLength);

	}
	else {

		const int validOutputs = CONVN_THREAD_SIZE2 - filterLength + 1;

		const dim3 grid(iDivUp(sizeX, CONVN_THREAD_SIZE1), iDivUp(sizeZ, validOutputs) * sizeY, 1);

		const dim3 block(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);

		dev_convn << <grid, block >> > (idata, odata, sizeX, sizeZ, sizeX * sizeY, sizeX, sizeY, filter, filterLength);

	}

	cudaFree(idata);

	return odata;


}


/*
* this kernel add one of the third order response to the timespatial filter
* * @param dRespScale

 *
 *            dRespScale[filter][position]
 *
 * @param derivativeFrame

 *
 * @param sizes

 *
 * @param orderX

 *
 * @param orderY

 *
 * @param orderT

*/
void MOTION_CLASS_NAME::accumDiffStims(float* d_res_tmp, float* diffV1GausBuf, dim3 sizes, int orderX, int orderY, int orderT) {

	constexpr int factorials[4] = { 1, 1, 2, 6 };


	const int derivCoeff = factorials[3] / (factorials[orderX] * factorials[orderY] * factorials[orderT]);

	const int pixelCount = static_cast<int>(sizes.x * sizes.y);

	constexpr int threadsperblock = 256;

	const int blocks = iDivUp(pixelCount, threadsperblock);

	dev_accumDiffStims<<<blocks, threadsperblock>>>(d_res_tmp, diffV1GausBuf, pixelCount, derivCoeff, orderX, orderY, orderT);



}
