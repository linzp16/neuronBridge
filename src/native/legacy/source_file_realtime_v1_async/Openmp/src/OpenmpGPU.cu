#include "../source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include <cuda_runtime.h>
#include <iostream>

int NumberOfOpenMPQueues;//CPU线程数
int NumberOfGPU;//GPU个数
int* GPUIndex;//GPU索引列表指针

/*
* 设置 OpenMP 线程数
*/
void setNumberOfOpenMPQueues(int n) {
	// 设置 CPU 线程数
	// omp_get_max_threads() 是进程当前的可变线程设置；不能把它当作硬件上限。
	// 例如前一个仿真设置 queues=1 后，它会变成 1，导致后续 queues=2/4
	// 被错误地降级为单线程。使用 omp_get_num_procs() 获取稳定的处理器上限。
	int available_processors = omp_get_num_procs();
	if (available_processors <= 0) {
		available_processors = omp_get_max_threads();
	}
	if (n > available_processors) {
		n = available_processors;
		std::cout << "Warning: Number of OpenMP queues exceeds available OpenMP processors. Setting it to " << n << std::endl;
	}
    #ifdef _OPENMP
	    std::cout << "OpenMP support is enabled" << std::endl;
    #else
	    std::cout << "OpenMP support is disabled" << std::endl;
    #endif
	(void) omp_set_num_threads(n);
	NumberOfOpenMPQueues = n;

	
	// 获取 GPU 数量
	int n_Device = 0; //GPU个数
	int validDevice = 0; //有效GPU个数(要求计算能力>=2.0，即CUDA版本大于6.5)
	// 读取 GPU 数量
    cudaGetDeviceCount(&n_Device);
	if (n_Device == 0) {
        std::cout << "Error: No GPU devices found." << std::endl;
		NumberOfGPU = 0;
	}
	else {
		GPUIndex = new int[n_Device]();
		for (int i = 0; i < n_Device; i++) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, i);
			if (deviceProp.major >= 2) {
				GPUIndex[validDevice] = i;
				validDevice++;
			}
		}
		if (validDevice == 0) {
			std::cout << "Error: No valid GPU devices found." << std::endl;
		}

		if (validDevice > NumberOfOpenMPQueues) {
			NumberOfGPU = NumberOfOpenMPQueues;
		}
		else {
			NumberOfGPU = validDevice;
		}
	}

}
