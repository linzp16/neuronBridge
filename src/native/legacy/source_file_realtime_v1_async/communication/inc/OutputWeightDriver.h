/*
* 文件名: OutputWeightDriver.h
* 定义了突触权重输出驱动器接口
*/
#ifndef FILE_OUTPUT_WEIGHT_DRIVER_H
#define FILE_OUTPUT_WEIGHT_DRIVER_H
class Simulation;

class OutputWeightDriver {
	public:
		/*
		* 默认构造函数
		*/
		OutputWeightDriver();

		/*
		* 默认析构函数
		*/
		~OutputWeightDriver();


		/*
		* 写入权重函数
		* @simulation: simulation object
		* @time: 时间
		*/
		virtual void WriteWeight(Simulation* simulation, int time) = 0;
		
};


#endif // FILE_OUTPUT_WEIGHT_DRIVER_H
