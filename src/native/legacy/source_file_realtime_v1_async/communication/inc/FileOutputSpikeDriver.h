/*
* 文件名: FileOutputSpikeDriver.h
* 用于将脉冲与状态信息写入到文件中
*/

#ifndef FILEOUTPUTSPIKEDRIVER_H
#define FILEOUTPUTSPIKEDRIVER_H

#include "../source_file_realtime_v1_async/communication/inc/OutputSpikeDriver.h"
#include <cstdlib>
#include <string>

class FileOutputSpikeDriver : public OutputSpikeDriver {

	public:

		//定义文件名
		std::string filename;

		//瀹氫箟鏂囦欢鎸囬拡
		FILE* Handler;

		/*
		* 构造函数
		* NewFileName: 文件名
		*/
		explicit FileOutputSpikeDriver(const char* NewFileName);

		/*
		* 鏋愭瀯鍑芥暟
		*/
		~FileOutputSpikeDriver();


		/*
		* 向文件中写入脉冲信息
		* NewSpike: 新脉冲信息
		*/
		virtual void WriteSpike(Spike* NewSpike, float basetimestep);


		/*
		* 鏄惁鍙鍐欏叆鑴夊啿
		*/
		virtual bool IsBuffered();

		/*
		* 铏氬嚱鏁帮紝鍙戝竷鑴夊啿淇℃伅
		*/
		virtual void FlushBuffers();

		



};


#endif // FILEOUTPUTSPIKEDRIVER_H