/*
* Integration_method_GPU_Interface_Facory.cuh
* GPU上积分方法接口的工厂
*/

#ifndef INTEGRATION_METHOD_GPU_INTERFACE_FACTORY_CUH
#define INTEGRATION_METHOD_GPU_INTERFACE_FACTORY_CUH
#include <boost/any.hpp>
#include <map>

#include "../source_file_realtime_v1_async/Intergration/inc/GPU_Method/FixStep/ForwardEulerMethodGPU_Interface.cuh"

template<typename NeuronModelGPUInerface>
class Integration_method_GPU_Interface_Factory {
public:
	static IntegrationMethodGPU_Interface* create_Integration_method_GPU_Interface(ModelDescription intDescription, NeuronModelGPUInerface* neuralmodel) {
#define NPGR_INTEGRATION_METHOD(model_name, cpu_class, gpu_class, gpu_interface_class) \
		if (intDescription.ModelName == model_name) {                                  \
			return gpu_interface_class<NeuronModelGPUInerface>::CreateIntegerationMethod( \
				intDescription.ModelParameter, neuralmodel);                     \
		}
#include "../source_file_realtime_v1_async/ModelFactory/IntegrationMethodList.inc"
#undef NPGR_INTEGRATION_METHOD
		std::cout << "Unknown integration method type: " << intDescription.ModelName << std::endl;
		return 0;
	}

};



#endif // !INTEGRATION_METHOD_GPU_INTERFACE_FACTORY_CUH
