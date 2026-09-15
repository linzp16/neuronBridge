/*
* 文件名：IntegrationMethodFactory.h
* 积分方法工厂
*/

#ifndef INTEGRATIONMETHODFACTORY_H
#define INTEGRATIONMETHODFACTORY_H
#include "../source_file_realtime_v1_async/Intergration/inc/IntegerationMethod.h"
#include "../source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
#include "../source_file_realtime_v1_async/Intergration/inc/FixStep/ForwardEulerMethod.h"

template<typename NeuralModel>
class IntegrationMethodFactory {
public:
	static IntegerationMethod* createIntegerationMethod(ModelDescription intDescription, NeuralModel* neuralmodel) {
#define NPGR_INTEGRATION_METHOD(model_name, cpu_class, gpu_class, gpu_interface_class) \
		if (intDescription.ModelName == model_name) {                                  \
			return cpu_class<NeuralModel>::CreateIntegerationMethod(              \
				intDescription.ModelParameter, neuralmodel);                     \
		}
#include "../source_file_realtime_v1_async/ModelFactory/IntegrationMethodList.inc"
#undef NPGR_INTEGRATION_METHOD
		std::cout << "Unknown integration method type: " << intDescription.ModelName << std::endl;
		return 0;
	}
};





#endif // INTEGRATIONMETHODFACTORY_H
