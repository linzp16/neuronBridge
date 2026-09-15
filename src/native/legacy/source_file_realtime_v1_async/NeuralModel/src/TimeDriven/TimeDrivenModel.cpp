#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenModel.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuronModel.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "../source_file_realtime_v1_async/ModelFactory/IntegrationMethodFactory.h"

TimeDrivenModel::TimeDrivenModel():NeuronModel(), integrationMethod(0){
	this->TimeDriven = true;
}

TimeDrivenModel::TimeDrivenModel(int timestep): NeuronModel(timestep), integrationMethod(0){
	this->TimeDriven = true;
}

TimeDrivenModel::~TimeDrivenModel(){
	if (this->integrationMethod != 0) {
		delete this->integrationMethod;
		this->integrationMethod = 0;
	}
}

void TimeDrivenModel::CheckValidIntegeration(int timestep, float valid_integeration) {
	//检查是否为NAN
	if (valid_integeration != valid_integeration) {
		for (int i = 0; this->StateVector->NumberofNeuron; i++) {
			if (this->StateVector->GetPrintableValuesAt(i, 0) != this->StateVector->GetPrintableValuesAt(i, 0)) {
				std::cout << timestep << "Error: The " << i << "th neuron is NAN" << std::endl;
				for (int j = 0; j < this->StateVector->NumberofStateVariable; j++) {
					std::cout << this->StateVector->GetPrintableValuesAt(i, j) << " ";
				}std::cout << std::endl;
				this->StateVector->ResetNeuronState(i);
			}
		}
	}
}

