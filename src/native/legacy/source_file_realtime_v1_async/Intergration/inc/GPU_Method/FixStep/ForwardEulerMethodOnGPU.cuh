/*

* 鏂囦欢鍚嶏細ForwardEulerMethodOnGPU.cuh

* 瀹氫箟浜嗗墠鍚戞鎷夋硶鍦? GPU 涓婄殑瀹炵幇

*/



#ifndef FORWARD_EULER_METHOD_ON_GPU_CUH

#define FORWARD_EULER_METHOD_ON_GPU_CUHGPU



#include "../source_file_realtime_v1_async/Intergration/inc/GPU_Method/FixStep/FixStepOnGPU.cuh"



template<typename NeuronModelOnGPU>

class ForwardEulerMethodOnGPU :public FixStepOnGPU<NeuronModelOnGPU> {

public:

	//杈呭姪澧為噺鍚戦噺

	float* AuxNeuronState;



	/*

	* 鏋勯?犲嚱鏁?

	*/

	__device__ ForwardEulerMethodOnGPU(NeuronModelOnGPU* Model, void** d_parm) : FixStepOnGPU<NeuronModelOnGPU>(Model, d_parm) {

		this->AuxNeuronState = ((float*)d_parm[1]);

	}



	/*

	* 鏋愭瀯鍑芥暟

	*/

	__device__ ~ForwardEulerMethodOnGPU() {}





	/*

	* 璁＄畻寰垎鏂圭▼澧為噺

	*/

	__device__ virtual void CaculateIncreament(int SizeStates, float* NeuronState) {

		//璁＄畻绾跨▼绱㈠紩

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		while (index < SizeStates) {

			float previous_voltage = NeuronState[index];

			//璁＄畻澧為噺

			this->neuron_model->CaculateDifferentialEquation(index, SizeStates, NeuronState, this->AuxNeuronState, this->dt);

			// 灏嗗井鍒嗘柟绋嬬粨鏋滅疮鍔犲洖鐘舵?佸彉閲?

			int batch_offset = gridDim.x * blockDim.x;

			int GPU_offset = blockDim.x * blockIdx.x + threadIdx.x;

			for (int i = 0; i < this->neuron_model->N_DifferentialStates; i++) {

				NeuronState[i * SizeStates + index] += this->dt * this->AuxNeuronState[i * batch_offset + GPU_offset];

			}

			//璁＄畻鐢靛

			this->neuron_model->CaculateTimeDependentEquation(index, SizeStates, NeuronState, this->dt, 0);

			// 绱姞璺濈涓婃鏀剧數鐨勬椂闂存鏁?

			this->neuron_model->Neuron_State_Vector->LastSpikeGPU[index] += 1;

			//鍒ゆ柇鏄惁鏀剧數

			this->neuron_model->CaculateSpike(previous_voltage, this->neuron_model->Neuron_State_Vector->Vector_of_State_VariableGPU, index, this->dt);

			// 妫?鏌ョН鍒嗙粨鏋滄槸鍚︽湁鏁?

			this->neuron_model->CheckValidIntegeration(index);



			index += gridDim.x * blockDim.x;



		}

	}



	/*

	* 閲嶇疆鐘舵??

	*/

	__device__ virtual void ResetState(int index) {};





	/*

	* 璁＄畻鐢靛

	*/

	__device__ virtual void Calculate_conductance_exp_values() {

		this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->TimeDependentInputSize, 1);

		this->neuron_model->Caculate_Conductance(0, this->dt);

	}



};





#endif // FORWARD_EULER_METHOD_ON_GPU_CUH

