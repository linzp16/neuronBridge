// Central legacy integration method registry list.
//
// The including file must define:
//   NPGR_INTEGRATION_METHOD(name, cpu_class, gpu_class, gpu_interface_class)
//
// The class entries are templates. CPU models use cpu_class<NeuralModel>,
// legacy GPU device models use gpu_class<NeuronModelGPU>, and GPU interface
// models use gpu_interface_class<NeuronModelGPUInterface>.
NPGR_INTEGRATION_METHOD("ForwardEulerMethod",
                        ForwardEulerMethod,
                        ForwardEulerMethodOnGPU,
                        ForwardEulerMethodGPU_Interface)
