// Central legacy neuron model registry list.
//
// The including file must define one of:
//   NPGR_LEGACY_TIMED_MODEL(implementation_name, class_name)
//   NPGR_LEGACY_GPU_TIMED_MODEL(implementation_name, class_name)
//   NPGR_LEGACY_SIMPLE_MODEL(implementation_name, class_name)
//   NPGR_LEGACY_CONFIGURED_SIMPLE_MODEL(implementation_name, class_name, configure_fn)
//
// Model names here are backend implementation names after NeuronModelCatalog
// alias resolution, not necessarily the public user-facing names.
NPGR_LEGACY_TIMED_MODEL("TimeDrivenLIF_Voltage_jump",
                        TimeDrivenLIF_Voltage_jump)
NPGR_LEGACY_TIMED_MODEL("TimeDrivenLIF_Exponential_Decay",
                        TimeDrivenLIF_Exponential_Decay)
NPGR_LEGACY_TIMED_MODEL("TimeDrivenLIF_Exponential_double",
                        TimeDrivenLIF_Exponential_double)
NPGR_LEGACY_TIMED_MODEL("TimeDrivenLIF_Exponential_triple",
                        TimeDrivenLIF_Exponential_triple)
NPGR_LEGACY_TIMED_MODEL("PoissonRate",
                        PoissonRate)
NPGR_LEGACY_TIMED_MODEL("HandwritingTimeDrivenModel",
                        HandwritingTimeDrivenModel)
NPGR_LEGACY_TIMED_MODEL("TimeDrivenIzhikevic_I_Exponential_Decay",
                        TimeDrivenIzhikevic_I_Exponential_Decay)
NPGR_LEGACY_TIMED_MODEL("TimeDrivenIzhikevic_Exponential_Decay",
                        TimeDrivenIzhikevic_I_Exponential_Decay)

NPGR_LEGACY_SIMPLE_MODEL("InputSpikeNeuronModel",
                         InputSpikeNeuronModel)
NPGR_LEGACY_SIMPLE_MODEL("InputCurrentNeuronModel",
                         InputCurrentNeuronModel)
NPGR_LEGACY_SIMPLE_MODEL("TriggerRelayNeuronModel",
                         TriggerRelayNeuronModel)
NPGR_LEGACY_CONFIGURED_SIMPLE_MODEL("OuterDynamicInterfaceNeuronModel",
                                    OuterDynamicInterfaceNeuronModel,
                                    ConfigureFromParameters)

NPGR_LEGACY_GPU_TIMED_MODEL("TimeDrivenLIF_Exponential_double_GPU",
                            TimeDrivenLIF_Exponential_double_GPU_Interface)
NPGR_LEGACY_GPU_TIMED_MODEL("TimeDrivenLIF_Exponential_triple_GPU",
                            TimeDrivenLIF_Exponential_triple_GPU_Interface)
NPGR_LEGACY_GPU_TIMED_MODEL("TimeDrivenLIF_Exponential_Decay_GPU",
                            TimeDrivenLIF_Exponential_Decay_GPU_Interface)
NPGR_LEGACY_GPU_TIMED_MODEL("TimeDrivenLIF_Voltage_jump_GPU",
                            TimeDrivenLIF_Voltage_jump_GPU_Interface)
NPGR_LEGACY_GPU_TIMED_MODEL("PoissonRate_GPU",
                            PoissonRate_GPU_Interface)
NPGR_LEGACY_GPU_TIMED_MODEL("TimeDrivenIzhikevic_I_Exponential_Decay_GPU",
                            TimeDrivenIzhikevic_Exponential_Decay_GPU_Interface)
NPGR_LEGACY_GPU_TIMED_MODEL("TimeDrivenIzhikevic_Exponential_Decay_GPU",
                            TimeDrivenIzhikevic_Exponential_Decay_GPU_Interface)
NPGR_LEGACY_GPU_TIMED_MODEL("EDLUTLikeLIF_GPU",
                            EDLUTLikeLIF_GPU_Interface)
