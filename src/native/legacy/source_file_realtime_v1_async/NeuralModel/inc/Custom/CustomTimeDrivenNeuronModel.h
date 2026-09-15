#ifndef NPGR_CUSTOM_TIME_DRIVEN_NEURON_MODEL_H
#define NPGR_CUSTOM_TIME_DRIVEN_NEURON_MODEL_H

#include "../source_file_realtime_v1_async/NeuralModel/inc/Custom/CustomEquationDescriptor.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenModel.h"

class CurrentSynapse;

// Shared legacy-CPU runtime contract for generated time-driven neurons.
// Generated concrete types retain their own equation and integration methods.
class CustomTimeDrivenNeuronModel : public TimeDrivenModel {
public:
    explicit CustomTimeDrivenNeuronModel(
        const npgr::CustomEquationDescriptor& descriptor,
        int timestep_size);
    ~CustomTimeDrivenNeuronModel() override;

    void InitStateVector(int neuron_count, int gpu_index) override;
    Neuron_State_Vector* InitState() override;
    InternalSpike* ProcessSpike(Interconnections* inter, int arrival_time) override;
    void ProcessCurrent(Interconnections* inter, Neuron* target, float current) override;
    void InitializeInputCurrentSynapseStructure() override;
    void CheckType(Interconnections* inter) override;
    int getV_index() override;
    int get_NumberOfState() override;
    NeuronModelType getNeuronModelType() override;

protected:
    const npgr::CustomEquationDescriptor& descriptor() const;
    void InitializeStateVector(
        int neuron_count,
        float* initial_state,
        float* initial_sigma);
    void MarkSpike(int neuron_index);

private:
    const npgr::CustomEquationDescriptor& descriptor_;
    CurrentSynapse* current_synapse_model_;

    const npgr::CustomInputBinding* FindInputBinding(
        int connection_type,
        npgr::CustomInputDelivery delivery) const;
};

#endif
