#ifndef HANDWRITING_TIME_DRIVEN_MODEL_H
#define HANDWRITING_TIME_DRIVEN_MODEL_H

#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenModel.h"
#include <string>
#include <vector>

class HandwritingTimeDrivenModel : public TimeDrivenModel {
public:
    int N_DifferentialStates = 1;

    HandwritingTimeDrivenModel();
    HandwritingTimeDrivenModel(int timesteps);
    ~HandwritingTimeDrivenModel();

    virtual void InitStateVector(int NumberOfNeurons, int GPUIndex) override;
    virtual Neuron_State_Vector* InitState() override;
    virtual void UpdateState(int index, int time, Simulation* simulation) override;
    virtual InternalSpike* ProcessSpike(Interconnections* inter, int time) override;
    virtual void ProcessCurrent(Interconnections* inter, Neuron* Target, float current) override;
    virtual void InitializeInputCurrentSynapseStructure() override;
    virtual void CheckType(Interconnections* inter) override;
    virtual int getV_index() override;
    virtual int get_NumberOfState() override;
    virtual enum NeuronModelType getNeuronModelType() override;
    virtual std::map<std::string, boost::any> getParameters() override;
    virtual bool compare(NeuronModel* neuralmodel) override;

    void SetParameters(std::map<std::string, boost::any> parametermap, float basetimesteps);
    void CaculateDifferentialEquation(float* NeuronState, float* AuxNeuronState, int index);
    void CaculateTimeDependentEquation(float* NeuronState, int index, float dt);
    void CaculateSpike(float previous_V, float* NeuronState, int index);

private:
    enum ChannelKind {
        CHANNEL_NONE = 0,
        CHANNEL_AMPA = 1,
        CHANNEL_NMDA = 2,
        CHANNEL_GABA = 3
    };

    std::string role_;
    int n_state_ = 1;
    int n_differential_state_ = 1;
    int index_v_ = 0;
    float model_dt_ = 0.1f;

    float v_th_ = -50.0f;
    float v_reset_ = -60.0f;
    float v_spike_ = 20.0f;
    float e_ampa_ = 0.0f;
    float e_nmda_ = 0.0f;
    float e_gaba_ = -70.0f;
    float e_leak_ = -70.0f;
    float cm_ = 500.0f;
    float g_l_ = 25.0f;
    float tau_ampa_ = 2.0f;
    float tau_nmda_ = 100.0f;
    float tau_gaba_ = 10.0f;
    int t_ref_ = 20;
    float g1_ = 0.0f;
    float g2_ = 0.0f;
    float g3_ = 0.0f;
    float g4_ = 0.0f;
    float g5_ = 0.0f;

    std::vector<float> init_;
    std::vector<float> sigma_;
    std::vector<int> channel_kinds_;
    std::vector<float> pending_inputs_;

    void ConfigureRole();
    int StateIndexForSynapseType(int synapse_type) const;
    float NextConductanceState(float current_state_value, float input_value, int state_index, float dt) const;
    float NMDAVoltageFactor(float voltage) const;
    float GetGain(int slot) const;
};

#endif
