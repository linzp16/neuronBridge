#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_triple.h"

#include "../source_file_realtime_v1_async/Current/inc/CurrentSynapse.h"
#include "../source_file_realtime_v1_async/ModelFactory/IntegrationMethodFactory.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <cmath>

namespace {

template <std::size_t N>
void ReadStateArray(const boost::any& value, std::array<float, N>& destination) {
    if (value.type() == typeid(std::array<float, N>)) {
        destination = boost::any_cast<std::array<float, N> >(value);
        return;
    }
    // The previous CPU triple inherited the double model's four-state arrays.
    if (N == 5 && value.type() == typeid(std::array<float, 4>)) {
        const std::array<float, 4> legacy = boost::any_cast<std::array<float, 4> >(value);
        for (std::size_t index = 0; index < legacy.size(); ++index) {
            destination[index] = legacy[index];
        }
        destination[4] = 0.0f;
        return;
    }
    throw boost::bad_any_cast();
}

}  // namespace

TimeDrivenLIF_Exponential_triple::TimeDrivenLIF_Exponential_triple()
    : TimeDrivenLIF_Exponential_double() {
    // Keep the CPU defaults identical to the legacy GPU triple interface.
    this->tau = 10.0f;
    delete this->StateVector;
    this->StateVector = new Neuron_State_Vector(this->N_NeuronStateVariables, true);
    this->setModelName(std::string("TimeDrivenLIF_Exponential_triple"));
}

TimeDrivenLIF_Exponential_triple::TimeDrivenLIF_Exponential_triple(int timesteps)
    : TimeDrivenLIF_Exponential_double(timesteps) {
    // Keep the CPU defaults identical to the legacy GPU triple interface.
    this->tau = 10.0f;
    delete this->StateVector;
    this->StateVector = new Neuron_State_Vector(this->N_NeuronStateVariables, true);
    this->setModelName(std::string("TimeDrivenLIF_Exponential_triple"));
}

TimeDrivenLIF_Exponential_triple::~TimeDrivenLIF_Exponential_triple() {
    this->ampa_decay_lookup_.clear();
    this->gaba_decay_lookup_.clear();
    this->nmda_decay_lookup_.clear();
}

void TimeDrivenLIF_Exponential_triple::ResetConductanceDecayLookup(float dt) {
    this->ampa_decay_lookup_.assign(1, std::exp(-dt / this->ampa_tau));
    this->gaba_decay_lookup_.assign(1, std::exp(-dt / this->gaba_tau));
    this->nmda_decay_lookup_.assign(1, std::exp(-dt / this->nmda_tau));
}

void TimeDrivenLIF_Exponential_triple::EnsureConductanceDecayLookupSize(float dt) {
    if (this->ampa_decay_lookup_.empty() || this->gaba_decay_lookup_.empty() ||
        this->nmda_decay_lookup_.empty()) {
        this->ResetConductanceDecayLookup(dt);
    }
}

void TimeDrivenLIF_Exponential_triple::InitStateVector(int NumberOfNeurons, int GPUIndex) {
    (void)GPUIndex;
    float new_init[] = {this->V_reset + this->init[0], this->init[1], this->init[2],
                        this->init[3], this->init[4]};
    float new_sigma[] = {this->sigma[0], this->sigma[1], this->sigma[2],
                         this->sigma[3], this->sigma[4]};
    this->StateVector->InitNeuronState(NumberOfNeurons, new_init, new_sigma);
    this->CurrentSynapeModel = new CurrentSynapse(NumberOfNeurons);
}

void TimeDrivenLIF_Exponential_triple::CaculateDifferentialEquation(
    float* NeuronState, float* AuxNeuronState, int index) {
    const float voltage = NeuronState[this->index_V];
    if (this->StateVector->LastSpike[index] <= this->t_ref) {
        AuxNeuronState[this->index_V] = 0.0f;
        return;
    }

    float current = 0.0f;
    if (this->Excited) {
        current += NeuronState[this->index_ampa] * (this->E_ampa - voltage);
    }
    if (this->Inhibitory) {
        current += NeuronState[this->index_gaba] * (this->E_gaba - voltage);
    }
    if (this->NMDA) {
        const float magnesium_block =
            1.0f / (1.0f + std::exp(-0.062f * voltage) * (1.2f / 3.57f));
        current += NeuronState[this->index_nmda] * magnesium_block *
                   (this->E_ampa - voltage);
    }
    if (this->I_EXT) {
        current += NeuronState[this->I_EXT_index];
    }
    AuxNeuronState[this->index_V] =
        (this->R * current + this->V_rest - voltage) / this->tau;
}

void TimeDrivenLIF_Exponential_triple::CaculateTimeDependentEquation(
    float* NeuronState, int index, float dt) {
    (void)index;
    this->EnsureConductanceDecayLookupSize(dt);
    const float zero_threshold = 1.0e-9f;

    if (NeuronState[this->index_ampa] < zero_threshold) {
        NeuronState[this->index_ampa] = 0.0f;
    } else {
        NeuronState[this->index_ampa] *= this->ampa_decay_lookup_[0];
    }
    if (NeuronState[this->index_gaba] < zero_threshold) {
        NeuronState[this->index_gaba] = 0.0f;
    } else {
        NeuronState[this->index_gaba] *= this->gaba_decay_lookup_[0];
    }
    if (NeuronState[this->index_nmda] < zero_threshold) {
        NeuronState[this->index_nmda] = 0.0f;
    } else {
        NeuronState[this->index_nmda] *= this->nmda_decay_lookup_[0];
    }
}

void TimeDrivenLIF_Exponential_triple::CaculateSpike(
    float previous_V, float* NeuronState, int index) {
    (void)previous_V;
    if (NeuronState[this->index_V] > this->V_th) {
        NeuronState[this->index_V] = this->V_reset;
        this->StateVector->SpikeIndex[this->StateVector->NumberofSpike] = index;
        ++this->StateVector->NumberofSpike;
        this->StateVector->LastSpike[index] = 0;
        this->integrationMethod->ResetState(index);
    }
}

InternalSpike* TimeDrivenLIF_Exponential_triple::ProcessSpike(
    Interconnections* inter, int time) {
    (void)time;
    if (inter->type == 0 && this->Excited) {
        this->StateVector->SetNeuronStateIncrement(
            inter->TargetNeuronModelIndex, this->index_ampa, inter->weight);
    } else if (inter->type == 1 && this->Inhibitory) {
        this->StateVector->SetNeuronStateIncrement(
            inter->TargetNeuronModelIndex, this->index_gaba, inter->weight);
    } else if (inter->type == 2 && this->NMDA) {
        this->StateVector->SetNeuronStateIncrement(
            inter->TargetNeuronModelIndex, this->index_nmda, inter->weight);
    }
    return 0;
}

void TimeDrivenLIF_Exponential_triple::ProcessCurrent(
    Interconnections* inter, Neuron* Target, float current) {
    this->CurrentSynapeModel->SetInputCurrentPerSynapse(
        Target->index_in_NeuronModel, inter->subindex_type, current);
    const float total_current = this->CurrentSynapeModel->GetTotalInputCurrentPerNeuron(
        Target->index_in_NeuronModel);
    this->StateVector->SetNeuronState(
        Target->index_in_NeuronModel, this->I_EXT_index, total_current);
}

void TimeDrivenLIF_Exponential_triple::CheckType(Interconnections* inter) {
    if (inter->type == 0) {
        this->Excited = true;
    } else if (inter->type == 1) {
        this->Inhibitory = true;
    } else if (inter->type == 2) {
        this->NMDA = true;
    } else if (inter->type == 3) {
        this->I_EXT = true;
        inter->subindex_type = this->CurrentSynapeModel->N_connections[
            inter->TargetNeuron->index_in_NeuronModel];
        this->CurrentSynapeModel->IncrementNInputCurrentSynapsesPerNeuron(
            inter->TargetNeuron->index_in_NeuronModel);
    } else {
        std::cout << "Error: Type of synapse is not supported in this model" << std::endl;
    }
}

int TimeDrivenLIF_Exponential_triple::get_NumberOfState() {
    return this->N_NeuronStateVariables;
}

bool TimeDrivenLIF_Exponential_triple::compare(NeuronModel* neuralmodel) {
    if (!TimeDrivenModel::compare(neuralmodel)) {
        return false;
    }
    TimeDrivenLIF_Exponential_triple* other =
        dynamic_cast<TimeDrivenLIF_Exponential_triple*>(neuralmodel);
    if (other == 0) {
        return false;
    }
    bool equal = this->V_rest == other->V_rest && this->tau == other->tau &&
                 this->V_th == other->V_th && this->R == other->R &&
                 this->V_reset == other->V_reset && this->t_ref == other->t_ref &&
                 this->E_ampa == other->E_ampa && this->ampa_tau == other->ampa_tau &&
                 this->E_gaba == other->E_gaba && this->gaba_tau == other->gaba_tau &&
                 this->nmda_tau == other->nmda_tau && this->init == other->init &&
                 this->sigma == other->sigma;
    if (equal && this->integrationMethod != 0 && other->integrationMethod != 0) {
        equal = this->integrationMethod->compare(other->integrationMethod);
    }
    return equal;
}

void TimeDrivenLIF_Exponential_triple::SetParameters(
    std::map<std::string, boost::any> parametermap, float basetimestep) {
    std::map<std::string, boost::any>::iterator iter;

#define READ_FLOAT_PARAMETER(name, member)                    \
    iter = parametermap.find(name);                            \
    if (iter != parametermap.end()) {                          \
        this->member = boost::any_cast<float>(iter->second);   \
        parametermap.erase(iter);                              \
    }

    READ_FLOAT_PARAMETER("V_rest", V_rest)
    READ_FLOAT_PARAMETER("tau", tau)
    READ_FLOAT_PARAMETER("V_th", V_th)
    READ_FLOAT_PARAMETER("R", R)
    READ_FLOAT_PARAMETER("V_reset", V_reset)

    iter = parametermap.find("t_ref");
    if (iter != parametermap.end()) {
        this->t_ref = boost::any_cast<int>(iter->second);
        parametermap.erase(iter);
    }

    READ_FLOAT_PARAMETER("E_ampa", E_ampa)
    READ_FLOAT_PARAMETER("ampa_tau", ampa_tau)
    READ_FLOAT_PARAMETER("E_gaba", E_gaba)
    READ_FLOAT_PARAMETER("gaba_tau", gaba_tau)
    READ_FLOAT_PARAMETER("nmda_tau", nmda_tau)
    READ_FLOAT_PARAMETER("Eexc", E_ampa)
    READ_FLOAT_PARAMETER("gexc_tau", ampa_tau)
    READ_FLOAT_PARAMETER("Einh", E_gaba)
    READ_FLOAT_PARAMETER("ginh_tau", gaba_tau)

#undef READ_FLOAT_PARAMETER

    iter = parametermap.find("random_mu");
    if (iter != parametermap.end()) {
        ReadStateArray(iter->second, this->init);
        parametermap.erase(iter);
    }
    iter = parametermap.find("random_sigma");
    if (iter != parametermap.end()) {
        ReadStateArray(iter->second, this->sigma);
        parametermap.erase(iter);
    }

    if (this->integrationMethod != 0) {
        delete this->integrationMethod;
        this->integrationMethod = 0;
    }
    iter = parametermap.find("int_method");
    if (iter != parametermap.end()) {
        ModelDescription description = boost::any_cast<ModelDescription>(iter->second);
        description.ModelParameter["step"] = basetimestep * this->getTimestepSize();
        this->integrationMethod =
            IntegrationMethodFactory<TimeDrivenLIF_Exponential_triple>::createIntegerationMethod(
                description, this);
    } else {
        ModelDescription description;
        description.ModelName = "ForwardEulerMethod";
        description.ModelParameter["step"] = basetimestep * this->getTimestepSize();
        this->integrationMethod =
            IntegrationMethodFactory<TimeDrivenLIF_Exponential_triple>::createIntegerationMethod(
                description, this);
    }
}

std::map<std::string, boost::any> TimeDrivenLIF_Exponential_triple::getParameters() {
    std::map<std::string, boost::any> parameters;
    parameters["V_rest"] = this->V_rest;
    parameters["tau"] = this->tau;
    parameters["V_th"] = this->V_th;
    parameters["R"] = this->R;
    parameters["V_reset"] = this->V_reset;
    parameters["t_ref"] = this->t_ref;
    parameters["E_ampa"] = this->E_ampa;
    parameters["ampa_tau"] = this->ampa_tau;
    parameters["E_gaba"] = this->E_gaba;
    parameters["gaba_tau"] = this->gaba_tau;
    parameters["nmda_tau"] = this->nmda_tau;
    parameters["random_mu"] = this->init;
    parameters["random_sigma"] = this->sigma;
    ModelDescription integration;
    integration.ModelParameter = this->integrationMethod->getParameters();
    integration.ModelName = boost::any_cast<std::string>(integration.ModelParameter["name"]);
    parameters["int_method"] = integration;
    return parameters;
}
