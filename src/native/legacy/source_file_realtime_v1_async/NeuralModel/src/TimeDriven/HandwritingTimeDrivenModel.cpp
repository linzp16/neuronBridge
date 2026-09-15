#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/HandwritingTimeDrivenModel.h"
#include "../source_file_realtime_v1_async/ModelFactory/IntegrationMethodFactory.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <algorithm>
#include <cmath>

namespace {

template <typename T>
bool ReadOptionalAny(std::map<std::string, boost::any>& values, const char* key, T& target) {
    std::map<std::string, boost::any>::iterator it = values.find(key);
    if (it == values.end()) {
        return false;
    }
    target = boost::any_cast<T>(it->second);
    values.erase(it);
    return true;
}

}

HandwritingTimeDrivenModel::HandwritingTimeDrivenModel() : TimeDrivenModel() {
    this->setModelName("HandwritingTimeDrivenModel");
    this->StateVector = new Neuron_State_Vector(this->n_state_, true);
}

HandwritingTimeDrivenModel::HandwritingTimeDrivenModel(int timesteps) : TimeDrivenModel(timesteps) {
    this->setModelName("HandwritingTimeDrivenModel");
    this->StateVector = new Neuron_State_Vector(this->n_state_, true);
}

HandwritingTimeDrivenModel::~HandwritingTimeDrivenModel() {
}

void HandwritingTimeDrivenModel::ConfigureRole() {
    if (this->StateVector != 0) {
        delete this->StateVector;
        this->StateVector = 0;
    }

    if (this->role_ == "CM") {
        this->n_state_ = 3;
        this->channel_kinds_ = {CHANNEL_NONE, CHANNEL_AMPA, CHANNEL_GABA};
    } else if (this->role_ == "BG") {
        this->n_state_ = 13;
        this->channel_kinds_ = {CHANNEL_NONE,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_AMPA, CHANNEL_NMDA};
    } else if (this->role_ == "MM") {
        this->n_state_ = 4;
        this->channel_kinds_ = {CHANNEL_NONE, CHANNEL_AMPA, CHANNEL_AMPA, CHANNEL_NMDA};
    } else if (this->role_ == "E") {
        this->n_state_ = 12;
        this->channel_kinds_ = {CHANNEL_NONE,
            CHANNEL_AMPA,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_GABA,
            CHANNEL_AMPA,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_AMPA, CHANNEL_NMDA,
            CHANNEL_AMPA, CHANNEL_NMDA};
    } else if (this->role_ == "I") {
        this->n_state_ = 5;
        this->channel_kinds_ = {CHANNEL_NONE, CHANNEL_AMPA, CHANNEL_AMPA, CHANNEL_NMDA, CHANNEL_GABA};
    } else {
        throw std::runtime_error("Unsupported handwriting neuron role: " + this->role_);
    }

    this->init_.assign(static_cast<std::size_t>(this->n_state_), 0.0f);
    this->sigma_.assign(static_cast<std::size_t>(this->n_state_), 0.0f);
    this->StateVector = new Neuron_State_Vector(this->n_state_, true);
}

void HandwritingTimeDrivenModel::InitStateVector(int NumberOfNeurons, int GPUIndex) {
    (void)GPUIndex;
    std::vector<float> initial_state(static_cast<std::size_t>(this->n_state_), 0.0f);
    std::vector<float> sigma_state(static_cast<std::size_t>(this->n_state_), 0.0f);
    initial_state[0] = this->v_reset_ + this->init_[0];
    sigma_state[0] = this->sigma_[0];
    for (int i = 1; i < this->n_state_; ++i) {
        initial_state[static_cast<std::size_t>(i)] = this->init_[static_cast<std::size_t>(i)];
        sigma_state[static_cast<std::size_t>(i)] = this->sigma_[static_cast<std::size_t>(i)];
    }
    this->StateVector->InitNeuronState(NumberOfNeurons, initial_state.data(), sigma_state.data());
    this->pending_inputs_.assign(static_cast<std::size_t>(NumberOfNeurons) * static_cast<std::size_t>(this->n_state_), 0.0f);
}

Neuron_State_Vector* HandwritingTimeDrivenModel::InitState() {
    return this->StateVector;
}

void HandwritingTimeDrivenModel::UpdateState(int index, int time, Simulation* simulation) {
    (void)index;
    this->StateVector->NumberofSpike = 0;
    this->integrationMethod->CaculateIncreament(simulation, time);
    this->CheckValidIntegeration(time, this->integrationMethod->GetValidIntegrationVariable());
}

float HandwritingTimeDrivenModel::NMDAVoltageFactor(float voltage) const {
    return 1.0f / (1.0f + std::exp(-0.062f * voltage / 3.57f));
}

float HandwritingTimeDrivenModel::GetGain(int slot) const {
    switch (slot) {
    case 1: return this->g1_;
    case 2: return this->g2_;
    case 3: return this->g3_;
    case 4: return this->g4_;
    case 5: return this->g5_;
    default: return 0.0f;
    }
}

void HandwritingTimeDrivenModel::CaculateDifferentialEquation(float* NeuronState, float* AuxNeuronState, int index) {
    if (this->StateVector->LastSpike[index] <= this->t_ref_) {
        AuxNeuronState[this->index_v_] = 0.0f;
        return;
    }

    const float vm = NeuronState[this->index_v_];
    const std::size_t base = static_cast<std::size_t>(index) * static_cast<std::size_t>(this->n_state_);
    const auto next_state = [&](int state_index) {
        return this->NextConductanceState(
            NeuronState[state_index],
            this->pending_inputs_[base + static_cast<std::size_t>(state_index)],
            state_index,
            this->model_dt_);
    };
    float i_ex = 0.0f;

    if (this->role_ == "CM") {
        i_ex += -this->g1_ * next_state(1) * (vm - this->e_ampa_);
        i_ex += -this->g2_ * next_state(2) * (vm - this->e_gaba_);
    } else if (this->role_ == "BG") {
        const float nmda_factor = this->NMDAVoltageFactor(vm);
        i_ex += -this->g1_ * next_state(1) * (vm - this->e_ampa_) - this->g1_ * next_state(2) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g1_ * next_state(3) * (vm - this->e_ampa_) - this->g1_ * next_state(4) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g1_ * next_state(5) * (vm - this->e_ampa_) - this->g1_ * next_state(6) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g2_ * next_state(7) * (vm - this->e_ampa_) - this->g2_ * next_state(8) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g3_ * next_state(9) * (vm - this->e_ampa_) - this->g3_ * next_state(10) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g3_ * next_state(11) * (vm - this->e_ampa_) - this->g3_ * next_state(12) * (vm - this->e_nmda_) * nmda_factor;
    } else if (this->role_ == "MM") {
        const float nmda_factor = this->NMDAVoltageFactor(vm);
        i_ex += -this->g1_ * next_state(1) * (vm - this->e_ampa_);
        i_ex += -this->g2_ * next_state(2) * (vm - this->e_ampa_) - this->g2_ * next_state(3) * (vm - this->e_nmda_) * nmda_factor;
    } else if (this->role_ == "E") {
        const float nmda_factor = this->NMDAVoltageFactor(vm);
        i_ex += -this->g1_ * next_state(1) * (vm - this->e_ampa_);
        i_ex += -this->g2_ * next_state(2) * (vm - this->e_ampa_) - this->g2_ * next_state(3) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g3_ * next_state(4) * (vm - this->e_gaba_);
        i_ex += -this->g4_ * next_state(5) * (vm - this->e_ampa_);
        i_ex += -this->g5_ * next_state(6) * (vm - this->e_ampa_) - this->g5_ * next_state(7) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g5_ * next_state(8) * (vm - this->e_ampa_) - this->g5_ * next_state(9) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g5_ * next_state(10) * (vm - this->e_ampa_) - this->g5_ * next_state(11) * (vm - this->e_nmda_) * nmda_factor;
    } else if (this->role_ == "I") {
        const float nmda_factor = this->NMDAVoltageFactor(vm);
        i_ex += -this->g1_ * next_state(1) * (vm - this->e_ampa_);
        i_ex += -this->g2_ * next_state(2) * (vm - this->e_ampa_) - this->g2_ * next_state(3) * (vm - this->e_nmda_) * nmda_factor;
        i_ex += -this->g3_ * next_state(4) * (vm - this->e_gaba_);
    }

    const float i_in = -this->g_l_ * (vm - this->e_leak_);
    AuxNeuronState[this->index_v_] = (i_in + i_ex) / this->cm_;
}

void HandwritingTimeDrivenModel::CaculateTimeDependentEquation(float* NeuronState, int index, float dt) {
    const std::size_t base = static_cast<std::size_t>(index) * static_cast<std::size_t>(this->n_state_);
    for (int state = 1; state < this->n_state_; ++state) {
        NeuronState[state] = this->NextConductanceState(
            NeuronState[state],
            this->pending_inputs_[base + static_cast<std::size_t>(state)],
            state,
            dt);
        this->pending_inputs_[base + static_cast<std::size_t>(state)] = 0.0f;
    }
}

void HandwritingTimeDrivenModel::CaculateSpike(float previous_V, float* NeuronState, int index) {
    (void)previous_V;
    if (NeuronState[this->index_v_] > this->v_th_) {
        NeuronState[this->index_v_] = this->v_reset_;
        this->StateVector->SpikeIndex[this->StateVector->NumberofSpike] = index;
        this->StateVector->NumberofSpike += 1;
        this->StateVector->LastSpike[index] = 0;
    }
}

int HandwritingTimeDrivenModel::StateIndexForSynapseType(int synapse_type) const {
    if (this->role_ == "CM") {
        if (synapse_type == 10) return 1;
        if (synapse_type == 11) return 2;
    } else if (this->role_ == "BG") {
        if (synapse_type == 20) return 1;
        if (synapse_type == 21) return 2;
        if (synapse_type == 22) return 3;
        if (synapse_type == 23) return 4;
        if (synapse_type == 24) return 5;
        if (synapse_type == 25) return 6;
        if (synapse_type == 26) return 7;
        if (synapse_type == 27) return 8;
        if (synapse_type == 28) return 9;
        if (synapse_type == 29) return 10;
        if (synapse_type == 30) return 11;
        if (synapse_type == 31) return 12;
    } else if (this->role_ == "MM") {
        if (synapse_type == 40) return 1;
        if (synapse_type == 41) return 2;
        if (synapse_type == 42) return 3;
    } else if (this->role_ == "E") {
        if (synapse_type == 50) return 1;
        if (synapse_type == 51) return 2;
        if (synapse_type == 52) return 3;
        if (synapse_type == 53) return 4;
        if (synapse_type == 54) return 5;
        if (synapse_type == 55) return 6;
        if (synapse_type == 56) return 7;
        if (synapse_type == 57) return 8;
        if (synapse_type == 58) return 9;
        if (synapse_type == 59) return 10;
        if (synapse_type == 60) return 11;
    } else if (this->role_ == "I") {
        if (synapse_type == 70) return 1;
        if (synapse_type == 71) return 2;
        if (synapse_type == 72) return 3;
        if (synapse_type == 73) return 4;
    }
    return -1;
}

float HandwritingTimeDrivenModel::NextConductanceState(float current_state_value, float input_value, int state_index, float dt) const {
    const int kind = this->channel_kinds_[static_cast<std::size_t>(state_index)];
    if (kind == CHANNEL_NMDA) {
        return current_state_value + (-current_state_value / this->tau_nmda_ + 0.63f * (1.0f - current_state_value) * input_value) * dt;
    }
    if (kind == CHANNEL_AMPA) {
        return current_state_value + (-current_state_value / this->tau_ampa_ + input_value) * dt;
    }
    if (kind == CHANNEL_GABA) {
        return current_state_value + (-current_state_value / this->tau_gaba_ + input_value) * dt;
    }
    return current_state_value;
}

InternalSpike* HandwritingTimeDrivenModel::ProcessSpike(Interconnections* inter, int time) {
    (void)time;
    const int state_index = this->StateIndexForSynapseType(inter->type);
    if (state_index < 0) {
        return 0;
    }
    const std::size_t slot = static_cast<std::size_t>(inter->TargetNeuronModelIndex) * static_cast<std::size_t>(this->n_state_) + static_cast<std::size_t>(state_index);
    this->pending_inputs_[slot] += inter->weight;
    return 0;
}

void HandwritingTimeDrivenModel::ProcessCurrent(Interconnections* inter, Neuron* Target, float current) {
    (void)inter;
    (void)Target;
    (void)current;
}

void HandwritingTimeDrivenModel::InitializeInputCurrentSynapseStructure() {
}

void HandwritingTimeDrivenModel::CheckType(Interconnections* inter) {
    if (this->StateIndexForSynapseType(inter->type) < 0) {
        std::cout << "Unsupported handwriting synapse type " << inter->type << " for role " << this->role_ << std::endl;
    }
}

int HandwritingTimeDrivenModel::getV_index() {
    return this->index_v_;
}

int HandwritingTimeDrivenModel::get_NumberOfState() {
    return this->n_state_;
}

enum NeuronModelType HandwritingTimeDrivenModel::getNeuronModelType() {
    return NEURAL_LAYER;
}

void HandwritingTimeDrivenModel::SetParameters(std::map<std::string, boost::any> parametermap, float basetimesteps) {
    if (!ReadOptionalAny(parametermap, "role", this->role_)) {
        throw std::runtime_error("HandwritingTimeDrivenModel requires 'role' parameter");
    }
    this->ConfigureRole();

    ReadOptionalAny(parametermap, "V_th", this->v_th_);
    ReadOptionalAny(parametermap, "V_reset", this->v_reset_);
    ReadOptionalAny(parametermap, "V_spike", this->v_spike_);
    ReadOptionalAny(parametermap, "E_AMPA", this->e_ampa_);
    ReadOptionalAny(parametermap, "E_NMDA", this->e_nmda_);
    ReadOptionalAny(parametermap, "E_GABA", this->e_gaba_);
    ReadOptionalAny(parametermap, "E_leak", this->e_leak_);
    ReadOptionalAny(parametermap, "Cm", this->cm_);
    ReadOptionalAny(parametermap, "gL", this->g_l_);
    ReadOptionalAny(parametermap, "tau_AMPA", this->tau_ampa_);
    ReadOptionalAny(parametermap, "tau_NMDA", this->tau_nmda_);
    ReadOptionalAny(parametermap, "tau_GABA", this->tau_gaba_);
    ReadOptionalAny(parametermap, "t_ref", this->t_ref_);
    ReadOptionalAny(parametermap, "g1", this->g1_);
    ReadOptionalAny(parametermap, "g2", this->g2_);
    ReadOptionalAny(parametermap, "g3", this->g3_);
    ReadOptionalAny(parametermap, "g4", this->g4_);
    ReadOptionalAny(parametermap, "g5", this->g5_);
    ReadOptionalAny(parametermap, "random_mu", this->init_);
    ReadOptionalAny(parametermap, "random_sigma", this->sigma_);

    if (this->init_.empty()) {
        this->init_.assign(static_cast<std::size_t>(this->n_state_), 0.0f);
    }
    if (this->sigma_.empty()) {
        this->sigma_.assign(static_cast<std::size_t>(this->n_state_), 0.0f);
    }
    if (static_cast<int>(this->init_.size()) != this->n_state_) {
        this->init_.assign(static_cast<std::size_t>(this->n_state_), 0.0f);
    }
    if (static_cast<int>(this->sigma_.size()) != this->n_state_) {
        this->sigma_.assign(static_cast<std::size_t>(this->n_state_), 0.0f);
    }

    this->model_dt_ = basetimesteps * static_cast<float>(this->getTimestepSize());

    if (this->integrationMethod != 0) {
        delete this->integrationMethod;
        this->integrationMethod = 0;
    }

    std::map<std::string, boost::any>::iterator iter = parametermap.find("int_method");
    if (iter != parametermap.end()) {
        ModelDescription temp = boost::any_cast<ModelDescription>(iter->second);
        temp.ModelParameter["step"] = this->model_dt_;
        this->integrationMethod = IntegrationMethodFactory<HandwritingTimeDrivenModel>::createIntegerationMethod(temp, this);
    } else {
        ModelDescription temp;
        temp.ModelName = "ForwardEulerMethod";
        temp.ModelParameter["step"] = this->model_dt_;
        this->integrationMethod = IntegrationMethodFactory<HandwritingTimeDrivenModel>::createIntegerationMethod(temp, this);
    }
}

std::map<std::string, boost::any> HandwritingTimeDrivenModel::getParameters() {
    std::map<std::string, boost::any> values;
    values["role"] = this->role_;
    values["V_th"] = this->v_th_;
    values["V_reset"] = this->v_reset_;
    values["V_spike"] = this->v_spike_;
    values["E_AMPA"] = this->e_ampa_;
    values["E_NMDA"] = this->e_nmda_;
    values["E_GABA"] = this->e_gaba_;
    values["E_leak"] = this->e_leak_;
    values["Cm"] = this->cm_;
    values["gL"] = this->g_l_;
    values["tau_AMPA"] = this->tau_ampa_;
    values["tau_NMDA"] = this->tau_nmda_;
    values["tau_GABA"] = this->tau_gaba_;
    values["t_ref"] = this->t_ref_;
    values["g1"] = this->g1_;
    values["g2"] = this->g2_;
    values["g3"] = this->g3_;
    values["g4"] = this->g4_;
    values["g5"] = this->g5_;
    values["random_mu"] = this->init_;
    values["random_sigma"] = this->sigma_;
    ModelDescription temp;
    temp.ModelParameter = this->integrationMethod->getParameters();
    temp.ModelName = boost::any_cast<std::string>(temp.ModelParameter["name"]);
    values["int_method"] = temp;
    return values;
}

bool HandwritingTimeDrivenModel::compare(NeuronModel* neuralmodel) {
    if (!TimeDrivenModel::compare(neuralmodel)) {
        return false;
    }
    HandwritingTimeDrivenModel* other = dynamic_cast<HandwritingTimeDrivenModel*>(neuralmodel);
    if (other == NULL) {
        return false;
    }
    return this->role_ == other->role_ &&
           this->n_state_ == other->n_state_ &&
           this->v_th_ == other->v_th_ &&
           this->v_reset_ == other->v_reset_ &&
           this->v_spike_ == other->v_spike_ &&
           this->e_ampa_ == other->e_ampa_ &&
           this->e_nmda_ == other->e_nmda_ &&
           this->e_gaba_ == other->e_gaba_ &&
           this->e_leak_ == other->e_leak_ &&
           this->cm_ == other->cm_ &&
           this->g_l_ == other->g_l_ &&
           this->tau_ampa_ == other->tau_ampa_ &&
           this->tau_nmda_ == other->tau_nmda_ &&
           this->tau_gaba_ == other->tau_gaba_ &&
           this->t_ref_ == other->t_ref_ &&
           this->g1_ == other->g1_ &&
           this->g2_ == other->g2_ &&
           this->g3_ == other->g3_ &&
           this->g4_ == other->g4_ &&
           this->g5_ == other->g5_ &&
           this->init_ == other->init_ &&
           this->sigma_ == other->sigma_;
}
