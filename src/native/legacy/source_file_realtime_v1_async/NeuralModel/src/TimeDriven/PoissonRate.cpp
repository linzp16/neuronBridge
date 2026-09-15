#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/PoissonRate.h"

#include "../source_file_realtime_v1_async/Current/inc/CurrentSynapse.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <algorithm>

namespace {

float HashUniform01(int neuron_index, int time_step) {
    unsigned int x = static_cast<unsigned int>(neuron_index) * 747796405u +
                     static_cast<unsigned int>(time_step + 1) * 2891336453u +
                     0x9e3779b9u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

}  // namespace

PoissonRate::PoissonRate() : TimeDrivenModel(), CurrentSynapeModel(0) {
    this->StateVector = new Neuron_State_Vector(this->N_NeuronStateVariables, true);
    this->setModelName(std::string("PoissonRate"));
}

PoissonRate::PoissonRate(int timesteps) : TimeDrivenModel(timesteps), CurrentSynapeModel(0) {
    this->StateVector = new Neuron_State_Vector(this->N_NeuronStateVariables, true);
    this->setModelName(std::string("PoissonRate"));
}

PoissonRate::~PoissonRate() {
    if (this->CurrentSynapeModel != 0) {
        delete this->CurrentSynapeModel;
        this->CurrentSynapeModel = 0;
    }
}

void PoissonRate::InitStateVector(int NumberOfNeurons, int GPUIndex) {
    (void)GPUIndex;
    float init[] = {0.0f, 0.0f};
    float sigma[] = {0.0f, 0.0f};
    this->StateVector->InitNeuronState(NumberOfNeurons, init, sigma);
    this->CurrentSynapeModel = new CurrentSynapse(NumberOfNeurons);
}

Neuron_State_Vector* PoissonRate::InitState() {
    return this->StateVector;
}

void PoissonRate::UpdateState(int index, int time, Simulation* simulation) {
    (void)index;
    this->StateVector->NumberofSpike = 0;
    const int neuron_count = this->StateVector->NumberofNeuron;
    const float dt_ms = simulation != 0 ? simulation->basetimesteps * this->getTimestepSize() : 1.0f;
    for (int neuron = 0; neuron < neuron_count; ++neuron) {
        float* state = this->StateVector->GetNeuronState(neuron);
        float rate_hz = this->rate_bias_hz + state[this->index_rate_hz] +
                        this->rate_gain_hz_per_current * state[this->I_EXT_index];
        rate_hz = std::max(0.0f, rate_hz);
        state[this->index_rate_hz] = rate_hz;
        const float probability = std::min(1.0f, rate_hz * dt_ms * 0.001f);
        if (HashUniform01(neuron, time) < probability) {
            this->StateVector->SpikeIndex[this->StateVector->NumberofSpike] = neuron;
            this->StateVector->NumberofSpike += 1;
            this->StateVector->LastSpike[neuron] = 0;
        } else {
            this->StateVector->LastSpike[neuron] += 1;
        }
        // Spike-driven rate increments are transient; current is kept by the
        // CurrentSynapse accumulator and refreshed through ProcessCurrent.
        state[this->index_rate_hz] = 0.0f;
    }
}

InternalSpike* PoissonRate::ProcessSpike(Interconnections* inter, int time) {
    (void)time;
    if (inter->type == 0) {
        this->StateVector->SetNeuronStateIncrement(inter->TargetNeuronModelIndex, this->index_rate_hz, inter->weight);
    } else if (inter->type == 1) {
        this->StateVector->SetNeuronStateIncrement(inter->TargetNeuronModelIndex, this->index_rate_hz, -inter->weight);
    }
    return 0;
}

void PoissonRate::ProcessCurrent(Interconnections* inter, Neuron* Target, float current) {
    this->CurrentSynapeModel->SetInputCurrentPerSynapse(Target->index_in_NeuronModel, inter->subindex_type, current);
    const float total_current = this->CurrentSynapeModel->GetTotalInputCurrentPerNeuron(Target->index_in_NeuronModel);
    this->StateVector->SetNeuronState(Target->index_in_NeuronModel, this->I_EXT_index, total_current);
}

void PoissonRate::InitializeInputCurrentSynapseStructure() {
    if (this->CurrentSynapeModel != 0) {
        this->CurrentSynapeModel->InitializeInputCurrentPerSynapseStructure();
    }
}

void PoissonRate::CheckType(Interconnections* inter) {
    if (inter->type == 3) {
        inter->subindex_type = this->CurrentSynapeModel->N_connections[inter->TargetNeuron->index_in_NeuronModel];
        this->CurrentSynapeModel->IncrementNInputCurrentSynapsesPerNeuron(inter->TargetNeuron->index_in_NeuronModel);
    }
}

int PoissonRate::getV_index() {
    return this->index_rate_hz;
}

int PoissonRate::get_NumberOfState() {
    return this->N_NeuronStateVariables;
}

enum NeuronModelType PoissonRate::getNeuronModelType() {
    return NEURAL_LAYER;
}

void PoissonRate::SetParameters(std::map<std::string, boost::any> parametermap, float basetimestep) {
    (void)basetimestep;
    std::map<std::string, boost::any>::iterator iter = parametermap.find("rate_bias_hz");
    if (iter != parametermap.end()) {
        this->rate_bias_hz = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
    iter = parametermap.find("poisson_rate_bias_hz");
    if (iter != parametermap.end()) {
        this->rate_bias_hz = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
    iter = parametermap.find("rate_gain_hz_per_current");
    if (iter != parametermap.end()) {
        this->rate_gain_hz_per_current = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
    iter = parametermap.find("poisson_rate_gain_hz_per_current");
    if (iter != parametermap.end()) {
        this->rate_gain_hz_per_current = boost::any_cast<float>(iter->second);
        parametermap.erase(iter);
    }
}

std::map<std::string, boost::any> PoissonRate::getParameters() {
    std::map<std::string, boost::any> parametermap;
    parametermap["rate_bias_hz"] = this->rate_bias_hz;
    parametermap["rate_gain_hz_per_current"] = this->rate_gain_hz_per_current;
    return parametermap;
}
