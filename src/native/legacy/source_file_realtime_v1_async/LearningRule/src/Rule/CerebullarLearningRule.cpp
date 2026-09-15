#include "../source_file_realtime_v1_async/LearningRule/inc/Rule/CerebullarLearningRule.h"
#include "../source_file_realtime_v1_async/LearningRule/inc/SpikeBuffer/BufferedActivityTime.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include <cmath>
#include <cstdint>

namespace {

float StableUnitRandom(unsigned int seed, int rule_id, int synapse_id, int spike_time) {
	std::uint32_t value = static_cast<std::uint32_t>(seed);
	value ^= static_cast<std::uint32_t>(rule_id + 0x9e3779b9u);
	value *= 1664525u;
	value += 1013904223u;
	value ^= static_cast<std::uint32_t>(synapse_id + 0x85ebca6bu);
	value *= 2246822519u;
	value ^= static_cast<std::uint32_t>(spike_time + 0xc2b2ae35u);
	value ^= value >> 16;
	value *= 0x7feb352du;
	value ^= value >> 15;
	value *= 0x846ca68bu;
	value ^= value >> 16;
	return static_cast<float>(value & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

void ApplyBoundedWeightIncrement(Interconnections* connection, float increment, float min_weight) {
	if (connection == 0) {
		return;
	}
	connection->weight += increment;
	if (connection->weight > connection->maximum_weight) {
		connection->weight = connection->maximum_weight;
	}
	else if (connection->weight < min_weight) {
		connection->weight = min_weight;
	}
}

}  // namespace

CerebullarLearningRule::CerebullarLearningRule(std::map<std::string, boost::any> parametermap) : AdditiveKernalChange(), NoneTriggerSpikeBuffer(0), kernaltable(0) {
	this->SetParameters(parametermap);
}

CerebullarLearningRule::~CerebullarLearningRule() {
	if (this->NoneTriggerSpikeBuffer != 0) {
		delete this->NoneTriggerSpikeBuffer;
	}
	if (this->kernaltable != 0) {
		delete[] this->kernaltable;
	}
}

void CerebullarLearningRule::InitState(int NumberOfConnections, int NumberOfNeuron, float basetimestep) {
	(void)NumberOfConnections;
	(void)basetimestep;
	if (this->maxpos <= this->initpos) {
		this->maxpos = this->initpos + 1e-6f;
	}

	const float kernel_width = this->maxpos - this->initpos;
	this->maxtimemeasured = this->maxpos;
	while (true) {
		const float value = (1.0f / kernel_width) * this->maxtimemeasured * std::exp(-(this->maxtimemeasured / kernel_width) + 1.0f);
		if (value < 1e-2f) {
			break;
		}
		this->maxtimemeasured += this->kernel_step_size;
	}
	this->maxtimemeasured += this->initpos;
	this->inv_kernel_step_size = 1.0f / this->kernel_step_size;
	this->N_elements = static_cast<int>(this->maxtimemeasured * this->inv_kernel_step_size) + 1;
	this->kernaltable = new float[this->N_elements];

	for (int i = 0; i < this->N_elements; ++i) {
		const float time = i * this->kernel_step_size;
		if (time < this->initpos) {
			this->kernaltable[i] = 0.0f;
		}
		else {
			const float shifted_time = time - this->initpos;
			this->kernaltable[i] = (1.0f / kernel_width) * shifted_time * std::exp(-(shifted_time / kernel_width) + 1.0f);
		}
	}

	this->NoneTriggerSpikeBuffer = new BufferedActivityTime(NumberOfNeuron);
}

void CerebullarLearningRule::ApplyPreSynaticSpike(Interconnections* connection, int SpikeTime, Simulation* simulation) {
	const float realtime = SpikeTime * simulation->basetimesteps;
	if (connection->TriggerLearning == false) {
		// Compromise with the legacy hand-written cerebellar update:
		// a normal GC/PF pre-spike applies alpha * U(0,1), while the later
		// CF trigger still performs the beta/kernel update.
		const float pre_random = StableUnitRandom(
			this->random_seed,
			this->LearningRuleID,
			connection->Index,
			SpikeTime);
		ApplyBoundedWeightIncrement(connection, this->a1pre * pre_random, this->min_weight);
		const int neuronID = connection->TargetNeuron->Neuron_index;
		const int synapseID = connection->LearningRuleIndex_withTrigger_Target;
		this->NoneTriggerSpikeBuffer->InsetElement(neuronID, realtime, realtime - this->maxtimemeasured, synapseID);
	}
	else {
		Neuron* neuron = connection->TargetNeuron;
		const int neuronID = neuron->Neuron_index;
		const int N_spikes = this->NoneTriggerSpikeBuffer->ProcessElement(neuronID, realtime - this->maxtimemeasured);
		SpikeData* spikedata = this->NoneTriggerSpikeBuffer->GetOutputSpikeData();
		for (int i = 0; i < N_spikes; ++i) {
			Interconnections* interi = neuron->TriggerSynapticLearning[this->LearningRuleID][spikedata[i].synapticID];
			const float elapsetime = realtime - spikedata[i].time;
			int tableindex = static_cast<int>(elapsetime * this->inv_kernel_step_size);
			if (tableindex < 0) {
				continue;
			}
			if (tableindex >= this->N_elements) {
				tableindex = this->N_elements - 1;
			}
			// CF/teaching spikes consume buffered GC/PF spikes and apply the
			// trigger-side beta/kernel term to the buffered plastic synapse.
			ApplyBoundedWeightIncrement(interi, this->a2prepre * this->kernaltable[tableindex], this->min_weight);
		}
	}
}

std::map<std::string, boost::any> CerebullarLearningRule::GetParameters() {
	std::map<std::string, boost::any> parameters = AdditiveKernalChange::GetParameters();
	parameters["initpos"] = this->initpos;
	parameters["maxpos"] = this->maxpos;
	parameters["min_weight"] = this->min_weight;
	parameters["random_seed"] = static_cast<int>(this->random_seed);
	return parameters;
}

void CerebullarLearningRule::SetParameters(std::map<std::string, boost::any> parameters) {
	std::map<std::string, boost::any>::iterator iter = parameters.find("initpos");
	if (iter != parameters.end()) {
		this->initpos = boost::any_cast<float>(iter->second);
	}
	iter = parameters.find("maxpos");
	if (iter != parameters.end()) {
		this->maxpos = boost::any_cast<float>(iter->second);
	}
	iter = parameters.find("kernel_step_size");
	if (iter != parameters.end()) {
		this->kernel_step_size = boost::any_cast<float>(iter->second);
	}
	iter = parameters.find("min_weight");
	if (iter != parameters.end()) {
		this->min_weight = boost::any_cast<float>(iter->second);
	}
	iter = parameters.find("minimum_weight");
	if (iter != parameters.end()) {
		this->min_weight = boost::any_cast<float>(iter->second);
	}
	iter = parameters.find("w_gcpc_min");
	if (iter != parameters.end()) {
		this->min_weight = boost::any_cast<float>(iter->second);
	}
	iter = parameters.find("random_seed");
	if (iter != parameters.end()) {
		this->random_seed = static_cast<unsigned int>(boost::any_cast<int>(iter->second));
	}
	AdditiveKernalChange::SetParameters(parameters);
}
