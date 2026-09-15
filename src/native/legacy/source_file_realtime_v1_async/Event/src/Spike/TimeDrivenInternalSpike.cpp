#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuronModelPropogationStructure.h"
#include "../source_file_realtime_v1_async/Neuron/inc/Neuron.h"
#include "../source_file_realtime_v1_async/Event/inc/Spike/TimeDrivenInternalSpike.h"
#include "../source_file_realtime_v1_async/Event/inc/Spike/PropogatedSpikeGroup.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "../source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include "../source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "../source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "../source_file_realtime_v1_async/mainprogram/inc/BenchProfiling.h"
TimeDrivenInternalSpike::TimeDrivenInternalSpike(int time, int QueueIndex, Neuron_State_Vector* State, NeuronModelPropogationStructure* ModelPropogationStructure, Neuron** neuron_array)
    : InternalSpike(NULL, time, QueueIndex), neuron_state_vector(State), neuronModelPropogationStructure(ModelPropogationStructure), Neurons(neuron_array) {
    this->propogatedSpikeGroup = new PropogatedSpikeGroup**[NumberOfOpenMPQueues];
    for (int i = 0; i < NumberOfOpenMPQueues; i++) {
        this->propogatedSpikeGroup[i] = new PropogatedSpikeGroup*[this->neuronModelPropogationStructure->NumberOfDelays[i]];
        for (int j = 0; j < this->neuronModelPropogationStructure->NumberOfDelays[i]; j++) {
            this->propogatedSpikeGroup[i][j] = new PropogatedSpikeGroup(time + this->neuronModelPropogationStructure->SynapseDelay[i][j], i);
        }
    }
}
TimeDrivenInternalSpike::~TimeDrivenInternalSpike() {
    for (int i = 0; i < NumberOfOpenMPQueues; i++) {
        for (int j = 0; j < this->neuronModelPropogationStructure->NumberOfDelays[i]; j++) {
            if (this->propogatedSpikeGroup[i][j] != NULL) {
                delete this->propogatedSpikeGroup[i][j];
                this->propogatedSpikeGroup[i][j] = NULL;
            }
        }
        delete[] this->propogatedSpikeGroup[i];
    }
    delete[] this->propogatedSpikeGroup;
    this->propogatedSpikeGroup = NULL;
}
void TimeDrivenInternalSpike::ProcessInternalSpikeEvent(Simulation* simulation, int index) {
    this->ProcessInternalSpikeEvent(simulation, index, ALL_EVENTS_ENABLED);
}
void TimeDrivenInternalSpike::ProcessInternalSpikeEvent(Simulation* simulation, int index, RealTimeRestrictionLevel level) {
    Neuron* neuron = this->Neurons[index];
    this->SourceNeuron = neuron;
    long long phase_start_ns = bench_profile::now_ns();
    simulation->WriteSpike(this);
    bench_profile::internal_spike_write_spike_ns.fetch_add(
        bench_profile::now_ns() - phase_start_ns, std::memory_order_relaxed);
    bool maxflag;
    for (int i = 0; i < NumberOfOpenMPQueues; i++) {
        for (int j = 0; j < neuron->PropogationStructure->NDifferentdelays[i]; j++) {
            int index_in_neuronmodel = neuron->PropogationStructure->SynapseDelayIndex[i][j];
            phase_start_ns = bench_profile::now_ns();
            maxflag = this->propogatedSpikeGroup[i][index_in_neuronmodel]->IncludeNewSourceNeuron(neuron->PropogationStructure->NInterconnections[i][j], neuron->PropogationStructure->interconnections[i][j]);
            bench_profile::internal_spike_include_ns.fetch_add(
                bench_profile::now_ns() - phase_start_ns, std::memory_order_relaxed);
            this->propogatedSpikeGroup[i][index_in_neuronmodel]->SourceNeuron = neuron;
            if (maxflag) {
                phase_start_ns = bench_profile::now_ns();
                if (i == this->getIndex()) {
                    simulation->EventHeap->Insert_a_Event(this->propogatedSpikeGroup[i][index_in_neuronmodel], i);
                }
                else {
                    simulation->EventHeap->Insert_a_Event_to_Buffer(this->propogatedSpikeGroup[i][index_in_neuronmodel], this->getIndex(), i);
                }
                bench_profile::internal_spike_insert_ready_ns.fetch_add(
                    bench_profile::now_ns() - phase_start_ns, std::memory_order_relaxed);
                phase_start_ns = bench_profile::now_ns();
                this->propogatedSpikeGroup[i][index_in_neuronmodel] = new PropogatedSpikeGroup(this->propogatedSpikeGroup[i][index_in_neuronmodel]->getTime(), i);
                bench_profile::internal_spike_rotate_group_ns.fetch_add(
                    bench_profile::now_ns() - phase_start_ns, std::memory_order_relaxed);
            }
        }
    }
    if (level < LEARNING_RULES_DISABLED) {
        int LearningRuleNum = simulation->network->LearningRuleNum;
        phase_start_ns = bench_profile::now_ns();
        for (int i = 0; i < LearningRuleNum; i++) {
            if (neuron->PostSynapticLearning_Number[i] > 0) {
                neuron->PostSynapticLearning[i][0]->LearningRule_withPost->ApplyPostSynaticSpike(neuron, this->getTime(), simulation);
            }
            if (neuron->TriggerAndPostSynapticLearning_Number[i] > 0) {
                neuron->TriggerAndPostSynapticLearning[i][0]->LearningRule_withPostAndTrigger->ApplyPostSynaticSpike(neuron, this->getTime(), simulation);
            }
        }
        bench_profile::internal_spike_learning_ns.fetch_add(
            bench_profile::now_ns() - phase_start_ns, std::memory_order_relaxed);
    }
}
void TimeDrivenInternalSpike::ProcessEvent(Simulation* simulation) {
    this->ProcessEvent(simulation, ALL_EVENTS_ENABLED);
}
void TimeDrivenInternalSpike::ProcessEvent(Simulation* simulation, RealTimeRestrictionLevel level) {
    if (level >= SPIKES_DISABLED) {
        return;
    }
    if (this->neuron_state_vector->IsGPU == false) {
        int* SpikingGenerateIndex = this->neuron_state_vector->SpikeIndex;
        int N_SpikingGenerateIndex = this->neuron_state_vector->NumberofSpike;
        for (int i = 0; i < N_SpikingGenerateIndex; i++) {
            this->ProcessInternalSpikeEvent(simulation, SpikingGenerateIndex[i], level);
        }
    }
    else {
        const long long gpu_scan_start_ns = bench_profile::now_ns();
        bool* GPUInternalSpike = this->neuron_state_vector->getInternalSpike();
        if (this->neuron_state_vector->IsMonitored) {
            for (int i = 0; i < this->neuron_state_vector->NumberofNeuron; i++) {
                if (GPUInternalSpike[i] == true) {
                    GPUInternalSpike[i] = false;
                    this->ProcessInternalSpikeEvent(simulation, i, level);
                }
            }
        }
        else {
            for (int i = 0; i < this->neuron_state_vector->NumberofNeuron; i++) {
                if (GPUInternalSpike[i] == true) {
                    GPUInternalSpike[i] = false;
                    this->ProcessInternalSpikeEvent(simulation, i, level);
                }
            }
        }
        bench_profile::gpu_internal_spike_scan_ns.fetch_add(
            bench_profile::now_ns() - gpu_scan_start_ns,
            std::memory_order_relaxed);
    }
    const long long finalize_start_ns = bench_profile::now_ns();
    for (int i = 0; i < NumberOfOpenMPQueues; i++) {
        for (int j = 0; j < this->neuronModelPropogationStructure->NumberOfDelays[i]; j++) {
            if (this->propogatedSpikeGroup[i][j]->N_Elements > 0) {
                if (i == this->getIndex()) {
                    simulation->EventHeap->Insert_a_Event(this->propogatedSpikeGroup[i][j], i);
                }
                else {
                    simulation->EventHeap->Insert_a_Event_to_Buffer(this->propogatedSpikeGroup[i][j], this->getIndex(), i);
                }
                this->propogatedSpikeGroup[i][j] = NULL;
            }
            else {
                delete this->propogatedSpikeGroup[i][j];
                this->propogatedSpikeGroup[i][j] = NULL;
            }
        }
    }
    bench_profile::internal_spike_finalize_group_ns.fetch_add(
        bench_profile::now_ns() - finalize_start_ns, std::memory_order_relaxed);
}
enum EventPriority TimeDrivenInternalSpike::getPriority() {
    return INTERNALSPIKE;
}
