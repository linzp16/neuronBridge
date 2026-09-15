#ifndef NPGR_DENSE_INTERFACE_SPIKE_NEURON_MODEL_H
#define NPGR_DENSE_INTERFACE_SPIKE_NEURON_MODEL_H

#include "source_file_realtime_v1_async/NeuralModel/inc/EventDriven/InputSpikeNeuronModel.h"

namespace npgr {

class DenseSubnetworkModel;

class DenseInterfaceSpikeNeuronModel : public InputSpikeNeuronModel {
public:
    DenseInterfaceSpikeNeuronModel();
    explicit DenseInterfaceSpikeNeuronModel(DenseSubnetworkModel* owner);
    ~DenseInterfaceSpikeNeuronModel();

    void SetOwner(DenseSubnetworkModel* owner);
    DenseSubnetworkModel* owner() const;

    virtual InternalSpike* ProcessSpike(Interconnections* inter, int time);
    virtual void UpdateState(int index, int time, Simulation* simulation);
    virtual bool compare(NeuronModel* neuronModel);
    virtual std::map<std::string, boost::any> getParameters();

    void ResetInterfaceBufferState(int interface_buffer_slot_count);
    bool AccumulateInterfaceSpikeDelta(int interface_buffer_slot_index,
                                       float weight,
                                       bool inhibitory,
                                       int time_step,
                                       std::string* reason = nullptr);
    bool HasPendingInterfaceBufferForTimeStep(int time_step) const;
    void ClearInterfaceStateBuffers();

private:
    DenseSubnetworkModel* owner_;
    int interface_buffer_slot_count_;
    int last_buffered_time_step_;
    bool has_pending_interface_buffer_;
};

}  // namespace npgr

#endif
