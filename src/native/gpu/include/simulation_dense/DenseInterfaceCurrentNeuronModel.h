#ifndef NPGR_DENSE_INTERFACE_CURRENT_NEURON_MODEL_H
#define NPGR_DENSE_INTERFACE_CURRENT_NEURON_MODEL_H

#include "source_file_realtime_v1_async/NeuralModel/inc/EventDriven/InputCurrentNeuronModel.h"

namespace npgr {

class DenseSubnetworkModel;

class DenseInterfaceCurrentNeuronModel : public InputCurrentNeuronModel {
public:
    DenseInterfaceCurrentNeuronModel();
    explicit DenseInterfaceCurrentNeuronModel(DenseSubnetworkModel* owner);
    ~DenseInterfaceCurrentNeuronModel();

    void SetOwner(DenseSubnetworkModel* owner);
    DenseSubnetworkModel* owner() const;

    virtual void ProcessCurrent(Interconnections* inter, Neuron* Target, float current);
    virtual bool compare(NeuronModel* neuronModel);
    virtual std::map<std::string, boost::any> getParameters();

private:
    DenseSubnetworkModel* owner_;
};

}  // namespace npgr

#endif
