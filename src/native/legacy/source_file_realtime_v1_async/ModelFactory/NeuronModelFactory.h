#ifndef NEURONMODELFACTORY_H
#define NEURONMODELFACTORY_H

#include <boost/any.hpp>

#include <map>
#include <string>

#include "neuron_model/NeuronModelCatalog.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/NeuronModel.h"

class NeuronModelFactory {
public:
    static NeuronModel* createNeuronModel(
        const std::string& type,
        std::map<std::string, boost::any> NeuronParameter,
        int timestepsize,
        float basetimestepsize,
        int QueueIndex);

    static std::map<std::string, boost::any> QueryDefaultParameters(
        const std::string& type,
        npgr::NeuronBackend backend,
        int timestepsize = 1,
        float basetimestepsize = 1.0f);

    static std::map<std::string, boost::any> QueryParameters(
        const std::string& type,
        npgr::NeuronBackend backend,
        const std::map<std::string, boost::any>& parameters,
        int timestepsize = 1,
        float basetimestepsize = 1.0f);
};

#endif
