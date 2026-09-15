#include "neuron_model/NeuronModelCatalog.h"
#include "source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.h"

#include <boost/any.hpp>

#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        const npgr::NeuronModelCatalog& catalog = npgr::NeuronModelCatalog::Instance();
        Require(!catalog.Entries().empty(), "catalog must contain neuron models");

        const std::map<std::string, boost::any> parameters =
            NeuronModelFactory::QueryDefaultParameters(
                "TimeDrivenLIF_Exponential_double",
                npgr::NeuronBackend::LegacyCpu,
                1,
                0.1f);
        Require(parameters.find("tau") != parameters.end(), "tau parameter is missing");
        Require(parameters.find("int_method") != parameters.end(), "integration method is missing");
        Require(std::abs(boost::any_cast<float>(parameters.at("tau")) - 20.0f) < 1.0e-6f,
                "tau default differs from the model query result");

        const std::map<std::string, boost::any> input_parameters =
            NeuronModelFactory::QueryDefaultParameters(
                "InputSpikeNeuronModel",
                npgr::NeuronBackend::LegacyCpu);
        Require(input_parameters.empty(), "input spike model should report no parameters");

        std::cout << "catalog_models=" << catalog.Entries().size()
                  << " lif_parameters=" << parameters.size() << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
