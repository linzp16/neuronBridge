#include "../source_file_realtime_v1_async/ModelFactory/InputConvModelFactory.h"

#include "input_conv/InputConvModelCatalog.h"
#include "../source_file_realtime_v1_async/InputConv/inc/InputConvV1.h"

#include <map>
#include <stdexcept>
#include <string>

namespace {

typedef InputConvModel* (*InputConvCreator)(const InputConvDescription&);

#define NPGR_INPUT_CONV_MODEL(model_name, model_class)                 \
InputConvModel* Create##model_class(const InputConvDescription& description) { \
    return new model_class(description.update_timestep);               \
}
#include "../source_file_realtime_v1_async/ModelFactory/InputConvModelList.inc"
#undef NPGR_INPUT_CONV_MODEL

std::map<std::string, InputConvCreator> BuildInputConvRegistry() {
    std::map<std::string, InputConvCreator> registry;
    // Built-in InputConv models are registered from one list so the public
    // factory stays stable as new visual front-end models are added.
#define NPGR_INPUT_CONV_MODEL(model_name, model_class) \
    registry[model_name] = &Create##model_class;
#include "../source_file_realtime_v1_async/ModelFactory/InputConvModelList.inc"
#undef NPGR_INPUT_CONV_MODEL
    return registry;
}

const std::map<std::string, InputConvCreator>& InputConvRegistry() {
    static const std::map<std::string, InputConvCreator> registry =
        BuildInputConvRegistry();
    return registry;
}

}  // namespace

InputConvModel* InputConvModelFactory::createInputConvModel(const InputConvDescription& description) {
    std::string catalog_reason;
    const std::string implementation_name =
        npgr::InputConvModelCatalog::Instance().ResolveImplementationName(
            description.ModelName,
            &catalog_reason);
    if (implementation_name.empty()) {
        throw std::runtime_error(catalog_reason);
    }
    const bool target_is_dense =
        description.output_target == InputConvOutputTarget::DenseSubnetwork;
    if (!npgr::InputConvModelCatalog::Instance().ValidateOutputTarget(
            description.ModelName,
            target_is_dense,
            &catalog_reason)) {
        throw std::runtime_error(catalog_reason);
    }
    const std::map<std::string, InputConvCreator>& registry = InputConvRegistry();
    std::map<std::string, InputConvCreator>::const_iterator found =
        registry.find(implementation_name);
    if (found != registry.end() && found->second != 0) {
        return found->second(description);
    }
    throw std::runtime_error("unregistered input conv implementation: " + implementation_name);
}
