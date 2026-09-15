#include "../source_file_realtime_v1_async/ModelFactory/OuterDynamicModelFactory.h"

#if SNN_WITH_PINOCCHIO
#include "../source_file_realtime_v1_async/OuterDynamic/inc/PlanarArm2DOFPinocchio.h"
#endif
#include "../source_file_realtime_v1_async/OuterDynamic/inc/StrictMatlabPlanarArm2DOFOuterDynamic.h"
#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeCounter.h"

#include <map>
#include <stdexcept>
#include <string>

namespace {

typedef OuterDynamicModel* (*OuterDynamicCreator)(const OuterDynamicDescription&);

OuterDynamicModel* CreateStrictMatlabPlanarArm2DOFOuterDynamic(
    const OuterDynamicDescription& description) {
    return new StrictMatlabPlanarArm2DOFOuterDynamic(description.update_timestep);
}

OuterDynamicModel* CreateOuterDynamicSpikeCounter(
    const OuterDynamicDescription& description) {
    return new OuterDynamicSpikeCounter(description.update_timestep);
}

#if SNN_WITH_PINOCCHIO
OuterDynamicModel* CreatePlanarArm2DOFPinocchio(
    const OuterDynamicDescription& description) {
    return new PlanarArm2DOFPinocchio(description.update_timestep);
}
#endif

std::map<std::string, OuterDynamicCreator> BuildOuterDynamicRegistry() {
    std::map<std::string, OuterDynamicCreator> registry;
    // Built-in models are registered in one place so adding a model does not
    // grow the public factory entry point into another long string branch.
#if SNN_WITH_PINOCCHIO
    registry["PlanarArm2DOFPinocchio"] = &CreatePlanarArm2DOFPinocchio;
#endif
    registry["StrictMatlabPlanarArm2DOFOuterDynamic"] =
        &CreateStrictMatlabPlanarArm2DOFOuterDynamic;
    registry["OuterDynamicSpikeCounter"] = &CreateOuterDynamicSpikeCounter;
    return registry;
}

const std::map<std::string, OuterDynamicCreator>& OuterDynamicRegistry() {
    static const std::map<std::string, OuterDynamicCreator> registry =
        BuildOuterDynamicRegistry();
    return registry;
}

}  // namespace

OuterDynamicModel* OuterDynamicModelFactory::createOuterDynamicModel(const OuterDynamicDescription& description) {
    const std::map<std::string, OuterDynamicCreator>& registry = OuterDynamicRegistry();
    std::map<std::string, OuterDynamicCreator>::const_iterator found =
        registry.find(description.ModelName);
    if (found != registry.end() && found->second != 0) {
        return found->second(description);
    }
    throw std::runtime_error("Unknown or unavailable outer dynamic model: " + description.ModelName);
}
