#ifndef INPUT_CONV_DESCRIPTION_H
#define INPUT_CONV_DESCRIPTION_H

#include <boost/any.hpp>
#include <map>
#include <string>
#include <vector>

enum class InputConvOutputTarget {
    MainNetwork,
    DenseSubnetwork,
};

struct InputConvDescription {
    std::string ModelName;
    std::map<std::string, boost::any> ModelParameter;
    int update_timestep = 1;
    int queue_index = 0;

    // InputConv has exactly one output target. DenseSubnetwork uses GPU-to-GPU
    // copy into the target dense runtime; MainNetwork is reserved for the
    // legacy-network input path.
    InputConvOutputTarget output_target = InputConvOutputTarget::MainNetwork;

    // Valid only when output_target is DenseSubnetwork.
    std::string target_dense_subnetwork_name;

    // InputConv output index -> target neuron id in the selected target.
    std::vector<int> output_source_indices;
    std::vector<int> output_target_neuron_ids;

    // Pending-channel route used by dense targets.
    int output_pending_channel = 0;
    float output_scale = 1.0f;
    std::vector<float> output_scales;
    bool output_overwrite = true;
};

#endif
