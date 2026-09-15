#ifndef NPGR_DENSE_NEURON_DEVICE_VIEWS_H
#define NPGR_DENSE_NEURON_DEVICE_VIEWS_H

namespace npgr {

struct DeviceCommonBuffersView {
    // Channel-major pending input pool:
    // d_pending_channels[channel * pending_channel_stride + neuron].
    float* d_pending_channels = nullptr;
    int pending_channel_count = 0;
    int pending_channel_stride = 0;
    int neuron_count = 0;
};

}  // namespace npgr

#endif
