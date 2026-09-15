#ifndef NPGR_INPUT_CONV_DEBUG_SOURCE_H
#define NPGR_INPUT_CONV_DEBUG_SOURCE_H

#include "debug_monitor/IDebugMonitorSource.h"

class InputConvModel;

namespace npgr {

class InputConvDebugSource : public IDebugMonitorSource {
public:
    InputConvDebugSource(InputConvModel* model, int inputconv_index, const std::string& name);

    DebugComponentKind kind() const override;
    int component_index() const override;
    const std::string& component_name() const override;
    bool HasAnyTarget(const DebugMonitorConfig& config) const override;
    bool Capture(int time_step,
                 const DebugMonitorConfig& config,
                 DebugMonitorFrame* frame,
                 std::string* reason) override;

private:
    bool IsSelected(const DebugMonitorConfig& config) const;

    InputConvModel* model_;
    int inputconv_index_;
    std::string name_;
};

}  // namespace npgr

#endif
