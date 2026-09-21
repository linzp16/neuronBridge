#pragma once

#include <array>

#include "target_sequence_generation_config.h"

// This is the generated-header-compatible fallback for the supplied source
// tree. It can be replaced by GenerateDualRingTargetSequence after the
// CoppeliaSim IK scene is available. The array length is exactly the number of
// input neurons, so input neuron i is active in phase i.
namespace generated_targets {

inline constexpr int INPUT_GROUP_COUNT = target_sequence_config::SEQUENCE_LENGTH;
inline constexpr int RING_COUNT = 2;
inline constexpr int LIF_COUNT_PER_RING = target_sequence_config::LIF_COUNT_PER_RING;

inline constexpr std::array<float, RING_COUNT> RING_TARGET_MIN_ANGLES = {
    -2.7925268f, -2.7925268f};
inline constexpr std::array<float, RING_COUNT> RING_TARGET_MAX_ANGLES = {
    2.7925268f, 2.7925268f};

// One entry per input neuron / phase. Values are one-based ring slots.
inline constexpr std::array<int, INPUT_GROUP_COUNT> RING0_TARGET_SEQUENCE_SLOTS_ONE_BASED = {
    8, 12, 8, 4, 8};
inline constexpr std::array<int, INPUT_GROUP_COUNT> RING1_TARGET_SEQUENCE_SLOTS_ONE_BASED = {
    8, 4, 8, 12, 8};
inline constexpr std::array<int, INPUT_GROUP_COUNT> INPUT_PHASE_STEPS = {
    target_sequence_config::PHASE_STEPS_PER_INPUT,
    target_sequence_config::PHASE_STEPS_PER_INPUT,
    target_sequence_config::PHASE_STEPS_PER_INPUT,
    target_sequence_config::PHASE_STEPS_PER_INPUT,
    target_sequence_config::PHASE_STEPS_PER_INPUT};

}  // namespace generated_targets
