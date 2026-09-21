#pragma once

// Reconstructed configuration for the supplied attractor project.
// The original generated configuration headers were absent from H:\\attractor.
// One input group is intentionally assigned to exactly one phase.
namespace target_sequence_config {

inline constexpr bool TRAINING_ENABLED = true;

// Five input neurons/groups, hence five phases.
inline constexpr int SEQUENCE_LENGTH = 5;

// Reconstructed ring resolution used by the restored build.
inline constexpr int LIF_COUNT_PER_RING = 16;

inline constexpr int SHOULDER_JOINT_INDEX = 0;
inline constexpr int ELBOW_JOINT_INDEX = 1;
inline constexpr int SHOULDER_HANDLE = 18;
inline constexpr int ELBOW_HANDLE = 20;

// One phase per input neuron; each phase lasts 1000 simulation steps.
inline constexpr int PHASE_STEPS_PER_INPUT = 1000;
inline constexpr int INITIAL_SETTLE_STEPS = 100;
inline constexpr int SETTLE_STEPS_PER_SAMPLE = 10;

// Figure-eight target parameters from MTB_Rob_Server::computeFigureEightTarget.
inline constexpr float CENTER_X = 0.25f;
inline constexpr float CENTER_Y = 0.20f;
inline constexpr float CENTER_Z = 0.20f;
inline constexpr float RADIUS_X = 0.14f;
inline constexpr float RADIUS_Y = 0.03f;
inline constexpr float Z_AMPLITUDE = 0.0f;

}  // namespace target_sequence_config
