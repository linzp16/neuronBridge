"""Configuration extracted from delivery_cpu_only_final_bundle.

This is kept as Python data so the training scripts do not depend on external
configuration files. The target slots are the generated IK sequence from the
successful CPU-only delivery bundle.
"""

INPUT_GROUPS = 100
PHASES = 100
RING_SIZE = 32
PHASE_STEPS = 3000
COMMUNICATION_INTERVAL = 1000
TIMESTEP_MS = 0.1

RING_ANGLE_RANGES = (
    (0.384012, 0.868843),
    (-1.89266, -0.855545),
)

RING0_TARGET_SLOTS_ONE_BASED = [
    19,18,18,17,16,16,15,14,14,13,12,11,10,10,9,8,7,6,6,5,
    4,3,3,2,2,2,1,1,1,1,1,1,2,2,2,3,4,4,5,6,7,8,9,10,11,12,
    14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,30,31,31,32,
    32,32,32,32,32,32,31,31,31,30,30,29,29,28,28,27,26,26,25,25,
    24,24,23,23,23,22,22,21,21,21,20,20,19,19,
]

RING1_TARGET_SLOTS_ONE_BASED = [
    14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,29,30,30,
    31,31,32,32,32,32,32,32,32,31,31,31,30,29,29,28,27,26,26,25,
    24,23,22,21,20,19,18,17,16,15,14,13,12,11,11,10,9,8,8,7,6,6,
    5,4,4,3,3,3,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,3,3,3,4,4,5,6,6,7,
    8,8,9,10,11,11,12,13,
]

RING_TARGET_SLOTS = (
    [slot - 1 for slot in RING0_TARGET_SLOTS_ONE_BASED],
    [slot - 1 for slot in RING1_TARGET_SLOTS_ONE_BASED],
)

assert len(RING0_TARGET_SLOTS_ONE_BASED) == PHASES
assert len(RING1_TARGET_SLOTS_ONE_BASED) == PHASES
