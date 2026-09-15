"""Handwriting stage-1 numeric helpers mirrored from the C++ example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class ArmParameters:
    link1: float = 0.34
    link2: float = 0.32
    mass1: float = 1.8
    mass2: float = 1.6


@dataclass(frozen=True)
class StrokePath:
    x: list[float]
    y: list[float]


@dataclass(frozen=True)
class JointTrajectory:
    theta1: list[float]
    theta2: list[float]
    dtheta1: list[float]
    dtheta2: list[float]
    ddtheta1: list[float]
    ddtheta2: list[float]


@dataclass(frozen=True)
class TorqueTrajectory:
    q1: list[float]
    q2: list[float]


@dataclass(frozen=True)
class SimulatedTrajectory:
    theta1: list[float]
    theta2: list[float]
    dtheta1: list[float]
    dtheta2: list[float]
    ddtheta1: list[float]
    ddtheta2: list[float]
    hand_path: StrokePath


@dataclass(frozen=True)
class StrokeComputation:
    desired_path: StrokePath
    joints: JointTrajectory
    torques: TorqueTrajectory
    replay: SimulatedTrajectory


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def mean_squared_error(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("mean_squared_error requires equal-length vectors")
    if not lhs:
        return 0.0
    return sum((a - b) * (a - b) for a, b in zip(lhs, rhs)) / len(lhs)


def load_numeric_series(file_path: Path) -> list[float]:
    values: list[float] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip().replace(",", " ").replace("\t", " ")
            if not text:
                continue
            values.extend(float(part) for part in text.split())
    if not values:
        raise ValueError(f"no numeric data found in {file_path}")
    return values


def find_existing_series_path(directory: Path, stem: str) -> Path:
    for extension in (".tsv", ".txt", ".csv"):
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    matlab_file = directory / f"{stem}.mat"
    if matlab_file.exists():
        raise RuntimeError(
            f"found MATLAB file {matlab_file}, but this Python migration reads exported text arrays "
            "(.tsv/.txt/.csv). Run export_stage1_series.m in the same folder first."
        )
    raise FileNotFoundError(f"could not find input series for stem {stem!r} in {directory}")


def load_stroke_path_from_directory(directory: Path, stem: str) -> StrokePath:
    path = StrokePath(
        x=load_numeric_series(find_existing_series_path(directory, f"X{stem}")),
        y=load_numeric_series(find_existing_series_path(directory, f"Y{stem}")),
    )
    if len(path.x) != len(path.y):
        raise ValueError(f"X/Y length mismatch for stroke {stem}")
    return path


def load_torque_sequence(file_path: Path) -> TorqueTrajectory:
    q1: list[float] = []
    q2: list[float] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip().replace(",", " ").replace("\t", " ")
            if not text:
                continue
            parts = text.split()
            if len(parts) < 2:
                continue
            try:
                first = float(parts[0])
                second = float(parts[1])
            except ValueError:
                continue
            q1.append(first)
            q2.append(second)
    if not q1 or len(q1) != len(q2):
        raise ValueError(f"invalid torque sequence in {file_path}")
    return TorqueTrajectory(q1=q1, q2=q2)


def gradient(values: list[float], dt_seconds: float) -> list[float]:
    count = len(values)
    if count == 0:
        return []
    if count == 1:
        return [0.0]
    result = [0.0] * count
    result[0] = (values[1] - values[0]) / dt_seconds
    for index in range(1, count - 1):
        result[index] = (values[index + 1] - values[index - 1]) / (2.0 * dt_seconds)
    result[count - 1] = (values[count - 1] - values[count - 2]) / dt_seconds
    return result


def inverse_kinematics(x: float, y: float, params: ArmParameters) -> tuple[float, float]:
    l1 = params.link1
    l2 = params.link2
    radius_squared = x * x + y * y
    cos_theta2_raw = (radius_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    theta2 = math.acos(clamp(cos_theta2_raw, -1.0, 1.0))

    denom = -2.0 * l1 * math.sqrt(max(radius_squared, 1.0e-12))
    cos_phi_raw = (l2 * l2 - l1 * l1 - radius_squared) / denom
    phi = math.acos(clamp(cos_phi_raw, -1.0, 1.0))
    theta1 = math.atan2(y, x) - phi
    return theta1, theta2


def build_desired_joint_trajectory(path: StrokePath, params: ArmParameters, dt_seconds: float) -> JointTrajectory:
    theta1: list[float] = []
    theta2: list[float] = []
    for x, y in zip(path.x, path.y):
        sample_theta1, sample_theta2 = inverse_kinematics(x, y, params)
        theta1.append(sample_theta1)
        theta2.append(sample_theta2)
    dtheta1 = gradient(theta1, dt_seconds)
    dtheta2 = gradient(theta2, dt_seconds)
    return JointTrajectory(
        theta1=theta1,
        theta2=theta2,
        dtheta1=dtheta1,
        dtheta2=dtheta2,
        ddtheta1=gradient(dtheta1, dt_seconds),
        ddtheta2=gradient(dtheta2, dt_seconds),
    )


def build_desired_torque_trajectory(joints: JointTrajectory, params: ArmParameters) -> TorqueTrajectory:
    l1 = params.link1
    l2 = params.link2
    m1 = params.mass1
    m2 = params.mass2
    d1 = l1 / 2.0
    d2 = l2 / 2.0
    i1 = (m1 * l1 * l1) / 3.0
    i2 = (m2 * l2 * l2) / 12.0

    q1: list[float] = []
    q2: list[float] = []
    for theta2, dtheta1, dtheta2, ddtheta1, ddtheta2 in zip(
        joints.theta2, joints.dtheta1, joints.dtheta2, joints.ddtheta1, joints.ddtheta2
    ):
        c2 = math.cos(theta2)
        s2 = math.sin(theta2)
        q1.append(
            (i1 + i2 + 2.0 * m2 * l1 * d2 * c2 + m1 * d1 * d1 + m2 * (d2 * d2 + l1 * l1)) * ddtheta1
            + (i2 + m2 * l1 * d2 * c2 + m2 * d2 * d2) * ddtheta2
            + (-2.0 * m2 * l1 * d2 * s2) * dtheta1 * dtheta2
            + (-m2 * l1 * d2 * s2) * dtheta2 * dtheta2
        )
        q2.append(
            (i2 + m2 * l1 * d2 * c2 + m2 * d2 * d2) * ddtheta1
            + (i2 + m2 * d2 * d2) * ddtheta2
            + (m2 * l1 * d2 * s2) * dtheta1 * dtheta1
        )
    return TorqueTrajectory(q1=q1, q2=q2)


def joint_acceleration(
    theta2: float,
    dtheta1: float,
    dtheta2: float,
    torque1: float,
    torque2: float,
    params: ArmParameters,
) -> tuple[float, float]:
    l1 = params.link1
    l2 = params.link2
    m1 = params.mass1
    m2 = params.mass2
    d1 = l1 / 2.0
    d2 = l2 / 2.0
    i1 = (m1 * l1 * l1) / 3.0
    i2 = (m2 * l2 * l2) / 12.0

    c2 = math.cos(theta2)
    s2 = math.sin(theta2)
    m11 = i1 + i2 + 2.0 * m2 * l1 * d2 * c2 + m1 * d1 * d1 + m2 * (d2 * d2 + l1 * l1)
    m12 = i2 + m2 * l1 * d2 * c2 + m2 * d2 * d2
    m22 = i2 + m2 * d2 * d2
    b1 = -2.0 * m2 * l1 * d2 * s2
    c12 = -m2 * l1 * d2 * s2
    c21 = m2 * l1 * d2 * s2

    rhs1 = torque1 - b1 * dtheta1 * dtheta2 - c12 * dtheta2 * dtheta2
    rhs2 = torque2 - c21 * dtheta1 * dtheta1
    determinant = m11 * m22 - m12 * m12
    if abs(determinant) < 1.0e-12:
        raise RuntimeError("singular mass matrix encountered in joint_acceleration")
    return (rhs1 * m22 - rhs2 * m12) / determinant, (m11 * rhs2 - m12 * rhs1) / determinant


def hand_position(theta1: list[float], theta2: list[float], params: ArmParameters) -> StrokePath:
    if len(theta1) != len(theta2):
        raise ValueError("theta arrays must have equal length")
    x: list[float] = []
    y: list[float] = []
    for t1, t2 in zip(theta1, theta2):
        x.append(params.link1 * math.cos(t1) + params.link2 * math.cos(t1 + t2))
        y.append(params.link1 * math.sin(t1) + params.link2 * math.sin(t1 + t2))
    return StrokePath(x=x, y=y)


def replay_forward_dynamics(
    desired_joints: JointTrajectory,
    torques: TorqueTrajectory,
    params: ArmParameters,
    dt_seconds: float,
) -> SimulatedTrajectory:
    if len(torques.q1) != len(torques.q2):
        raise ValueError("torque vector length mismatch")
    if not desired_joints.theta1 or not desired_joints.theta2:
        raise ValueError("desired joint trajectory is empty")
    if len(torques.q1) < 2:
        raise ValueError("need at least two torque samples")

    step_count = len(torques.q1) - 1
    theta1 = [desired_joints.theta1[0]] + [0.0] * step_count
    theta2 = [desired_joints.theta2[0]] + [0.0] * step_count
    dtheta1 = [0.0] * (step_count + 1)
    dtheta2 = [0.0] * (step_count + 1)
    ddtheta1 = [0.0] * step_count
    ddtheta2 = [0.0] * step_count

    for index in range(step_count):
        acc1, acc2 = joint_acceleration(
            theta2[index], dtheta1[index], dtheta2[index], torques.q1[index], torques.q2[index], params
        )
        ddtheta1[index] = acc1
        ddtheta2[index] = acc2
        dtheta1[index + 1] = dtheta1[index] + acc1 * dt_seconds
        dtheta2[index + 1] = dtheta2[index] + acc2 * dt_seconds
        theta1[index + 1] = theta1[index] + dtheta1[index] * dt_seconds
        theta2[index + 1] = theta2[index] + dtheta2[index] * dt_seconds

    return SimulatedTrajectory(
        theta1=theta1,
        theta2=theta2,
        dtheta1=dtheta1,
        dtheta2=dtheta2,
        ddtheta1=ddtheta1,
        ddtheta2=ddtheta2,
        hand_path=hand_position(theta1, theta2, params),
    )


def replay_shifted_stroke(
    initial_joint: tuple[float, float],
    torques: TorqueTrajectory,
    params: ArmParameters,
    dt_seconds: float,
) -> SimulatedTrajectory:
    if len(torques.q1) != len(torques.q2):
        raise ValueError("torque vector length mismatch")
    if not torques.q1:
        raise ValueError("torque trajectory is empty")

    step_count = len(torques.q1)
    theta1 = [initial_joint[0]] * (step_count + 1)
    theta2 = [initial_joint[1]] * (step_count + 1)
    dtheta1 = [0.0] * (step_count + 1)
    dtheta2 = [0.0] * (step_count + 1)
    ddtheta1 = [0.0] * step_count
    ddtheta2 = [0.0] * step_count

    for index in range(step_count):
        acc1, acc2 = joint_acceleration(
            theta2[index], dtheta1[index], dtheta2[index], torques.q1[index], torques.q2[index], params
        )
        ddtheta1[index] = acc1
        ddtheta2[index] = acc2
        dtheta1[index + 1] = dtheta1[index] + acc1 * dt_seconds
        dtheta2[index + 1] = dtheta2[index] + acc2 * dt_seconds
        theta1[index + 1] = theta1[index] + dtheta1[index] * dt_seconds
        theta2[index + 1] = theta2[index] + dtheta2[index] * dt_seconds

    return SimulatedTrajectory(
        theta1=theta1,
        theta2=theta2,
        dtheta1=dtheta1,
        dtheta2=dtheta2,
        ddtheta1=ddtheta1,
        ddtheta2=ddtheta2,
        hand_path=hand_position(theta1, theta2, params),
    )


def compute_stroke(path: StrokePath, params: ArmParameters, dt_seconds: float) -> StrokeComputation:
    joints = build_desired_joint_trajectory(path, params, dt_seconds)
    torques = build_desired_torque_trajectory(joints, params)
    replay = replay_forward_dynamics(joints, torques, params, dt_seconds)
    return StrokeComputation(desired_path=path, joints=joints, torques=torques, replay=replay)


def write_two_column_file(file_path: Path, first: list[float], second: list[float], header_a: str, header_b: str) -> None:
    if len(first) != len(second):
        raise ValueError(f"column size mismatch when writing {file_path}")
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{header_a}\t{header_b}\n")
        for a, b in zip(first, second):
            handle.write(f"{a:.17g}\t{b:.17g}\n")


def write_stroke_outputs(output_directory: Path, stem: str, result: StrokeComputation) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    write_two_column_file(output_directory / f"A{stem}.tsv", result.joints.theta1, result.joints.theta2, "theta1", "theta2")
    write_two_column_file(output_directory / f"Q{stem}.tsv", result.torques.q1, result.torques.q2, "torque1", "torque2")
    write_two_column_file(
        output_directory / f"desired_path_{stem}.tsv", result.desired_path.x, result.desired_path.y, "x", "y"
    )
    write_two_column_file(
        output_directory / f"replay_path_{stem}.tsv", result.replay.hand_path.x, result.replay.hand_path.y, "x", "y"
    )
    with (output_directory / f"summary_{stem}.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"path_mse_x\t{mean_squared_error(result.desired_path.x, result.replay.hand_path.x):.17g}\n")
        handle.write(f"path_mse_y\t{mean_squared_error(result.desired_path.y, result.replay.hand_path.y):.17g}\n")
