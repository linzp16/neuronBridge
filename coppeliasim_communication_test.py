"""通信专项入口：运行已有 CoppeliaSim 闭环并生成通信验收报告。

该脚本不改变机械臂控制逻辑，只把现有
``coppeliasim_attractor_closed_loop.py`` 的结果转换为通信专项指标。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("coppeliasim_attractor_closed_loop.py")),
        "--output-dir", str(output_dir),
        "--host", args.host,
        "--port", str(args.port),
        "--phases", str(args.phases),
        "--communication-interval", str(args.communication_interval),
        "--epochs", str(args.epochs),
    ]
    if args.reference_delivery:
        command.append("--reference-delivery")
    joint_paths = args.joint_path or ["/MTB/axis", "/MTB/link/axis"]
    for joint_path in joint_paths:
        command.extend(["--joint-path", joint_path])

    completed = subprocess.run(command, capture_output=True, text=True)
    (output_dir / "closed_loop_stdout.txt").write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    timing = result.get("timing_seconds", {})
    request_count = int(timing.get("request_count", 0))
    feedback_batches = int(result.get("feedback_batches", 0))
    communication_errors = result.get("peer_error")
    checks = {
        "remote_api_peer_started": communication_errors is None,
        "req_rep_requests_observed": request_count > 0,
        "feedback_returned": feedback_batches > 0,
        "phase_control_executed": int(result.get("phase_control_count", 0)) > 0,
        "trajectory_sampled": int(result.get("trajectory_samples", 0)) > 0,
        "no_peer_error": communication_errors is None,
    }
    report = {
        "test": "CoppeliaSim closed-loop communication",
        "command": command,
        "config": {
            "host": args.host,
            "remote_api_port": args.port,
            "zmq_req_rep_port": 5565,
            "phases": args.phases,
            "communication_interval": args.communication_interval,
            "epochs": args.epochs,
            "joint_paths": joint_paths,
        },
        "protocol": {
            "wheel_to_peer": "ZMQ synchronous REQ/REP",
            "request_header": "<II: time_step, output_spike_count>",
            "spike_record": "<iff: neuron_id, spike_time, base_timestep>",
            "reply": "<I: feedback_spike_count> followed by <iff> records",
            "peer_to_coppeliasim": f"CoppeliaSim ZMQ Remote API {args.host}:{args.port}",
        },
        "communication_metrics": {
            "req_rep_request_count": request_count,
            "feedback_batch_count": feedback_batches,
            "received_output_batches": result.get("received_output_batches", 0),
            "received_output_spikes": result.get("received_output_spikes", 0),
            "feedback_spikes": result.get("ring_spikes_from_zmq", 0),
            "remote_api_error": communication_errors,
            "peer_request_wait_seconds": timing.get("peer_request_wait_seconds", 0.0),
            "peer_parse_seconds": timing.get("peer_parse_seconds", 0.0),
            "coppeliasim_control_seconds": timing.get("coppeliasim_control_seconds", 0.0),
            "reply_send_seconds": timing.get("reply_send_seconds", 0.0),
            "total_run_wall_seconds": timing.get("total_run_wall_seconds", 0.0),
        },
        "checks": checks,
        "pass": completed.returncode == 0 and all(checks.values()),
        "source_result": result,
    }
    (output_dir / "communication_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = [
        "# CoppeliaSim 闭环通信测试报告",
        "",
        f"结论：**{'PASS' if report['pass'] else 'FAIL'}**",
        "",
        "## 通信链路",
        "",
        "```text",
        "NeuronBridge wheel -- ZMQ REQ/REP --> Python Peer",
        "Python Peer -- ZMQ Remote API --> CoppeliaSim",
        "CoppeliaSim -- joint/end-effector feedback --> Python Peer",
        "Python Peer -- reward/punishment spike --> NeuronBridge wheel",
        "```",
        "",
        "## 指标",
        "",
        f"- REQ/REP 请求数：{request_count}",
        f"- 反馈批次：{feedback_batches}",
        f"- 输出 spike 批次：{result.get('received_output_batches', 0)}",
        f"- 输出 spike 数：{result.get('received_output_spikes', 0)}",
        f"- 反馈 spike 数：{result.get('ring_spikes_from_zmq', 0)}",
        f"- 阶段控制次数：{result.get('phase_control_count', 0)}",
        f"- 轨迹采样点：{result.get('trajectory_samples', 0)}",
        f"- Remote API 错误：{communication_errors or '无'}",
        "",
        "## 验收项",
        "",
    ]
    markdown.extend(f"- {'通过' if value else '失败'}：{key}" for key, value in checks.items())
    markdown.extend([
        "",
        "详细原始结果见 `result.json`，通信专项结构化结果见 `communication_report.json`。",
    ])
    (output_dir / "communication_report.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--phases", type=int, default=64)
    parser.add_argument("--communication-interval", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--reference-delivery", action="store_true")
    parser.add_argument("--joint-path", action="append", default=None)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
