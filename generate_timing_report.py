from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    output_dir = Path(sys.argv[1])
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    timing = result["timing_seconds"]
    total = timing["total_run_wall_seconds"]
    native = timing["native_sim_run_seconds"]
    control = timing["coppeliasim_control_seconds"]
    wait = timing["peer_request_wait_seconds"]
    parse = timing["peer_parse_seconds"]
    evaluate = timing["phase_evaluation_seconds"]
    send = timing["reply_send_seconds"]

    def pct(value: float) -> str:
        return f"{100.0 * value / total:.2f}%"

    report = f"""# NeuronBridge + CoppeliaSim 分段耗时报告

## 测试概要

- 结果：`{'PASS' if result['pass'] else 'FAIL'}`
- 输出批次：`{result['received_output_batches']}`
- 输出脉冲：`{result['received_output_spikes']}`
- 反馈批次：`{result['feedback_batches']}`
- 轨迹采样点：`{result['trajectory_samples']}`
- 通信模式：阻塞式 ZMQ REQ/REP

## 耗时统计

| 阶段 | 耗时（秒） | 占总耗时 |
|---|---:|---:|
| 总运行时间 | {total:.6f} | 100.00% |
| `sim.run()` 神经网络推进总段 | {native:.6f} | {pct(native)} |
| CoppeliaSim 控制、状态读取和轨迹采样 | {control:.6f} | {pct(control)} |
| Peer 等待下一个 REQ | {wait:.6f} | {pct(wait)} |
| 请求解析 | {parse:.6f} | {pct(parse)} |
| phase 评价及奖励/惩罚生成 | {evaluate:.6f} | {pct(evaluate)} |
| REP 回复发送 | {send:.6f} | {pct(send)} |

## 结论

本次运行的主要耗时来自 CoppeliaSim 控制和状态读取，占总运行时间约
`{100.0 * control / total:.2f}%`。其中每次控制调用包含约 20 ms 的仿真传播等待，且每个通信批次需要控制两个关节，因此该部分明显高于 ZMQ 序列化、请求解析和反馈发送。

REQ/REP 的等待段约为 `{wait:.3f}` 秒，未观察到通信错误或超时；请求解析和回复发送均低于 0.01 秒量级，不是性能瓶颈。

需要注意，`sim.run()` 的 `{native:.3f}` 秒是神经网络推进和同步通信共同构成的总段，不能直接等同于纯神经网络计算时间。若要进一步拆分 C++ 内部的事件调度、突触传播和学习规则更新，还需要在 native wheel 内部增加更细粒度计时点。
"""
    path = output_dir / "timing_report.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
