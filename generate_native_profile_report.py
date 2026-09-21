from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    output_dir = Path(sys.argv[1])
    data = json.loads((output_dir / "native_internal_profile.json").read_text(encoding="utf-8"))
    snap = data["snapshot"]
    derived = data["derived_ms"]
    wall = data["wall_seconds"]
    def ms(key: str) -> float:
        return snap[key] / 1_000_000.0

    report = f"""# Native wheel 内部性能剖析报告

## 测试配置

- 运行步数：`{data['steps']}`
- 神经元数：`{data['neurons']}`
- 连接数：`{data['connections']}`
- 学习规则数：`{data['learning_rules']}`
- 运行模式：纯 native wheel，无 CoppeliaSim、无 ZMQ
- 实测 wall time：`{wall:.6f} s`

## 关键计数

- 事件移除次数：`{snap['remove_count']}`
- 事件处理次数：`{snap['run_step_event_count']}`
- TimeEvent 次数：`{snap['time_event_count']}`
- 同步次数：`{snap['sync_count']}`

## 内部阶段耗时

| 阶段 | 累计耗时 |
|---|---:|
| 事件队列调度相关计时 | `{derived['event_queue_scheduling']:.3f} ms` |
| 事件处理总计 | `{derived['event_processing']:.3f} ms` |
| 神经元状态更新 | `{derived['neuron_state_update']:.3f} ms` |
| 突触传播相关计时 | `{derived['synaptic_propagation']:.3f} ms` |
| 学习规则更新 | `{derived['learning_rule_update']:.3f} ms` |
| TimeEvent 重调度 | `{derived['time_event_reschedule']:.3f} ms` |
| 线程同步 | `{ms('sync_ns'):.3f} ms` |

## 与 CoppeliaSim 闭环对比

上一轮闭环测试中，`sim.run()` 耗时约 `83.326 s`，其中 CoppeliaSim 控制、状态读取和轨迹采样约 `79.344 s`。

本次关闭 CoppeliaSim 和 ZMQ 后，同等网络规模只耗时约 `{wall:.3f} s`。因此当前长耗时的主要来源不是 native wheel 内部的事件调度、突触传播或 R-STDP 学习更新，而是外部仿真控制路径。

## 判断

1. 事件调度和神经元状态更新是 native wheel 的主要内部计算项。
2. 突触传播约 `{derived['synaptic_propagation']:.3f} ms`，学习规则更新约 `{derived['learning_rule_update']:.3f} ms`，在纯 native 测试中占比较小。
3. CoppeliaSim 路径中的关节控制、Remote API 状态读取和固定传播等待是主要瓶颈。
4. 各 native 计时器是累计计时，部分阶段存在嵌套关系，不能简单相加作为总耗时。

## 结论

没有证据表明事件调度、突触传播或学习规则更新导致了 80 秒级别的异常延迟。下一步优化重点应放在 CoppeliaSim 控制调用次数、每次调用中的等待时间，以及是否可以批量读取/写入关节状态。
"""
    path = output_dir / "native_internal_profile_report.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
