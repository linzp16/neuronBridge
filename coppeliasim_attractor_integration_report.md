# CoppeliaSim 接入说明

已新增 CoppeliaSim 控制端适配器：

`D:\\testfile\\coppeliasim_attractor_closed_loop.py`

## 角色

- NeuronBridge wheel：异步 ZMQ SNN 端，发布输出 spike、订阅输入/reward/punishment spike。
- `CoppeliaPeer`：Python 控制端，订阅 SNN 输出，映射到两个关节目标位置，读取实际关节位置，按误差发送 reward 或 punishment。
- CoppeliaSim：通过 `RemoteAPIClient` 连接 `localhost:23000`，默认寻找 `/MTB/joint1`、`/MTB/joint2`。

## 启动条件

当前环境不能完成真实运行验收，原因如下：

1. `H:\\attractor` 没有 `.ttt` 或 `.ttm` 场景文件。
2. 当前 Python venv 没有 `coppeliasim_zmqremoteapi_client` 或 `zmqRemoteApi` 模块。
3. 当前机器没有运行中的 CoppeliaSim 进程。

适配器支持两种常见 Python Remote API 模块名：

```powershell
python coppeliasim_attractor_closed_loop.py `
  --output-dir D:\\testfile\\coppeliasim_attractor_run `
  --port 23000 `
  --joint-path /MTB/joint1 `
  --joint-path /MTB/joint2
```

如果场景中的对象路径不同，替换 `--joint-path`。启动顺序建议为：

1. 打开 CoppeliaSim，加载 MTB 场景。
2. 启用 ZMQ Remote API 服务端口 23000。
3. 确认场景中存在两个关节和目标对象。
4. 将 CoppeliaSim Python Remote API client 加入 `PYTHONPATH`。
5. 运行上述适配器脚本。

## 数据闭环

```text
wheel 输出 spike
        ↓ ZMQ PUB/SUB
CoppeliaPeer → setJointTargetPosition
        ↓
CoppeliaSim MTB 动力学
        ↓ getJointPosition
误差判定 → reward/punishment spike
        ↓ ZMQ PUB/SUB
wheel R-STDP 更新
```

当前默认目标角序列为两个关节的 4 个阶段目标，仅用于验证接口；真实任务应替换为 `H:\\attractor` 缺失的生成目标序列。

## 真实场景运行结果（2026-09-18）

已使用 `H:\\attractor\\MTB_scene.ttt` 启动 CoppeliaSim 4.10 headless 场景，并确认：

- Remote API：`localhost:23000`，连接成功。
- 原工程句柄 `18/20` 对应 `/MTB/axis`、`/MTB/link/axis`。
- wheel 输出批次：81。
- wheel 输出 spike：8。
- CoppeliaSim 返回反馈批次：4/4。
- 通信错误：0。
- reward/punishment：0/4。

这次结果证明 wheel—CoppeliaSim 的通信和反馈通道已接通，但还不能判定机器人控制任务成功。检查发现两个目标关节在场景中为 `jointmode_passive=0`，并且场景脚本会保持/重置关节状态；`setJointTargetPosition` 没有形成实际运动轨迹。因此适配器对 passive 关节使用了 `setJointPosition` 作为接口验证 fallback，但最终场景仍未产生满足目标角的真实轨迹。

下一步应在场景中将 `/MTB/axis`、`/MTB/link/axis` 配置为可控的动力学/位置伺服关节，或修改 `/MTB/Script` 的控制逻辑，使其不覆盖 Remote API 的目标位置；完成后再以 reward 非零和实际关节轨迹为验收条件。

## 平滑轨迹验证（2026-09-18）

适配器已增加逐周期平滑位置推进，并生成：

`D:\\testfile\\coppeliasim_trajectory_run\\joint_trajectory.csv`

结果：

- 轨迹采样点：82
- `/MTB/axis`：41 个采样，位置范围 `0.036–0.144 rad`
- `/MTB/link/axis`：41 个采样，位置范围 `0.054–0.126 rad`
- 最大单步位置变化：`0.144 rad`

因此当前场景已经形成非零、连续变化的关节轨迹。由于场景关节仍是 passive 模式，这一结果是运动学轨迹验证，不等同于动力学伺服控制；后续切换关节模式后应重新验证力、速度、加速度和 reward 收敛。

## 八字训练闭环复核

已将规划器改为真实 MTB 的 `x-y` 平面解析 IK，并执行 16 阶段训练：

- wheel 输出批次：297
- 反馈批次：16/16
- reward/punishment：5/11
- Remote API 错误：0

但最终末端实测轨迹的 `x`、`y` 范围仍为 0，说明当前 `.ttt` 的场景脚本/关节控制覆盖了 Remote API 写入。该结果不能作为“机械臂已沿八字形运动”的验收通过，只能证明 wheel 训练通信和误差反馈链路已工作。

要完成真实八字运动，必须修改场景：将 `/MTB/axis`、`/MTB/link/axis` 配置为位置伺服/动力学可控关节并停止脚本覆盖，或补充原工程假设的 `/MTB/IK`、`/MTB/ikTarget` 和 `computeIk` 接口。完成后再以末端 `x-y` 范围非零、轨迹闭合度和 reward 收敛作为最终验收指标。

## 场景脚本闭环修复结果（2026-09-18）

已修改 `/MTB/Script`，并保存副本：

`H:\\attractor\\MTB_scene_wheel_closed_loop.ttt`

修改内容：

- 新增 Remote API 可调用函数 `setExternalJointTargets(joints)`。
- 新增 external-control 状态和逐仿真步插值。
- external-control 开启时跳过原 `simMTB.step()` 的关节覆盖逻辑。
- wheel 适配器通过场景脚本接口发送两个主关节目标，而不是跨过场景脚本直接写关节。

真实训练闭环结果：

- wheel 输出批次：297
- 输出 spike：8
- 反馈批次：16/16
- reward/punishment：4/12
- Remote API 错误：0
- 末端轨迹采样点：64
- 末端 `x`：`0.3309–0.4418 m`
- 末端 `y`：`0.6795–0.7639 m`

轨迹数据：

`D:\\testfile\\coppeliasim_trained_scene_run\\joint_trajectory.csv`

本次已形成非零的末端空间运动，并且动作目标经过场景脚本的仿真步插值；reward/punishment 已返回 wheel。由于原场景没有 `/MTB/IK` 和 `/MTB/ikTarget`，当前八字路径使用场景实测几何的解析二维 IK 目标，后续如补齐原工程的 IK 对象，可替换为原项目的 `computeIk` 路径。
## 末端轨迹图

最终训练闭环运行的末端实测轨迹已绘制并保存为：

- `D:\testfile\coppeliasim_trained_scene_run\end_effector_trajectory.png`
- `D:\testfile\coppeliasim_trained_scene_run\end_effector_trajectory.svg`

图中左侧为按控制阶段着色的末端 `x-y` 轨迹，右侧为 `x/y/z` 随轨迹采样点的变化。数据来自同一轮成功运行的 `joint_trajectory.csv`，共 64 个采样点；末端运动范围为 `x=0.110894682 m`、`y=0.084334448 m`。
