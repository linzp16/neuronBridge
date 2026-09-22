# NeuronBridge 未覆盖接口补测报告

日期：2026-09-21

## 被测对象

- Wheel：`D:\neuronBridge\artifacts\wheel_weight_api_fix\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- SHA-256：`0F6EB0E02B65886590C860BE8D55469A5D29A53CB5AB90F5313E2C366BA9066F`
- 隔离环境：`D:\neuronBridge\artifacts\wheel_weight_api_fix_testvenv`

## 新增固定测试

- `.nbnet` batch、单连接、abort、is_valid、重复 finalize 和异常清理。
- 参数类型及 InputConv 像素格式/payload helper。
- DebugMonitorResult metadata、路径和 pandas 委托。
- DebugMonitor disable/re-enable 生命周期。
- InputConv queue clear 和 monitor 动态开关。
- InputConvFramePublisher 批量发布、receiver 等待成功和超时。
- 权重快照错误 header、截断、网络不匹配、非有限值以及加载原子性。
- 学习后 reset 在快速/流式路径保持权重，并与 DebugMonitor 组合运行。
- OuterDynamic reset/desired state 和错误参数。
- 真实异步 spike ZMQ PUB/SUB 端到端。

## 重新执行的专项测试

- 7 种学习规则 × 快速/流式构建：全部通过。
- 3 类 OuterDynamic × 快速/流式构建：全部通过。
- 10 类公开绘图：全部通过并生成 PNG。
- triple 神经元 CPU、legacy GPU、dense GPU：全部通过。
- 同步 spike REQ/REP：heap 与 TimeWheel 均通过；请求窗口为 5、10、15、20、25。
- 异步 spike PUB/SUB：真实输入、输出、topic 和 `publish_output()` 通过。

异步 PUB/SUB 测试需要等待订阅握手。第一次一次性运行在握手不足时外部 SUB 收到 0 batch；将测试固定为 1 秒握手后稳定通过。这是 ZeroMQ PUB/SUB slow-joiner 行为，当前未发现 native 消息格式或发布实现错误。

## 回归结果

- 扩展核心回归：`92 passed, 18 deselected`。
- 排除：Handwriting、`ei_connectivity`。
- JUnit：`D:\neuronBridge\artifacts\wheel_weight_api_fix_validation\pytest_interface_expansion.xml`
- 学习/OuterDynamic/绘图结果：`D:\neuronBridge\artifacts\wheel_weight_api_fix_validation\learning_outer_plot\learning_outer_plot_report.json`
- 三后端结果：`D:\neuronBridge\artifacts\wheel_weight_api_fix_validation\triple_backend.json`

## 暂缓开放与尚未完成

- 9 个实时运行和实时统计接口统一标记为“暂缓开放（Deferred Experimental）”，不属于当前稳定支持范围，也不阻塞本版本发布。
- `to_pandas()` 只完成接口委托测试；真实 pandas 集成仍被当前 NumPy/pandas ABI 冲突阻塞。
- 大规模流式构建内存基准和完整小脑/CoppeliaSim 长时间闭环尚未在该 wheel 上重跑。
