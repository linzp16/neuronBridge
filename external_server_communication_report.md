# 外部服务器同步/异步通信测试报告

日期：2026-09-17

测试 wheel：`neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`

测试环境：`D:\testfile\venv_bugfix`，服务器地址为本机回环地址 `127.0.0.1`，使用随机可用 TCP 端口。

## 测试结果

| 测试项 | 结果 | 结果摘要 |
|---|---|---|
| Python 同步 REQ/REP | PASS | 100/100 请求成功，100/100 payload 校验正确，0 failed_requests |
| native 同步外部服务器 | PASS | 41/41 请求收到并返回，0 provider_errors，0 dropped_frames |
| native 异步 PUB/SUB | PASS | 20/20 帧收到并消费，0 dropped_frames，最后时间步 19 |
| 生命周期关闭 | PASS | client、server、publisher 均正确关闭 |

同步通信使用 `InputConvFrameServer`/`InputConvFrameClient` 与 native REQ/REP source；异步通信使用 `InputConvFramePublisher` 与 native SUB source，并显式绑定到 InputConv 通道。

完整 JSON 结果：`external_server_communication_results.json`。
