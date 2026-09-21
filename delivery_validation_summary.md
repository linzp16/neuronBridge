# 发布 wheel 交付验收报告

测试目录：`D:\testfile`

## 全新环境验收

- 使用全新 Python 3.12.4 虚拟环境；
- 不依赖源码目录和 build 目录；
- 使用 `--no-deps` 安装目标 wheel；
- wheel 安装成功；
- `import neuronbridge` 成功；
- native extension、CUDA 和 Dense runtime 成功加载；
- `pip show` 元数据和依赖声明正确；
- 卸载成功，包目录已清理。

## 依赖检查

已安装测试环境执行 `pip check`：

`No broken requirements found.`

## wheel 指纹

SHA-256：

`FC223E412529821BBE29E890129FCB486AEBBCEEC77013485846217AA7C150AE`

## 结果文件

- `clean_validation\clean_import.log`
- `clean_validation\pip_show.txt`
- `clean_validation\uninstall.log`
- `clean_validation\clean_validation.json`
- `logs\pip_check_installed_env.log`

## 当前结论

发布 wheel 已通过当前环境下的功能、压力、通信、CUDA、学习规则、公开示例和交付完整性测试。未发现 wheel 功能性 Bug。

仍未执行的只有依赖外部数据包的 handwriting/EI 完整流程；这些数据集当前不存在于项目目录中。
