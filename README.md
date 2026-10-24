# ASMRTranslator Backend

独立 Python 后端的本地拆分预览，包含从桌面项目合并 1.6.0 编排功能后的后端实现。无需桌面前端即可通过命令行或 Python 调用使用。

## 环境与启动

当前随附的 FFmpeg、FFprobe 和 FSMN-VAD DLL 面向 Windows x64；Python 版本为 3.11。安装 uv 后，在本目录执行：

```powershell
uv sync --frozen
uv run python main.py --help
uv run python main.py --check-vad-runtime
uv run python main.py --check-japanese-runtime
uv run python main.py "audio.mp3" "subtitles.vtt" --output "output/demo"
```

字幕输入支持 VTT、SRT 和 LRC。不提供字幕时，可选择 ASR 服务生成字幕。其他模式和参数以 `--help` 为准。

保留的能力包括：ASR、字幕读取与优化、VAD、TTS、通用服务插件适配、配音合成与同步、幻听处理、音频元数据、高音质导出，以及编排快照的编辑和渲染。编排的图形界面位于原桌面项目；本仓库提供 `--arrangement-editor`、`--arrangement-run` 的 stdin JSON 协议与 `modules/arrangement*.py` 实现。

外部 ASR、TTS 和 LLM 服务需要使用者提供自己的凭据、端点或本地服务。服务插件本体由各自项目提供，本仓库保留调用适配器及插件协议文档。

## 配置与日志

配置读取进程环境变量和用户主目录下的 `.asmrts`（`KEY=VALUE` 格式）；已设置的环境变量优先。示例见 `.env.example`，该文件不会自动加载。

本地运行日志、原始输入、过程产物与审计报告保持原有行为。远程错误日志上传默认关闭，代码及测试不包含原维护者的日志服务器地址。需要上传时，由调用方设置：

```powershell
$env:ASMR_LOG_UPLOAD_URL = "https://logs.example.com/log"
# 如果服务要求身份验证，再设置 ASMR_LOG_WRITE_TOKEN。
```

设置 `ASMR_DISABLE_LOG_UPLOAD=1` 可强制关闭上传。上传失败不会改变处理流水线的错误契约。

## 测试与后端打包

```powershell
uv run python -m unittest discover -s tests
uv sync --all-groups --frozen
node scripts/build-nuitka.js
```

打包脚本只生成 Python sidecar，不构建桌面安装包；输出沿用兼容路径 `build_staging_elec/resources/main.exe`。运行该 EXE 仍需通过部署目录或 PATH 提供 FFmpeg/FFprobe；本仓库源码运行使用 `ffmpeg_bin/` 内附程序。日语词典由锁定的 Python 依赖提供，VAD 模型和 Windows DLL 随源码附带。

## 拆分状态

这是全新初始化的本地仓库，来源与范围见 [EXTRACTION.md](EXTRACTION.md)。不含旧 Git 历史、桌面 UI、个人配置、运行日志、review 截图或安装包。

目前尚未创建 GitHub 远端，原项目也尚未切换为 submodule。版本号暂沿用原项目的 1.5.7，1.6.0 指已合入的功能分支，不代表本次发布了 1.6.0。

公开发布前仍需确定本项目许可证，并核对第三方源码、模型和 FFmpeg 二进制的许可及分发材料；本地预览不替这些内容指定许可证。
