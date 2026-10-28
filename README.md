<h1 align="center">ASMRTranslator Backend</h1>

<p align="center"><strong>日文 ASMR → 中文配音</strong></p>
<p align="center">可独立运行、可接入桌面应用的 Python 音频处理后端</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/package_manager-uv-DE5FE9?style=flat-square&amp;logo=uv&amp;logoColor=white" alt="uv"></a>
  <a href="https://ffmpeg.org/"><img src="https://img.shields.io/badge/audio-FFmpeg-007808?style=flat-square&amp;logo=ffmpeg&amp;logoColor=white" alt="FFmpeg"></a>
  <a href="#快速开始"><img src="https://img.shields.io/badge/platform-Windows_x64-0078D4?style=flat-square" alt="Windows x64"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code_license-Noncommercial-F59E0B?style=flat-square" alt="Code License: PolyForm Noncommercial 1.0.0"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#处理流程">处理流程</a> ·
  <a href="#配置与日志">配置与日志</a> ·
  <a href="#开发与打包">开发与打包</a> ·
  <a href="#许可">许可</a>
</p>

---

支持可选字幕、语音识别、LLM 翻译润色、多种 TTS、音频同步、编排与导出。

## 快速开始

> **运行环境**：Windows x64 · Python 3.11 · uv。仓库已随附 Windows 版 FFmpeg、FFprobe 和 VAD 运行资源。

```powershell
uv sync --frozen
uv run python main.py "audio.mp3" "subtitles.vtt" --output "./output/demo"
```

字幕支持 **VTT / SRT / LRC**；省略字幕时需配置 ASR 及翻译服务。外部服务的地址、API Key 和音色由使用者自行提供。

<details>
<summary>查看全部命令行参数</summary>

```powershell
uv run python main.py --help
```

</details>

## 处理流程

```text
  声音（原始音频）字幕可选
    ↓ 字幕准备（二选一）
     ├─ 有字幕 → parse_vtt 解析
     └─ 无字幕 → VAD 检测人声 → 贪心切片 → ASR 多线程转写（标记时间线）
    ↓ LLM 模块（唯一一次，切音之前）
     ├─ ASR 路径 → 摘要 + 翻译切句 + 按句分配时间 + 润色
     └─ 有字幕路径 → 摘要 + 润色
    ↓ split 切分（按时间戳切原音频成句子片段）
    ↓ TTS 模块（并行逐句配音）
    ↓ 音频处理（防爆音/phantom DSP → 响度对齐）
    ↓ 听音模式（替换/保留/同步/潜意识）
    ↓ 拼接导出（follow_translation 截断 → join → concat → MP3+WAV）
```

## 配置与日志

### 外部人声检测与强制对齐

桌面设置中的“人声检测与时间对齐”支持本地 VAD 和独立对齐服务：

| 插件 | 默认地址 | 用途 |
| --- | --- | --- |
| ASMR VAD | 本地进程，无需地址 | 使用 Whisper VAD ONNX 检测日语耳语的人声区间 |
| Qwen Force Align | `http://127.0.0.1:8007` | 使用 Qwen3 ForcedAligner 为日语原文预测词时间戳并组合句子 |

安装包由插件仓库提供，可在插件安装器中导入本地 ZIP。ASMR VAD 需要 0.2.0 或更高版本：安装完成后自动出现在设置选项中，未安装或模型、运行环境缺失时隐藏。主程序从用户配置目录的 `ASMRTranstor/plugins.json` 读取安装位置，使用插件自带的 Python 通过 `asmr-vad-local/1` 标准输入输出协议离线推理，不需要启动插件管理器、HTTP 服务或日志网关。保留完整安装根目录（包括依赖环境和托管 Python）及安装记录即可持续使用；主程序重启后自动重新发现，任务内重复检测复用已加载的模型进程。Qwen Force Align 仍需启动服务，再扫描本地端口或填写局域网地址。

CLI 对应 `--vad-backend asmr-vad`、`--vad-threshold`，以及 `--detect-stage --sentence-aligner qwen-force-align --qwen-align-endpoint`。`--vad-endpoint` 为旧配置兼容保留，本地 VAD 忽略该地址。默认仍使用内置 FSMN VAD，句子对齐默认关闭。桌面端的句子对齐用于 Fish ASR 或通用 ASR + 大模型翻译；后端对齐阶段在清洗、翻译与切音之前运行。异常时间戳、文本不匹配或服务故障会保留原片段，并写入 `detect_output/sentence_alignment.json`。已有字幕不调用这些插件，海南鸡专属 VAD 开启时不调用项目 VAD。

模型和运行依赖均保存在独立插件环境中，主程序不需要安装 PyTorch、ONNX Runtime 或 qwen-asr。

### MOSS 转写与说话人区分

“ASR 模型”可选择独立的 MOSS Transcribe Diarize 插件，默认地址 `http://127.0.0.1:8008`。可扫描插件安装器中的本机端口或配置远程地址，再测试服务连接。CLI 使用 `--asr-backend moss-transcribe-diarize --asr-kwargs '{"endpoint":"http://127.0.0.1:8008","language":"ja"}'`。

MOSS 返回整段音频的原生时间戳和 `S01` 等录音内说话人标记，跳过项目 VAD 与句界对齐；日语文本进入既有清洗与翻译流程。原始响应保存到 `asr_subtitles/raw_response.json`，清洗字幕和缓存恢复保留说话人。接口单次音频最长 30 分钟，生成截断或不完整时间轴会明确失败。模型和 Transformers 5 运行环境由插件独立管理。

用户配置沿用用户目录下的 `.asmrts`，已有环境变量优先。

| 配置项 | 说明 |
| --- | --- |
| 本地日志与过程产物 | 保存在任务输出目录 |
| `ASMR_LOG_UPLOAD_URL` | 远程错误上报地址；源码未配置时默认为空 |
| `ASMR_DISABLE_LOG_UPLOAD=1` | 关闭远程错误上报 |

正式构建可将 Workflow 提供的地址嵌入 EXE，运行时环境变量仍可覆盖。日志上报不需要 token。

## 开发与打包

**运行测试**

```powershell
uv run python -m unittest discover -s tests
```

**构建后端**（需要 Node.js）

```powershell
uv sync --all-groups --frozen
node scripts/build-nuitka.js
```

产物：`build_staging_elec/resources/main.exe`；桌面项目可通过构建环境变量 `ASMR_SIDECAR_RESOURCE_DIR` 指定产物目录。部署时还需提供 `ffmpeg_bin/` 中的 FFmpeg、FFprobe。

| 位置 | 用途 |
| --- | --- |
| [`main.py`](main.py) | 命令行入口 |
| [`modules/pipeline.py`](modules/pipeline.py) | 流程编排 |
| [`modules/`](modules/) | 各处理阶段的实现 |
| [`tests/`](tests/) | 单元测试与契约验证 |

## 许可

项目原创代码采用 [PolyForm Noncommercial 1.0.0](LICENSE)，仅授权许可规定的非商业用途，禁止未经另行授权的商业使用。修改和再分发须遵守许可条款并保留必要声明。

本项目属于**源码可用（Source Available）**，不属于 OSI 定义的开源软件。第三方代码、模型、字典及 FFmpeg 等资源保留各自许可，详见 [第三方声明](THIRD_PARTY_NOTICES.md)。

[贡献指南](CONTRIBUTING.md) · [安全报告](SECURITY.md)
