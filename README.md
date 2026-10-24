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
