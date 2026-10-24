# 第三方许可说明

根目录的 [PolyForm Noncommercial 1.0.0 许可证](LICENSE) 适用于本项目原创代码，不会替代第三方代码、模型、字典、依赖或可执行文件的原始许可。各文件中已有的版权和许可声明继续有效。

## 随附源码与运行资源

| 内容 | 来源及许可 | 许可文本 |
| --- | --- | --- |
| `ffmpeg_bin/ffmpeg.exe`、`ffprobe.exe` | FFmpeg 8.1.2，Gyan essentials build；程序自报 GPL-3.0-or-later，启用 `--enable-gpl --enable-version3` | [GPL v3](LICENSES/GPL-3.0.txt) |
| FSMN-VAD 原生实现中的 FunASR 移植代码 | [FunASR](https://github.com/modelscope/FunASR/tree/main/runtime/llama.cpp)，MIT；本仓库保留本地适配，未记录提取时的精确上游提交 | [FunASR MIT](LICENSES/FunASR-MIT.txt) |
| 原生 VAD 使用的 ggml / llama.cpp | [llama.cpp](https://github.com/ggml-org/llama.cpp)，CMake 固定提交 `8086439a4cea94c71a5dfb8fe4ad1546aebd640f`，MIT | [ggml MIT](LICENSES/llama.cpp-MIT.txt) |
| `modules/vad/cpp/miniaudio.h` | miniaudio 0.11.25，David Reid；可选 Public Domain 或 MIT-0 | 完整声明保留在该文件末尾 |
| `modules/vad/model/fsmn-vad.gguf` | 下载脚本指向 [FunAudioLLM/fsmn-vad-GGUF](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF)，其模型卡标注 Apache-2.0 | [Apache 2.0](LICENSES/Apache-2.0.txt) |
| `model.pt`、`am.mvn` 及 VAD 配置 | 本地配置记录原始模型为 `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`；这些资源不由本项目许可证重新授权，发布前应核对原始下载包的许可与来源 | 以原模型发布方的许可为准 |

## Python 依赖

全部依赖和版本以 `pyproject.toml`、`uv.lock` 为准，各包及其数据保留自身许可。下列项目尤其需要在打包时保留许可声明；本表不是完整的传递依赖清单。

| 依赖 | 当前锁定版本 | 许可与来源 |
| --- | --- | --- |
| Mutagen | 1.48.1 | GPL-2.0-or-later；[上游](https://github.com/quodlibet/mutagen)，[许可文本](LICENSES/GPL-2.0.txt) |
| pykakasi | 2.3.0 | GPL-3.0-or-later；[上游](https://codeberg.org/miurahr/pykakasi)，[许可文本](LICENSES/GPL-3.0.txt) |
| unidic-lite | 1.0.8 | 包装代码 [MIT](LICENSES/unidic-lite-MIT.txt)；字典提供 [BSD 条款](LICENSES/UniDic-BSD-3-Clause.txt)，另见安装包的字典许可文件 |
| mecab-python3 | 1.0.12 | [安装包 COPYING](LICENSES/mecab-python3-COPYING.txt) 提供 GPL/LGPL/BSD 选择；同时附上 [MeCab BSD 条款](LICENSES/MeCab-BSD.txt) |

## 分发边界

- PolyForm Noncommercial 仅说明本项目原创代码的授权，不为第三方组件增加非商业限制，也不能据此将完整依赖组合或安装包宣称为统一采用本项目许可证。
- 当前依赖包含 Mutagen、pykakasi 等 GPL 组件。GPL 所覆盖的组合作品不能附加禁止商业使用的限制；若组合分发构成 GPL 所要求的衍生或组合作品，应先解决与本项目非商业许可证的兼容性，不能仅靠保留声明满足要求。
- 再分发 FFmpeg 等 GPL 组件时，应满足其适用的源码提供、版权与许可通知等条件。仅增加本文件不代表二进制分发义务已经全部履行。参考 [FFmpeg 官方许可说明](https://ffmpeg.org/legal.html) 和 [Gyan 构建来源](https://www.gyan.dev/ffmpeg/builds/)。
- 分发打包后的 Python 依赖时，也需保留实际包含组件的许可、版权和适用的源码材料。
- 模型和原生 DLL 的精确来源、构建依赖及来源未记录的资源，应在公开发布前补齐核验；本文件不为无法核验的资源新增授权。
- 用户输入的音频、字幕及外部服务生成的内容不因本仓库公开源码 而自动获得使用或传播许可。
