# FSMN-VAD 原生后端（C++）

从 `fsmn-vad-tool` 提取的 FSMN-VAD C++ 移植（ggml 运行时，无 funasr/torch 依赖）。
引擎 `modules/vad/engine.py` 优先使用它；DLL 未构建时自动回退到 funasr。

## 当前状态（2026-08-15）

`modules/vad/native/fsmn_vad.dll` **已编译并部署**（MinGW-w64 14.2.0 + CMake 3.30.3，
`GGML_OPENMP=OFF` 禁用 OpenMP，`-static` 静态链接 C/C++ 运行时）。依赖检查：仅
ADVAPI32/KERNEL32 + UCRT（Windows 10/11 自带），**用户无需安装任何运行时**。
`python -c "from modules.vad.native import native_available; print(native_available())"`
返回 `True` 即原生后端就绪。

## 与 funasr 版的对应参数

| 管道参数（不变） | 原生 C API |
|---|---|
| `max_single_segment_time` | `max_seg_ms` |
| `max_end_silence_time`（停顿切句阈值） | `max_end_silence_ms` |

`funasr_vad.h` 已加 `max_end_silence_ms`：>0 时用固定停顿阈值（与
`generate(max_end_silence_time=...)` 一致），<=0 保留自适应调度。

## 构建（需要 CMake + MSVC 或 MinGW）

```bash
cd modules/vad/cpp
build.bat            # Windows 一键（或: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j）
```

构建会通过 FetchContent 拉取 llama.cpp（需网络）。产物：
`build/bin/fsmn_vad.dll` —— 放到以下任一位置即被引擎识别：

1. `modules/vad/cpp/build/bin/fsmn_vad.dll`（源码布局，默认查找位置）
2. `modules/vad/native/fsmn_vad.dll`（打包分发位置，PyInstaller 建议放这）

模型权重 `modules/vad/model/fsmn-vad.gguf`（1.7 MB）已随包内置，无需下载。

## 打包进 EXE

PyInstaller spec 的 `datas` 已包含整个 `modules/` 目录，GGUF 和 cpp 源码自动
进包；`modules.vad.native` 已在 hiddenimports 中。构建出的 DLL 放到
`modules/vad/native/` 后，`engine.py` 的 `_candidate_dll_paths()` 会通过
`sys._MEIPASS` 定位到它。

## 命令行自检（构建后）

```bash
python -c "from modules.vad.native import native_available; print(native_available())"
# True = DLL + GGUF 就绪，VAD 走原生后端（无需 funasr/torch）
```

参考源码：`C:\Users\XTW\Desktop\fsmn-vad-tool`（含 ctypes 调用示例
`vad_native_ctypes.py`）。
