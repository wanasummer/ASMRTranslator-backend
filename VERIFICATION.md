# 本地拆分验证

验证日期：2026-10-24。

- `uv sync --frozen`：通过，在本目录创建独立 `.venv`，未更改锁文件。
- `.venv/Scripts/python.exe -m unittest discover -s tests`：687 项，成功，9 项跳过（包括独立插件仓库相关测试）。
- 日志上报模块：10 项测试通过，覆盖默认关闭上传、自定义端点、凭据脱敏和上传失败隔离。
- `main.py --help`：通过。
- `main.py --check-vad-runtime`：available=true，DLL 与 GGUF 均来自本仓库。
- `main.py --check-japanese-runtime`：available=true，使用本项目环境中的 MeCab/UniDic Lite。
- FFmpeg/FFprobe：随附程序均能启动。
- `py_compile`：入口、日志模块及新增编排模块通过。
- `node --check scripts/build-nuitka.js`：通过。
- `git diff --cached --check`：通过。
- 对照来源提交，必要后端文件无遗漏；已排除原日志服务地址、文档部署实例地址及本机开发路径。

本次没有构建独立 Nuitka EXE，也没有调用收费服务或进行真实外部 ASR/TTS/LLM 任务。原项目合并版本另已通过后端 686 项、前端 280 项、Rust 98 项测试（Rust 另跳过 1 项）、前端类型检查和生产构建。
