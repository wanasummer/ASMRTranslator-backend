# 后端拆分记录

- 来源项目：ASMRTranslator
- 来源分支：master
- 来源提交：`1c74cd9a03fdc5f88c185f011e323b0da6ca75f9`
- 已合并功能分支：`feature/1.6.0-audio-arrangement`
- 目的：本地独立后端预览，保留原项目完整运行与自动构建流程。

复制 main.py、modules/、tests/、Python 依赖及锁文件、Python 版本声明、后端开发规范、Nuitka 配置及脚本、后端文档、FFmpeg/FFprobe 和 VAD 运行资源。

有意调整：远程错误上报默认地址置空；保留可注入上传器和环境变量配置；测试验证默认不上传与显式配置上传；文档中的部署实例地址替换为通用示例，本机开发路径移除。另外清理复制文件原有的行尾空白，以通过新仓库的 Git 空白检查；其他后端实现保持来源版本。

没有复制：桌面前端、Tauri/Electron 宿主、桌面发布 workflow、旧 Git 历史、.venv、缓存、构建输出、用户数据、运行日志和 review 材料。

公开后端的 main 为当前预览基线，dev 初始指向同一提交。当前主项目尚未引用这个仓库。
