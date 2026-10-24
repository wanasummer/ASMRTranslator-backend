# 贡献指南

提交问题时，请提供版本、运行环境、复现步骤和已脱敏的错误信息。不要提交真实凭据或未获授权的音频。

## 本地开发

使用 Python 3.11，在仓库根目录执行：

```powershell
uv sync --all-groups --frozen
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

遵循 [AGENTS.md](AGENTS.md) 的模块边界和验证要求。网络及外部服务在单元测试中应使用 mock；修复问题时补充能体现实际行为的测试。

修改依赖后更新 `uv.lock`，提交前运行相关测试及 `git diff --check`。提交信息采用 Conventional Commits，例如 `fix(tts): preserve segment timing`。

Pull Request 请说明问题、改动和验证结果。新增或复制第三方代码、数据、模型时，注明来源并保留原始许可；提交的原创贡献按本项目 [PolyForm Noncommercial 1.0.0 许可证](LICENSE) 授权。
