# 通用 TTS 插件接入

通用插件使用独立 HTTP 服务，通过 `common-tts/1` 契约对接。已有的 Edge、
MiMo、Fish、MiniMax Adapter 保持原实现。Genie 与 IndexTTS 2.5 已迁入通用 TTS 插件，
前端不再提供它们的独立引擎设置、专用参数面板或旧 Genie ONNX 音色导入接口。

## 使用插件

1. 在插件安装器中导入并安装插件。
2. 主程序设置页选择「通用 TTS 插件（自动识别）」，自动读取已安装插件的类型、端口和参数，无需手动导入 YAML。
3. 多个插件可在列表中切换；安装完成后返回设置页会自动刷新，也可点击「重新识别插件」。启动插件并等待模型就绪后，点击「检测并启用」。
4. 音频工作台选择通用 TTS 插件；插件参数、音色和音高等合成配置仅在工作台按照插件参数 Schema 渲染，设置页只负责识别与连接检测。

本机识别读取安装器的 `plugins.json`，从安装目录的 `plugin-manifest.json`
确认 `apiVersion: common-tts/1`，再读取 `plugin.yml`。损坏的描述会显示错误，
不会影响其他插件。通用 ASR 同样根据 `common-asr/1` 自动识别，不依赖固定插件名称。
服务器模式填写服务地址与可选 API Key，点击「重新识别插件」通过 `/manifest`
读取参数描述，然后检测并启用。

YAML 和合成参数会进入项目配置与任务快照。连接地址和 API Key 保存在应用配置中。
设置页选择的插件描述与地址共同决定当前服务。打开项目时，仅在项目描述与当前服务描述一致时
恢复项目参数；插件不同或版本变化时使用当前插件默认参数，避免旧 Audio8 配置覆盖新选择的 Genie。
旧的 `genie-plugin` / `indextts-2.5` 独立引擎配置会退回只生成字幕，需重新选择通用插件并检测启用；旧参数不自动转换。
插件描述变更会重置参数并要求重新检测；描述未变时重新识别会保留参数与验证状态。
安装器分配的端口变更后也需重新检测；已排队任务继续使用原有快照。
嵌套对象有分组表单，数组使用 JSON 编辑器，音频字段使用本地文件选择器。
条件显示不替代校验：模式与参数之间的约束写在 Schema 的 `if/then` 中。

工作台通过独立的 `PluginConfigRenderer`（含 `ParameterField`、`ParameterGroup`、`ParameterToggle`）
组件递归渲染参数，由 `PluginParameterFields` 注入桌面能力。布尔字段使用小型开关，
不套用工作台 `.compact-fields input` 的通用输入框尺寸。样式回归预览位于
`desktop-tauri/tests/fixtures/plugin-config/index.html`，启动 Vite 后可直接访问。

## 开发插件

插件仓库的 `common-tts/Audio8-TTS/` 提供可运行的参考实现；完整接入指南位于
插件仓库的 `common-tts/README.md`。插件目录采用：

```text
source/        官方源码、许可证、模型
inject/        对接实现、控制器、参数校验、安装和日志
plugin.yml     合成参数与能力声明
app.py         安装器启动入口
install.py     安装入口
```

实现 `inject.engine.load_engine()`，返回实现 `Engine` 的对象。核心方法
`synthesize(request, runtime)` 接收逐段中文、说话人、时间信息、
原声音频、多个参考素材和有效配置，返回完整 PCM WAV。模型算法和上游调用放在
`inject/engine.py` 或其子模块中，直接调用 `source/`；不要向 `app.py` 堆积实现。

参考 [plugin.example.yml](plugin.example.yml)。该示例覆盖逐段音色、独立音色参考、
逐段情感、情感音频、向量、文字描述以及嵌套采样参数，但它不代表已经安装了
IndexTTS 模型。Audio8 的实际模型参数以它自己的 `plugin.yml` 为准；
复制接入层时须替换模型实现、安装资源和能力声明。

HTTP 契约和扩展边界见 [PROTOCOL.md](PROTOCOL.md)。本版支持逐段完整 WAV、
参数校验、文件上传、音色列表和任务日志。跨段会话、远端取消、音色管理及宿主
实时音频流需要后续可选接口；能力描述不能提前声称已经支持。

## 宿主执行边界

`split_audio` 返回每段的 `speech_file`、`segment_file`、文字和时间信息。
`TTSSynthesisRequest.from_segment` 复制这些返回值，不读取流水线其他内部对象。
`CommonTTSAdapter` 在独立段上下文中解析参数、上传素材、关联日志并接收结果；
`plugin_writer` 校验音频和保存临时产物，TTS 阶段完成后才原子发布。
伪影检测后的重生成也绑定同一段输入。

输出缓存包含插件描述、有效参数、时间/上下文和音频内容摘要。随机引擎可以用
`execution.cache: false` 禁用合成缓存复用。参数中 `x-secret` 字段在诊断中脱敏，
不会因名字包含 token 而误丢失 `max_mel_tokens` 等影响合成的参数。

桌面通过 stdin 把通用插件配置传给 Python，避免 Windows 命令行长度限制，
也不把通用插件 API Key 放到进程参数或额外配置文件里。配置上限为 2 MiB。
CLI 可使用 `--tts-backend common-tts --tts-backend-kwargs-stdin`，stdin 输入包含
`manifest`（解析后的 YAML 对象）或 `manifest_path`、`base_url`、`parameters`、
可选 `api_key` 的 JSON 对象。

## 验证

宿主测试：`python -m unittest discover -s tests`；前端运行 `npm run check` 和
`npm test`。插件仓库运行 `python -m unittest discover -s test/common-tts`。
跨仓库集成测试设置 `ASMR_TRANSLATOR_ROOT` 为宿主检出目录，使用注入假引擎验证
真实 Adapter 到服务的上传、参数、日志及 WAV 返回，不下载模型。

## 独立参数渲染模块

通用表单位于 `desktop-tauri/src/lib/components/plugin-config/`，与插件识别、TTS 连接和桌面桥分离。工作台的 PluginParameterFields 仅注入文件选择回调并转交共享渲染器。新插件可通过 YAML 声明已有控件；固定音色可使用 readOnly、const 和 x-enum-labels 显示，服务端仍以 const/enum 校验。添加全新交互控件时才需要扩展通用渲染器，不添加模型名称判断。

## 参考音频选择与裁切

音频字段设置 `x-control: audio-file` 和 `x-audio-editor: true` 后，通用渲染器使用与 IndexTTS 共用的音频编辑器，支持波形、缩放/平移、选段试听和起止调整。需先打开项目，确认后将 3–15 秒选段保存到项目缓存，参数保存缓存路径；HTTP 对接仍使用素材上传，不向服务端传递客户端路径。`x-role: emotion_reference` 使用情感缓存，其他角色使用音色缓存。不加编辑器扩展时保持普通文件选择。

共享组件位于 `desktop-tauri/src/lib/components/audio-reference/`；YAML 字段和适配回调见 [渲染模块说明](../../desktop-tauri/src/lib/components/plugin-config/README.md)。编辑器与原生缓存目前固定为 3–15 秒；数组参数仍通过 JSON 输入文件路径。需要参考文本的模型，应让用户填写与最终选段匹配的文字，不从上游原文自动填入。
