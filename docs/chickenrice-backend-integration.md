# 海南鸡字幕服务后端对接文档

版本：ChickenRice Subtitle Service `0.3.0`
接口版本：`apiVersion=2`，结果契约：`contract_version=1`
示例服务地址（请替换为自己的部署）：`http://127.0.0.1:7870`

> 当前地址是明文 HTTP。生产环境建议通过 HTTPS 反向代理暴露，并启用 API Key。

## 1. 服务定位

服务安装时会一次性安装以下能力，但每个任务可独立选择实际使用的流程：

- 日语 ASR 核心；
- 海南鸡 ASMR VAD；
- 日语音频直出中文字幕。

任务由两个正交参数决定：

| 参数 | 可选值 | 说明 |
|---|---|---|
| `vad_provider` | `none` | 不使用 VAD，以 ASR 片段时间戳为输出时间轴 |
| `vad_provider` | `external` | 使用调用方提交的 VAD 切片；主项目推荐 |
| `vad_provider` | `chickenrice` | 使用海南鸡 ASMR VAD |
| `output_language` | `ja` | 输出日语原文字幕，进入现有清洗、总结、翻译流程 |
| `output_language` | `zh` | 直接输出中文字幕，跳过日语清洗、总结和翻译 |

默认值是 `vad_provider=none`、`output_language=ja`。

## 2. 鉴权

服务器设置 `CHICKENRICE_API_KEY` 后，所有接口都要求：

```http
Authorization: Bearer <api-key>
```

未配置 API Key 时接口不鉴权。鉴权失败返回 `401`：

```json
{"detail":"API key 无效"}
```

后端日志不得记录 API Key。

## 3. 健康检查

```http
GET /health
```

示例：

```bash
curl http://127.0.0.1:7870/health
```

响应：

```json
{
  "status": "ready",
  "version": "0.3.0",
  "device": "cuda",
  "compute_type": "int8_float16",
  "queue_size": 0,
  "capabilities": {
    "japanese_asr": true,
    "chinese_translation": true,
    "chickenrice_vad": true
  },
  "runtime": {
    "available": true,
    "device": "cuda",
    "provider": "ctranslate2-cuda",
    "problems": []
  },
  "problems": []
}
```

注意：运行时不完整时接口仍可能返回 HTTP `200`，调用方必须同时检查 `status == "ready"` 和三项 capability。

自 `0.3.0` 起，Windows 安装器会安装 cuBLAS 12 和 cuDNN 9，并在安装完成及每次启动前执行推理运行时自检。DLL、NVIDIA 驱动或 CTranslate2 不可用时，`status` 必须为 `not_ready`，具体原因位于 `runtime.problems` 和顶层 `problems`。

## 4. 创建字幕任务

```http
POST /v1/jobs
Content-Type: multipart/form-data
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | File | 是 | 待处理音频或可被 ffmpeg 解码的媒体文件 |
| `vad_provider` | String | 否 | `none`、`external`、`chickenrice` |
| `output_language` | String | 否 | `ja`、`zh` |
| `segments` | JSON String | 条件必填 | 仅 `vad_provider=external` 时必须提交 |

### 4.1 使用当前项目 VAD，输出日语

这是当前主工作流的推荐组合：

```bash
curl -X POST http://127.0.0.1:7870/v1/jobs \
  -H "Authorization: Bearer YOUR_KEY" \
  -F "audio=@audio.wav" \
  -F "vad_provider=external" \
  -F "output_language=ja" \
  -F 'segments=[{"index":0,"start_ms":1200,"end_ms":4600},{"index":1,"start_ms":5100,"end_ms":8300}]'
```

`segments` 约束：

- `index`、`start_ms`、`end_ms` 必须是整数；
- `index >= 0`、`start_ms >= 0`、`end_ms > start_ms`；
- 必须按时间升序排列；
- 相邻切片不能重叠，允许首尾相接；
- 外部 VAD 模式下至少有一个切片；
- 非 `external` 模式不得提交 `segments`。

外部 VAD 是 authored timeline。结果会原样保留每个输入切片的 `index/start_ms/end_ms` 和顺序，Whisper 片段时间戳不得覆盖它。即使某个切片没有识别出文本，JSON 中也保留该块并返回 `text=""`。

### 4.2 使用海南鸡 VAD，输出日语

```bash
curl -X POST http://127.0.0.1:7870/v1/jobs \
  -F "audio=@audio.wav" \
  -F "vad_provider=chickenrice" \
  -F "output_language=ja"
```

### 4.3 使用海南鸡 VAD，直出中文

```bash
curl -X POST http://127.0.0.1:7870/v1/jobs \
  -F "audio=@audio.wav" \
  -F "vad_provider=chickenrice" \
  -F "output_language=zh"
```

### 4.4 只使用核心 ASR

```bash
curl -X POST http://127.0.0.1:7870/v1/jobs \
  -F "audio=@audio.wav" \
  -F "vad_provider=none" \
  -F "output_language=ja"
```

该模式使用 ASR 自身片段时间戳，不符合主项目当前“VAD 切片是 authored timeline”的默认约束，建议仅用于独立转录、诊断或明确改变产品需求后的流程。

### 4.5 创建响应

成功返回 HTTP `202`：

```json
{
  "id": "e68c1593a65d4ee19a7853f3d16a42d3",
  "status": "queued",
  "filename": "audio.wav",
  "created_at": 1787800000.0,
  "updated_at": 1787800000.0,
  "progress": 0.0,
  "message": "等待处理",
  "error": null,
  "block_count": 0,
  "vad_provider": "external",
  "output_language": "ja",
  "segments": [
    {"index":0,"start_ms":1200,"end_ms":4600}
  ]
}
```

## 5. 查询任务

```http
GET /v1/jobs/{job_id}
```

任务状态：

| 状态 | 说明 |
|---|---|
| `queued` | 已进入单工作线程队列 |
| `running` | GPU/CPU 正在处理 |
| `succeeded` | 结果已持久化，可获取 JSON/VTT |
| `failed` | 处理失败，查看 `error` 和 `message` |

推荐每 `1–2` 秒轮询一次，不要高频请求。服务采用单工作线程，同一时刻只执行一个推理任务，`queue_size` 代表等待中的任务数。

服务重启时，原先处于 `queued/running` 的任务会被标为 `failed`，错误为“服务重启中断了任务，请重新提交”。

## 6. 获取结构化结果

```http
GET /v1/jobs/{job_id}/result
```

仅 `succeeded` 状态可获取；其他状态返回 `409`，`detail` 中包含当前任务对象。

日语 ASR + 外部 VAD 示例：

```json
{
  "job_id": "e68c1593a65d4ee19a7853f3d16a42d3",
  "contract_version": 1,
  "stage": "asr",
  "language": "ja",
  "source": "chickenrice-ja-asr",
  "vad_provider": "external",
  "timeline_source": "external_vad",
  "vtt_filename": "subtitles.vtt",
  "blocks": [
    {
      "index": 0,
      "start_ms": 1200,
      "end_ms": 4600,
      "text": "今日もお疲れさまでした。",
      "skip_tts": false
    },
    {
      "index": 1,
      "start_ms": 5100,
      "end_ms": 8300,
      "text": "",
      "skip_tts": false
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `stage` | 固定为 `asr` |
| `language` | `ja` 或 `zh`，下游必须据此选择流程 |
| `source` | 实际识别实现标识 |
| `vad_provider` | 本任务选择的 VAD 实现 |
| `timeline_source` | `asr`、`external_vad` 或 `chickenrice_vad` |
| `blocks` | 按时间轴顺序排列的字幕块 |
| `skip_tts` | ASR 阶段默认 `false`；由后续清洗/中文短句规则决定最终值 |

结构化 JSON 是主项目的规范输入；VTT 主要用于人工检查和字幕导出。外部 VAD 的空文本块会保留在 JSON 中，但不会写成无效的空 VTT cue。

## 7. 获取 VTT

```http
GET /v1/jobs/{job_id}/subtitles.vtt
```

成功响应：

```http
Content-Type: text/vtt; charset=utf-8
Content-Disposition: attachment; filename="subtitles.vtt"
```

字幕不可用时返回 `404`。

## 8. 删除任务

```http
DELETE /v1/jobs/{job_id}
```

- 删除成功：`204 No Content`；
- 任务不存在：`404`；
- `running` 任务不能删除，返回 `409`。

任务默认保留 24 小时，过期任务会在后续提交任务时触发清理。

## 9. 与当前流水线的对接规则

### 9.1 推荐：项目 VAD + 日语 ASR

```text
当前项目 VAD
  → POST /v1/jobs (external + ja)
  → 日语字幕块（保持 VAD 时间轴）
  → 日语清洗及审计
  → LLM 总结
  → 翻译
  → 可选中文润色
  → 中文 <= 3 CJK 字符 skip_tts 规则
  → TTS / 音频处理 / 拼接
```

后端适配器应直接把返回块映射为当前 `VTTBlock`：

```python
VTTBlock(
    index=block["index"],
    start_ms=block["start_ms"],
    end_ms=block["end_ms"],
    text=block["text"],
    skip_tts=block.get("skip_tts", False),
)
```

不得根据日语文本长度过滤字幕。空日语块进入 cleaner 后仍保留时间轴，并由后续阶段标记 `skip_tts=True`。

### 9.2 中文直出

当 `language == "zh"` 时：

```text
海南鸡返回中文块
  → 可选中文润色
  → 现有中文 <= 3 CJK 字符 skip_tts 规则
  → TTS / 音频处理 / 拼接
```

必须显式绕过日语清洗、LLM 总结和翻译阶段。不得仅根据配置假设语言，应以响应中的 `language` 为准；如果预期中文却收到日语，应中止流程，不能把日语静默送入中文 TTS。

## 10. Python 调用示例

```python
import json
import time

import requests


BASE_URL = "http://127.0.0.1:7870"
API_KEY = ""
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

segments = [
    {"index": 0, "start_ms": 1200, "end_ms": 4600},
    {"index": 1, "start_ms": 5100, "end_ms": 8300},
]

with open("audio.wav", "rb") as audio:
    response = requests.post(
        f"{BASE_URL}/v1/jobs",
        headers=HEADERS,
        files={"audio": ("audio.wav", audio, "audio/wav")},
        data={
            "vad_provider": "external",
            "output_language": "ja",
            "segments": json.dumps(segments, ensure_ascii=False),
        },
        timeout=120,
    )
response.raise_for_status()
job_id = response.json()["id"]

while True:
    response = requests.get(f"{BASE_URL}/v1/jobs/{job_id}", headers=HEADERS, timeout=15)
    response.raise_for_status()
    job = response.json()
    if job["status"] == "succeeded":
        break
    if job["status"] == "failed":
        raise RuntimeError(job.get("error") or job.get("message"))
    time.sleep(1.5)

response = requests.get(f"{BASE_URL}/v1/jobs/{job_id}/result", headers=HEADERS, timeout=30)
response.raise_for_status()
result = response.json()
if result["language"] != "ja":
    raise RuntimeError(f"预期日语 ASR，实际收到 {result['language']}")
```

## 11. HTTP 错误处理

| 状态码 | 场景 | 调用方处理 |
|---|---|---|
| `401` | API Key 缺失或错误 | 停止重试并提示配置鉴权 |
| `404` | 任务或字幕不存在 | 校验 job ID；过期任务需重新提交 |
| `409` | 结果未完成或删除运行中任务 | 继续轮询，或等待任务结束 |
| `413` | 上传超限或空文件 | 不重试，提示用户更换输入 |
| `422` | VAD/语言/segments 契约错误 | 修正参数后重新提交 |
| `503` | 模型或运行时不完整 | 停止提交，提示服务器管理员修复安装 |
| `5xx`/网络异常 | 服务端或链路暂时异常 | 指数退避重试；不要重复并发提交同一文件 |

默认上传上限为 4 GiB，单任务推理超时为 6 小时。上传请求超时、任务轮询超时和总业务超时应分别配置。

## 12. 建议的主项目配置

```yaml
subtitle_source:
  provider: chickenrice
  endpoint: http://127.0.0.1:7870
  api_key: ${CHICKENRICE_API_KEY}
  request_timeout_seconds: 120
  poll_interval_seconds: 1.5
  vad_provider: external
  output_language: ja
```

配置切换只选择实现，不应在 `run_pipeline` 中增加长条件链。建议实现独立的 ChickenRice ASR adapter，通过现有 ASR factory/registry 注册；adapter 只负责请求服务并返回 ASR 阶段契约，不得导入 cleaner、翻译或 TTS 模块。
