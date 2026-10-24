# Common TTS HTTP contract, version 1

## Layout and ownership

Each plugin owns `source/` (official source, licenses, model resources), `inject/`
(integration code), `plugin.yml` (synthesis descriptor), `app.py` (thin controller
and process entry) and `install.py` (thin installer entry). Installer metadata,
requirements and documentation may also live at the root. `source/` is not a
place for HTTP or installer glue. `inject/` can wrap or patch upstream code.

The installer launches `app.py`. It must retain host/port arguments and the
shared console gateway supervisor. Keep runtime/model imports lazy. Install
models explicitly and verify offline startup before publishing a real plugin.

## Descriptor

`schema_version: 1`, `protocol: common-tts/1`, `id`, `version` and `name` are
required. `parameters` is a JSON Schema 2020-12 object with
`additionalProperties: false`. External `$ref` is prohibited; local references
and `allOf`/`if`/`then` validation are supported. Parameters must be finite JSON
values. Defaults in object properties are applied recursively. Object/array
overrides replace the entire value; null is a value, never an implicit deletion.

UI extensions:

| Extension | Meaning |
| --- | --- |
| `title`, `description`, `enum`, numeric limits | Label, help, selection and bounds |
| `x-control: audio-file` | Local file picker; bridge uploads the file |
| `x-role` | Reference purpose, e.g. voice_reference or emotion_reference |
| `x-control: textarea` | Multiline text |
| `x-placeholder`, `x-rows` | Textarea placeholder and row count |
| `x-control: file` | Local file path parameter; not uploaded as an audio asset |
| `x-control: range` | Numeric slider; minimum/maximum set its bounds |
| `x-step`, `x-decimals`, `x-suffix` | Slider step, displayed decimal places and unit suffix; UI-only, not validation |
| `x-display: inline` / `grid` | Object children in the parent layout / a two-column grid without a folding group |
| `x-open: true` | Initially expand an ordinary object group |
| `x-appearance: studio` | Root setting, inherited by fields, matching the existing studio panel appearance |
| `x-audio-editor: true` | Project-backed reference trimming and preview for audio fields, currently 3–15 seconds |
| `enum`, `x-enum-labels` | Static dropdown values and display labels; dynamic `x-source: voices` is not supported |
| `x-visible-when: {field: value}` | Sibling-field visibility; does not disable validation |
| `x-visible-if` | Root-parameter presentation condition |
| `x-option-visible-if` | Per-enum-value root-parameter presentation conditions |
| `x-normalize` | Root-level ordered rules with `when` and dotted-path `set`, applied once after user edits |
| `x-notes`, `x-footer` | Conditional text notes after a field / at a group's end, after its errors |
| `x-errors: start` / `end` | Group-level field error placement; inline objects inherit unless overridden |
| `x-commit-on-blur: true` | Commit numeric edits on blur or Enter, retaining the previous value for invalid input |
| `readOnly: true` | Read-only display; enforce fixed values through server-side const/enum validation |
| `x-secret: true` | Password input and diagnostic redaction |
| `x-segment-override: true` | Top-level parameter may be overridden for one segment |

Arrays use a JSON editor in the initial desktop renderer. File annotations work
inside nested objects and arrays. Pair a reference's audio with its transcript,
language or weight in the same object; the structure is preserved.

Presentation conditions support only `properties`, `required`, `const`, `enum`,
`minimum`, `exclusiveMinimum`, `allOf`, `anyOf` and `not`. They always see the
root parameters, even inside nested groups; `x-visible-when` remains local to
siblings. Hidden values are retained. Normalization is a single ordered pass
after an edit, not a migration or server validation. Notes contain `text` and
optional visibility extensions. These UI extensions require a supporting desktop
renderer; updating a plugin alone does not update an older host.

The desktop snapshots descriptor and synthesis parameters with the project/task.
Connection URL and API key are separate application settings. Authentication
must not be embedded in a shareable YAML. Use plugin/model version changes to
invalidate synthesis caches. All effective parameters and input content hashes
participate in caching; request IDs, timeout and log cursor do not identify audio.

`capabilities.execution` is `segment`. `voices`, `rate`, `pitch` are booleans.
`text_policy` is `preserve` or `strip_tags`. `required_assets` lists required roles.
Connection probes validate the descriptor and health without requiring workspace
references. Task preflight validates parameter Schema, configured reference files
and their file-count/upload limits, but defers required roles because upstream
segment audio does not exist yet. Once a segment's assets are available, cache
input preparation and synthesis request preparation check all required roles and
combined upload limits before sending a synthesis request. Missing required roles
are errors at that boundary, not skipped validation. The required-role list is
unconditional; mode-specific requirements belong in plugin validation.
An unsupported nondefault rate/pitch is rejected rather than silently ignored.
`execution.max_concurrent`, `timeout_seconds`, `max_upload_bytes` and `max_output_bytes` are positive. Set `execution.cache: false` to disable host synthesis reuse for stochastic engines.
The desktop limit and plugin limit both apply; separate plugin processes need
their own service-wide/GPU/credential limiter where applicable.

## Discovery and health

- `GET /manifest`: parsed YAML as JSON, without credentials or model downloads.
- `GET /health`: `{contract_version:1, plugin_id, plugin_version, ready, status}`.
  Ready is HTTP 200. HTTP 503 is reserved for loading; a failed load is HTTP 500,
  logged with diagnostic type. ID and version must match the imported YAML.
- `GET /voices`: `{voices:[{id,name,language?,gender?,description?}]}`.

Optional authentication uses `Authorization: Bearer <token>` on every endpoint.

## Synthesis

`POST /job` is multipart/form-data. `payload` is a UTF-8 JSON form field. Each
asset is a file part whose name is the asset ID. Local client paths are never
sent to the server. Server filenames must be generated from validated IDs,
never trusted upload names.

```json
{
  "contract_version": 1,
  "plugin_id": "example-common-tts",
  "plugin_version": "0.1.0",
  "request_id": "job-uuid",
  "task_id": "run-uuid",
  "attempt": 1,
  "segment": {
    "id": "stable-segment-id", "index": 0,
    "text": "中文译文", "authored_text": "中文译文", "source_text": null,
    "speaker": null, "language": "zh-CN",
    "timing": {"original_start_ms": 100, "original_end_ms": 1200,
               "max_allowed_tts_ms": 1500, "tts_delay_ms": 100},
    "context": {}
  },
  "parameters": {"voice_source": "segment"},
  "controls": {"voice": "", "rate": "+0%", "pitch": "+0Hz"},
  "assets": [{"id": "source_speech", "role": "source_speech", "sha256": "content digest",
              "media_type": "audio/wav", "sample_rate_hz": 24000, "channels": 2}],
  "output": {"container": "wav"}
}
```

Header `X-Request-ID` must match payload request_id (1–128 ASCII letters, digits,
`._:-`, beginning with a letter or digit). source_speech is the original audio
inside the subtitle span, including any background sound; source_segment includes
surrounding pauses. References from audio-file parameters become
`{asset_id: "parameter_..."}` in the wire parameters; the server resolves them to
temporary local paths before parameter validation and model invocation. At most
32 assets are accepted. Verify SHA-256 and upload limits before inference.

Success returns a complete, nonempty, decodable PCM WAV with `Content-Type:
audio/wav`, `X-TTS-Contract-Version: 1`, `X-Request-ID` and `X-TTS-Segment-ID`.
A structured `application/json` result may instead return `contract_version: 1`, `request_id`, `segment_id`, `audio_base64` (complete WAV), optional `warnings` (strings), and optional `boundaries: {type: word, unit: ms, words: [{text, offset_ms, duration_ms}]}`. The same protocol/request headers are required. The host publishes word boundaries beside audio as `.boundaries.json`. Declare `capabilities.boundaries: true` when supported. The host validates audio before publishing it atomically. Sample rate, channels
and sample width are described by WAV, not inferred from the extension. Native
streaming services can buffer internally; incomplete audio is never a success.

Failures return a non-2xx status and
`{contract_version:1, request_id, error:{code, retryable:false}}`. Never echo
credentials or the complete request body in errors. Unknown extension fields may
be ignored, but existing field semantics cannot change within v1.

Use `OUT_OF_MEMORY` for memory allocation failures (including `bad_alloc` and
`bad allocation`), and `MODEL_LOAD_FAILED` for other model initialization failures.
The host maps these to actionable user messages. Generic `SYNTHESIS_FAILED` and
other HTTP 5xx errors are service processing failures; they do not imply the
service is unreachable. Provider messages may be inspected for memory failure
classification but are not displayed or copied into host error details.

## Logging and lifecycle

`GET /log?request_id=...&after=0` returns
`{contract_version:1,request_id,cursor,logs:[{sequence,timestamp,level,message,current?,total?}]}`.
Sequences increase per request. Retention is bounded. Console logging includes
request IDs so the existing process gateway can correlate output. Process session
logs may not have a segment ID and must not be interpreted as synthesis progress.
The desktop's optional polling failure never makes synthesis fail.

Requests are not automatically idempotent. A timeout after submission can mean
the model still runs; the host does not automatically replay. Cancelling local
waiting does not promise remote cancellation. The service keeps its inference
slot and assets alive until native work actually completes. Close model/client
resources during shutdown. Asset paths are scoped to one model call.

## Extension boundaries

The v1 delivery supports complete per-segment WAV synthesis. Cross-segment
sessions, host audio streaming, remote job cancellation, resume/idempotency,
voice-management operations, and phoneme result sidecars require negotiated
optional interfaces or a later protocol version; they must not be declared as
supported by this template. Vendor-specific parameters and references can be
added to the YAML without changing the synthesis function. Lifecycle and
management operations must not be hidden inside synthetic text parameters.

Tests inject an Engine, never download models, and check parameter validation,
asset digests, correlation, logging, auth, cleanup, and complete WAV responses.
