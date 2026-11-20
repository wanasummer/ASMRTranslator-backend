"""Shared async text generation: network routing, retries, cache and diagnostics."""
from __future__ import annotations
import asyncio
import json
import random
from dataclasses import asdict, replace
from typing import Optional
import httpx
from ..error import LLMError
from ..network import NetworkPolicy, async_client
from ..task_cache import SqliteResultCache, cache_key, public_settings
from .config import LLMConfig, LLMConnectionConfig
from .providers import create_provider
from .artifacts import RequestArtifactWriter
from .contracts import Provider

class LLMRuntime:
    def __init__(self, cfg: LLMConnectionConfig, cache_dir: str | None = None, *, provider: Provider | None = None,
                 client_factory=None, cache=None, sleep=None):
        self.cfg = cfg.connection() if isinstance(cfg, LLMConfig) else replace(cfg)
        self.provider = provider or create_provider(self.cfg)
        self.client_factory = client_factory
        self.sleep = sleep
        self._network_policy = NetworkPolicy()
        self.cache = cache if cache is not None else (SqliteResultCache(cache_dir, "llm_results") if cache_dir else None)
        self.artifacts = RequestArtifactWriter(cache_dir, secret=cfg.api_key)

    @staticmethod
    def _gemini_retry_delay(response, attempt: int) -> float:
        header = getattr(response, "headers", {}).get("retry-after", "")
        try:
            advised = max(0.0, float(header))
        except (TypeError, ValueError):
            advised = 0.0
        return min(60.0, max(2.0 ** (attempt + 1), advised) + random.uniform(0, 0.5))


    def request_key(self, system: str, user: str, max_tokens: Optional[int]) -> str:
        return cache_key({"settings": public_settings(asdict(self.cfg)),
                          "system": system, "user": user, "max_tokens": max_tokens})


    async def complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        key = self.request_key(system, user, max_tokens)
        self.artifacts.request(key, self.cfg, system, user, max_tokens)
        if self.cache:
            cached = self.cache.load(key)
            if isinstance(cached, str) and cached.strip():
                print("       [cached-llm] 复用已完成请求")
                return cached
        try:
            result = await self._post_uncached(system, user, max_tokens)
            if self.cache:
                self.cache.save(key, result)
            return result
        except Exception as error:
            if self.cache:
                self.cache.fail(key, error)
            raise


    async def _post_uncached(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        url = self.provider.endpoint()
        headers = self.provider.headers()
        payload = self.provider.payload(system, user, max_tokens)
        last_error: Optional[Exception] = None
        direct_gemini = self.provider.is_direct_gemini()
        # Keep other OpenAI-compatible providers on the existing retry policy.
        gemini_limit = max(self.cfg.max_retries, 5) if direct_gemini else self.cfg.max_retries
        for attempt in range(gemini_limit + 1):
            try:
                async with async_client(client_factory=self.client_factory or httpx.AsyncClient, network_policy=self._network_policy, timeout=self.cfg.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                gemini_transient = direct_gemini and response.status_code in (429, 503)
                retry_limit = gemini_limit if gemini_transient else self.cfg.max_retries
                if (response.status_code >= 500 or gemini_transient) and attempt < retry_limit:
                    last_error = self._http_error(response, url, attempt + 1)
                    delay = (self._gemini_retry_delay(response, attempt) if gemini_transient
                             else 2 ** attempt)
                    if gemini_transient:
                        print(f"       [llm-retry] Gemini HTTP {response.status_code}, "
                              f"{delay:.1f}s 后重试（{attempt + 1}/{gemini_limit}）")
                    await (self.sleep or asyncio.sleep)(delay)
                    continue
                if response.status_code != 200:
                    raise self._http_error(response, url, attempt + 1)
                response_data = response.json()
                self.artifacts.response(self.request_key(system, user, max_tokens), response_data)
                try:
                    return self.provider.parse(response_data)
                except LLMError as exc:
                    if not str(exc).startswith("大模型未返回文本"):
                        raise
                    choices = response_data.get("choices") if isinstance(response_data, dict) else None
                    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
                    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
                    usage = response_data.get("usage") if isinstance(response_data, dict) else None
                    safe_usage = {
                        key: value for key, value in usage.items()
                        if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        and isinstance(value, int)
                    } if isinstance(usage, dict) else {}
                    print(f"       [llm-empty-response] model={self.cfg.model} "
                          f"attempt={attempt + 1} finish_reason={choice.get('finish_reason')!r} "
                          f"message_keys={sorted(message)} usage={safe_usage}")
                    if direct_gemini and attempt < self.cfg.max_retries:
                        last_error = exc
                        await (self.sleep or asyncio.sleep)(2 ** attempt)
                        continue
                    raise
            except LLMError:
                # 4xx / unparseable responses are deterministic — retrying won't help.
                raise
            except Exception as e:  # noqa: BLE001 — transport errors are retriable
                last_error = e
                if attempt < self.cfg.max_retries:
                    await (self.sleep or asyncio.sleep)(2 ** attempt)
                    continue
                break
        raise LLMError(
            f"LLM request failed after {self.cfg.max_retries + 1} tries: {last_error}"
        )


    def _http_error(self, response, url: str, attempt: int) -> LLMError:
        """Log bounded, redacted service diagnostics before constructing an error.

        Request headers and prompts are deliberately excluded. The pipeline tee
        persists this record, and the error reporter includes it without the
        500-character user-facing detail limit.
        """
        from urllib.parse import urlsplit, urlunsplit
        from ..error import descriptor, safe_detail

        def redact(value, limit=32768):
            text = str(value or "")
            if self.cfg.api_key:
                text = text.replace(self.cfg.api_key, "[REDACTED]")
            return safe_detail(text, limit=limit)

        parsed_url = urlsplit(url)
        host = parsed_url.netloc.rsplit("@", 1)[-1]
        endpoint = urlunsplit((parsed_url.scheme, host, parsed_url.path, "", ""))
        body = response.text
        try:
            data = response.json()
        except (ValueError, TypeError):
            data = {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        service_error = data.get("error", {}) if isinstance(data, dict) else {}
        service_error = service_error if isinstance(service_error, dict) else {}
        response_headers = getattr(response, "headers", {})
        request_ids = {key: redact(response_headers.get(key), 500)
                       for key in ("x-request-id", "request-id", "x-amzn-requestid", "cf-ray")
                       if response_headers.get(key)}
        record = {
            "provider": self.cfg.provider, "model": redact(self.cfg.model, 500),
            "endpoint": redact(endpoint, 2000), "status": response.status_code,
            "attempt": attempt, "request_ids": request_ids,
            "error_type": redact(service_error.get("type"), 500),
            "error_code": redact(service_error.get("code"), 500),
            "error_param": redact(service_error.get("param"), 500),
            "error_message": redact(service_error.get("message"), 4000),
            "response_body": redact(body), "response_body_chars": len(body),
            "response_body_truncated": len(body) > 32768,
        }
        print("       [llm-http-error] " + json.dumps(record, ensure_ascii=False))
        detail = (f"{self.cfg.provider} LLM returned {response.status_code}: "
                  f"type={record['error_type']} code={record['error_code']} "
                  f"message={record['error_message']} body={record['response_body']}")
        if service_error.get("type") == "content_policy_violation" or service_error.get("code") == "content_policy_violation":
            error = LLMError(descriptor("LLM_CONTENT_REJECTED"), detail=detail)
        elif self.provider.is_direct_gemini() and response.status_code == 429:
            error = LLMError(descriptor("LLM_GEMINI_RATE_LIMITED"), detail=detail)
        elif self.provider.is_direct_gemini() and response.status_code == 503:
            error = LLMError(descriptor("LLM_SERVICE_BUSY"), detail=detail)
        else:
            error = LLMError(detail)
        error.context.update({"status": response.status_code, "backend": self.cfg.provider})
        if error.descriptor.code in ("LLM_CONTENT_REJECTED", "LLM_ACCESS_DENIED") and record["error_message"]:
            from dataclasses import replace
            error.descriptor = replace(
                error.descriptor,
                user_message=f"{error.descriptor.user_message} 服务返回：{redact(record['error_message'], 300)}",
            )
        return error


    async def check_connection(self) -> str:
        """Perform a real, minimal request through the production HTTP path."""
        return await self.complete(
            "你是连接检测助手。",
            "只回复：好",
            max_tokens=256,
        )
