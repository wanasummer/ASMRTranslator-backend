"""Best-effort client for the local plugin ``GET /log`` protocol."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import inspect
from typing import Mapping

import httpx


async def relay_plugin_logs(
    client: httpx.AsyncClient,
    base_url: str,
    request_id: str,
    headers: Mapping[str, str],
    stop: asyncio.Event,
    *,
    source: str,
    interval: float = 0.5,
    emit=None,
) -> None:
    """Poll one plugin job and copy new events to the pipeline console.

    ``/log`` is optional. A missing or incompatible endpoint disables polling
    silently so existing third-party plugins keep working unchanged.
    """
    cursor = 0
    while True:
        try:
            response = await client.get(
                f"{base_url.rstrip('/')}/log",
                params={"request_id": request_id, "after": cursor},
                headers=dict(headers),
                timeout=1.5,
            )
            if response.status_code in {404, 405, 501}:
                return
            checked = response.raise_for_status()
            if inspect.isawaitable(checked):
                await checked
            payload = response.json()
            if inspect.isawaitable(payload):
                payload = await payload
            if not isinstance(payload, dict):
                return
            if payload.get("contract_version") != 1:
                return
            events = payload.get("logs", [])
            if not isinstance(events, list):
                return
            for event in events:
                if not isinstance(event, dict):
                    continue
                sequence = event.get("sequence")
                if isinstance(sequence, int):
                    cursor = max(cursor, sequence)
                message = str(event.get("message") or "").strip()
                if not message:
                    continue
                current, total = event.get("current"), event.get("total")
                progress = (
                    f" ({current}/{total})"
                    if isinstance(current, int) and isinstance(total, int) and total > 0
                    else ""
                )
                line = f"[plugin:{source}] {message}{progress}"
                if emit is not None:
                    emit(line)
                else:
                    print(line, flush=True)
        except (httpx.HTTPError, ValueError, TypeError):
            # This is a console enhancement, never a reason to fail generation.
            return
        if stop.is_set():
            return
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)
