import asyncio
import contextlib
import io
import unittest

import httpx

from modules.plugin_logs import relay_plugin_logs


class PluginLogRelayTests(unittest.TestCase):
    def test_relays_incremental_plugin_log_to_console(self):
        async def scenario():
            async def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(request.url.params["request_id"], "request-1")
                return httpx.Response(200, json={
                    "contract_version": 1,
                    "cursor": 2,
                    "logs": [{
                        "sequence": 2,
                        "level": "info",
                        "message": "正在推理",
                        "current": 3,
                        "total": 10,
                    }],
                })

            stop = asyncio.Event()
            stop.set()
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await relay_plugin_logs(
                    client, "http://plugin", "request-1", {}, stop,
                    source="test-plugin",
                )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            asyncio.run(scenario())
        self.assertIn("[plugin:test-plugin] 正在推理 (3/10)", output.getvalue())

    def test_missing_log_endpoint_is_optional(self):
        async def scenario():
            async def handler(_: httpx.Request) -> httpx.Response:
                return httpx.Response(404)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await relay_plugin_logs(
                    client, "http://old-plugin", "request-1", {}, asyncio.Event(),
                    source="old-plugin",
                )

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
