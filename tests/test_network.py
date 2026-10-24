"""Offline contracts for shared routing, streaming safety and service wiring."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import aiohttp
import httpx

from modules import network
from modules.llm_enhancer import LLMClient, LLMConfig
from modules.preflight import run_preflight_check


PROXY = "http://user:proxy-password@proxy.test:7890"
URL = "https://service.test/generate?token=query-secret"


class PolicyTests(unittest.TestCase):
    def test_discovery_preserves_launcher_snapshot_and_live_system_proxy(self):
        snapshot = {"HTTPS_PROXY": PROXY, "NO_PROXY": "local.test"}
        with patch.dict(os.environ, {network.PROXY_SNAPSHOT_ENV: json.dumps(snapshot)}), \
                patch.object(network, "getproxies", return_value={"http": "system.test:8080"}):
            self.assertEqual(network.discover_proxies(), {
                "http": "system.test:8080", "https": PROXY, "no": "local.test",
            })

    def test_bad_snapshot_does_not_break_system_discovery(self):
        for value in ("not-json", "[]", "null"):
            with self.subTest(value=value), patch.dict(os.environ, {network.PROXY_SNAPSHOT_ENV: value}), \
                    patch.object(network, "getproxies", return_value={"https": PROXY}):
                self.assertEqual(network.discover_proxies(), {"https": PROXY})

    def test_loopback_and_bypass_never_use_proxy(self):
        policy = network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY, "http": PROXY, "no": "private.test"})
        for url in ("http://localhost:8000", "http://127.0.0.2", "http://[::1]", "http://app.localhost", "https://private.test"):
            self.assertEqual(policy.routes(url), [None])

    def test_no_proxy_without_configuration_and_invalid_configuration(self):
        for proxies in ({}, {"https": "socks4://localhost:9999"}, {"https": "http://[bad"}, {"https": "http://host:invalid"}):
            self.assertEqual(network.NetworkPolicy(proxy_provider=lambda: proxies).routes(URL), [None])

    def test_cache_is_per_origin_expires_and_refreshes_on_proxy_change(self):
        proxies = {"https": PROXY}
        now = [0.0]
        policy = network.NetworkPolicy(proxy_provider=lambda: dict(proxies), clock=lambda: now[0])
        routes = policy.routes(URL)
        self.assertEqual(routes, [None, PROXY])
        policy.succeeded(URL, PROXY, routes)
        self.assertEqual(policy.routes(URL), [PROXY, None])
        self.assertEqual(policy.routes("https://other.test"), [None, PROXY])
        now[0] = 61
        self.assertEqual(policy.routes(URL), [None, PROXY])
        policy.succeeded(URL, PROXY, policy.routes(URL))
        proxies["https"] = "http://new.test:8080"
        self.assertEqual(policy.routes(URL), [None, "http://new.test:8080"])

    def test_frequent_success_does_not_postpone_direct_reprobe(self):
        now = [0.0]
        policy = network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY}, clock=lambda: now[0])
        policy.succeeded(URL, PROXY, policy.routes(URL))
        now[0] = 30
        policy.succeeded(URL, PROXY, policy.routes(URL))
        now[0] = 61
        self.assertEqual(policy.routes(URL), [None, PROXY])


class AsyncRoutingTests(unittest.IsolatedAsyncioTestCase):
    def make_transport(self, outcome):
        calls = []
        created = []
        policy = network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY})

        def factory(**kwargs):
            created.append(kwargs)
            async def handle(request):
                route = kwargs["proxy"]
                calls.append((route, request))
                result = outcome(route, request)
                if isinstance(result, Exception):
                    raise result
                return httpx.Response(result, json={"ok": True})
            return httpx.MockTransport(handle)

        return network.AutoAsyncTransport(policy, factory=factory), calls, created

    async def test_success_and_http_errors_do_not_fallback(self):
        for status in (200, 401, 403, 429, 500, 503):
            with self.subTest(status=status):
                transport, calls, created = self.make_transport(lambda *_: status)
                async with network.async_client(transport=transport) as client:
                    response = await client.post(URL, json={"input": "hello"})
                self.assertEqual(response.status_code, status)
                self.assertEqual([r for r, _ in calls], [None])
                self.assertEqual(created, [{"proxy": None, "trust_env": False}])

    async def test_connection_failure_uses_proxy_and_retains_body_timeout(self):
        transport, calls, _ = self.make_transport(
            lambda route, request: httpx.ConnectTimeout("blocked") if route is None else 200)
        async with network.async_client(timeout=120, transport=transport) as client:
            request = client.build_request("POST", URL, json={"input": "hello"}, headers={"Authorization": "Bearer api-secret"})
            response = await client.send(request)
            await client.post(URL, json={"input": "next"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([r for r, _ in calls], [None, PROXY, PROXY])
        self.assertEqual(calls[1][1].headers["Authorization"], "Bearer api-secret")
        self.assertEqual(await calls[1][1].aread(), b'{"input":"hello"}')
        self.assertEqual(calls[1][1].extensions["timeout"]["connect"], 8)
        self.assertEqual(calls[1][1].extensions["timeout"]["read"], 120)
        self.assertEqual(request.extensions["timeout"]["connect"], 120)

    async def test_both_fail_report_routes_without_secrets(self):
        transport, calls, _ = self.make_transport(lambda *_: httpx.ConnectError("api-secret proxy-password"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            async with network.async_client(transport=transport) as client:
                with self.assertRaises(network.NetworkConnectionError) as caught:
                    await client.post(URL, json={"secret": "prompt-secret"})
        self.assertEqual([r for r, _ in calls], [None, PROXY])
        self.assertIn("直连", str(caught.exception))
        self.assertIn("代理", str(caught.exception))
        for secret in ("api-secret", "proxy-password", "query-secret", "prompt-secret"):
            self.assertNotIn(secret, output.getvalue() + str(caught.exception))

    async def test_read_write_and_protocol_failures_never_replay(self):
        for error_type in (httpx.ReadTimeout, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError):
            with self.subTest(error_type=error_type):
                transport, calls, _ = self.make_transport(lambda *_: error_type("failed after connect"))
                async with network.async_client(transport=transport) as client:
                    with self.assertRaises(error_type):
                        await client.post(URL, content=b"upload")
                self.assertEqual([r for r, _ in calls], [None])

    async def test_proxy_fails_after_cached_success_retries_direct(self):
        state = [False]
        def outcome(route, _request):
            if state[0]:
                return httpx.ConnectError("proxy stopped") if route else 200
            return 200 if route else httpx.ConnectError("direct blocked")
        transport, calls, _ = self.make_transport(outcome)
        async with network.async_client(transport=transport) as client:
            await client.get(URL)
            state[0] = True
            await client.get(URL)
        self.assertEqual([r for r, _ in calls], [None, PROXY, PROXY, None])

    async def test_stream_upload_is_not_consumed_by_failed_connect(self):
        calls = []
        async def body():
            yield b"audio-data"
        def factory(**kwargs):
            class Transport(httpx.AsyncBaseTransport):
                async def handle_async_request(self, request):
                    calls.append(kwargs["proxy"])
                    if kwargs["proxy"] is None:
                        raise httpx.ConnectError("blocked before sending")
                    content = b"".join([part async for part in request.stream])
                    self_content.append(content)
                    return httpx.Response(200, content=b"ok")
            return Transport()
        self_content = []
        transport = network.AutoAsyncTransport(network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY}), factory)
        async with network.async_client(transport=transport) as client:
            await client.post(URL, content=body())
        self.assertEqual(calls, [None, PROXY])
        self.assertEqual(self_content, [b"audio-data"])

    async def test_aiohttp_operation_and_stream_guard(self):
        policy = network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY})
        calls = []
        async def operation(connection):
            route = connection.route
            calls.append(route)
            if route is None:
                raise aiohttp.ConnectionTimeoutError("blocked")
            return "ok"
        self.assertEqual(await network.async_operation(URL, operation, network_policy=policy), "ok")
        self.assertEqual(calls, [None, PROXY])
        calls.clear()
        with self.assertRaises(aiohttp.ConnectionTimeoutError):
            await network.async_operation(URL, operation, network_policy=network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY}), retry_allowed=lambda: False)
        self.assertEqual(calls, [None])

    async def test_aiohttp_socks_uses_connector_and_closes_every_route(self):
        proxy = "socks5h://localhost:1080"
        connections = []
        async def operation(connection):
            connections.append(connection)
            self.assertEqual(connection.request_options["proxy"], "")
            if connection.route is None:
                raise aiohttp.ConnectionTimeoutError("blocked")
            return "ok"
        result = await network.async_operation(URL, operation, network_policy=network.NetworkPolicy(proxy_provider=lambda: {"all": proxy}))
        self.assertEqual(result, "ok")
        self.assertEqual([c.route for c in connections], [None, proxy])
        self.assertTrue(all(c.connector.closed for c in connections))

    async def test_aiohttp_response_error_and_read_timeout_do_not_fallback(self):
        for error in (aiohttp.ServerTimeoutError("read timeout"), aiohttp.ClientResponseError(None, (), status=401)):
            calls = []
            async def operation(connection):
                calls.append(connection.route)
                raise error
            with self.assertRaises(type(error)):
                await network.async_operation(URL, operation, network_policy=network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY}))
            self.assertEqual(calls, [None])

    async def test_llm_and_preflight_use_shared_route_contract(self):
        calls = []
        real_transport = network.AutoAsyncTransport
        def factory(**kwargs):
            async def handle(request):
                calls.append(kwargs["proxy"])
                if kwargs["proxy"] is None:
                    raise httpx.ConnectError("blocked")
                return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
            return httpx.MockTransport(handle)
        with patch.object(network, "discover_proxies", return_value={"https": PROXY}), \
                patch.object(network, "AutoAsyncTransport", side_effect=lambda _policy: real_transport(network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY}), factory)):
            result = await run_preflight_check("llm", {"llmProvider": "openai", "llmBaseUrl": "https://service.test", "llmModel": "test", "llmApiKey": "secret"})
            self.assertTrue(result.success)
            client = LLMClient(LLMConfig(provider="openai", base_url="https://service.test", model="test", api_key="secret"))
            self.assertEqual(await client._post("system", "input"), "ok")
        self.assertEqual(calls, [None, PROXY, None, PROXY])


class SyncRoutingTests(unittest.TestCase):
    def test_sync_connection_uses_same_policy_and_closes_transports(self):
        calls = []
        closed = []
        def factory(**kwargs):
            class Transport(httpx.BaseTransport):
                def handle_request(self, request):
                    calls.append(kwargs["proxy"])
                    if kwargs["proxy"] is None:
                        raise httpx.ConnectError("blocked")
                    return httpx.Response(200, content=b"ok")
                def close(self):
                    closed.append(kwargs["proxy"])
            return Transport()
        transport = network.AutoTransport(network.NetworkPolicy(proxy_provider=lambda: {"https": PROXY}), factory)
        with network.sync_client(transport=transport) as client:
            self.assertEqual(client.get(URL).text, "ok")
        self.assertEqual(calls, [None, PROXY])
        self.assertEqual(closed, [None, PROXY])
