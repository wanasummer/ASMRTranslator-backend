"""Explicit direct-first networking shared by service clients.

Only connection establishment errors permit fallback: response errors and read/
write failures must not replay a possibly accepted generation or upload request.
OS-level tunnels still apply to direct connections. Proxy discovery is refreshed
on every request; successful routes are cached only in the owning policy.
"""
from __future__ import annotations

import ipaddress
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass_environment

import httpx
import aiohttp
from aiohttp_socks import ProxyConnector, ProxyConnectionError, ProxyTimeoutError, ProxyError


PROXY_SNAPSHOT_ENV = "ASMR_NETWORK_PROXY_ENV"
CONNECT_TIMEOUT = 8.0
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError)
_T = TypeVar("_T")


def discover_proxies() -> dict[str, str]:
    """Read live OS/env proxies plus proxies isolated by the desktop launcher."""
    proxies = dict(getproxies())
    try:
        snapshot = json.loads(os.environ.get(PROXY_SNAPSHOT_ENV, "{}"))
    except (ValueError, TypeError):
        snapshot = {}
    if isinstance(snapshot, dict):
        for key, value in snapshot.items():
            scheme = str(key).lower().removesuffix("_proxy")
            if scheme in {"http", "https", "all", "no"} and isinstance(value, str) and value:
                proxies[scheme] = value
    return proxies


def is_loopback(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class NetworkConnectionError(httpx.ConnectError):
    """All configured connection routes failed; messages exclude credentials."""

    def __init__(self, failures: list[tuple[str | None, Exception]], request=None):
        self.failures = tuple(failures)
        details = "；".join(
            f"{'直连' if route is None else '代理'}连接失败（{type(error).__name__}）"
            for route, error in failures
        )
        super().__init__(details, request=request)


@dataclass
class NetworkPolicy:
    proxy_provider: Callable[[], Mapping[str, str]] = discover_proxies
    clock: Callable[[], float] = time.monotonic
    cache_seconds: float = 60.0
    _successes: dict[str, tuple[tuple[str | None, ...], str | None, float]] = field(default_factory=dict, repr=False)

    def routes(self, url: str) -> list[str | None]:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if is_loopback(host):
            return [None]
        proxies = self.proxy_provider()
        if proxy_bypass_environment(parsed.netloc, proxies):
            return [None]
        routes: list[str | None] = [None]
        # HTTP proxies tunnel HTTPS/WSS; do not assume an HTTP-only setting
        # applies to HTTPS. Unknown/SOCKS4 schemes cannot be used by HTTPX.
        scheme = "https" if parsed.scheme == "wss" else parsed.scheme
        for key in (scheme, "all"):
            raw = proxies.get(key, "")
            if not raw:
                continue
            normalized = raw if "://" in raw else "http://" + raw
            try:
                proxy = urlsplit(normalized)
                valid = bool(proxy.hostname) and (proxy.port is None or 1 <= proxy.port <= 65535)
            except ValueError:
                continue
            if valid and proxy.scheme in {"http", "https", "socks5", "socks5h"} and normalized not in routes:
                routes.append(normalized)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cached = self._successes.get(origin)
        fingerprint = tuple([None] + sorted(r for r in routes if r is not None))
        if cached and cached[0] == fingerprint and self.clock() - cached[2] < self.cache_seconds:
            routes.remove(cached[1])
            routes.insert(0, cached[1])
        return routes

    def succeeded(self, url: str, route: str | None, candidates: list[str | None]) -> None:
        parsed = urlsplit(url)
        # Canonical fingerprint keeps cached ordering from invalidating itself.
        fingerprint = tuple([None] + sorted(r for r in candidates if r is not None))
        origin = f"{parsed.scheme}://{parsed.netloc}"
        previous = self._successes.get(origin)
        started = (previous[2] if previous and previous[:2] == (fingerprint, route)
                   and self.clock() - previous[2] < self.cache_seconds else self.clock())
        self._successes[origin] = (fingerprint, route, started)


def _connection_request(request: httpx.Request) -> httpx.Request:
    extensions = dict(request.extensions)
    timeout = dict(extensions.get("timeout", {}))
    existing = timeout.get("connect")
    timeout["connect"] = CONNECT_TIMEOUT if existing is None else min(existing, CONNECT_TIMEOUT)
    extensions["timeout"] = timeout
    return httpx.Request(request.method, request.url, headers=request.headers,
                         stream=request.stream, extensions=extensions)


def _fallback_log(request: httpx.Request, route: str | None, error: Exception) -> None:
    # Neither URL paths/query strings nor proxy URLs/exception messages are safe.
    print(f"       [network-fallback] host={request.url.host} "
          f"route={'direct' if route is None else 'proxy'} error={type(error).__name__}")


class AutoAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, policy: NetworkPolicy, factory: Callable[..., httpx.AsyncBaseTransport] = httpx.AsyncHTTPTransport):
        self.policy = policy
        self.factory = factory
        self.transports: dict[str | None, httpx.AsyncBaseTransport] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        routes = self.policy.routes(str(request.url))
        failures = []
        for route in routes:
            try:
                if route not in self.transports:
                    self.transports[route] = self.factory(proxy=route, trust_env=False)
                response = await self.transports[route].handle_async_request(_connection_request(request))
            except _CONNECT_ERRORS as error:
                failures.append((route, error))
                _fallback_log(request, route, error)
                continue
            self.policy.succeeded(str(request.url), route, routes)
            return response
        raise NetworkConnectionError(failures, request=request) from failures[-1][1]

    async def aclose(self) -> None:
        for transport in self.transports.values():
            await transport.aclose()


class AutoTransport(httpx.BaseTransport):
    def __init__(self, policy: NetworkPolicy, factory: Callable[..., httpx.BaseTransport] = httpx.HTTPTransport):
        self.policy = policy
        self.factory = factory
        self.transports: dict[str | None, httpx.BaseTransport] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        routes = self.policy.routes(str(request.url))
        failures = []
        for route in routes:
            try:
                if route not in self.transports:
                    self.transports[route] = self.factory(proxy=route, trust_env=False)
                response = self.transports[route].handle_request(_connection_request(request))
            except _CONNECT_ERRORS as error:
                failures.append((route, error))
                _fallback_log(request, route, error)
                continue
            self.policy.succeeded(str(request.url), route, routes)
            return response
        raise NetworkConnectionError(failures, request=request) from failures[-1][1]

    def close(self) -> None:
        for transport in self.transports.values():
            transport.close()


def async_client(*, client_factory: Callable[..., Any] | None = None,
                 network_policy: NetworkPolicy | None = None, **kwargs: Any) -> httpx.AsyncClient:
    """Create a policy client; retain explicitly injected test/native transports."""
    kwargs["trust_env"] = False
    if kwargs.get("transport") is None:
        kwargs["transport"] = AutoAsyncTransport(network_policy or NetworkPolicy())
    return (client_factory or httpx.AsyncClient)(**kwargs)


def sync_client(*, client_factory: Callable[..., Any] | None = None,
                network_policy: NetworkPolicy | None = None, **kwargs: Any) -> httpx.Client:
    kwargs["trust_env"] = False
    if kwargs.get("transport") is None:
        kwargs["transport"] = AutoTransport(network_policy or NetworkPolicy())
    return (client_factory or httpx.Client)(**kwargs)


class _ConnectionTimeoutMixin:
    async def connect(self, req, traces, timeout: aiohttp.ClientTimeout):
        bounded = aiohttp.ClientTimeout(
            total=timeout.total, sock_read=timeout.sock_read,
            connect=min(timeout.connect or CONNECT_TIMEOUT, CONNECT_TIMEOUT),
            sock_connect=min(timeout.sock_connect or CONNECT_TIMEOUT, CONNECT_TIMEOUT),
            ceil_threshold=timeout.ceil_threshold,
        )
        return await super().connect(req, traces, bounded)


class _DirectConnector(_ConnectionTimeoutMixin, aiohttp.TCPConnector):
    pass


class _SocksConnector(_ConnectionTimeoutMixin, ProxyConnector):
    pass


@dataclass(frozen=True)
class AiohttpConnection:
    """Explicit connection options for third-party aiohttp consumers."""
    route: str | None = field(repr=False)
    connector: aiohttp.BaseConnector

    @property
    def request_options(self) -> dict[str, Any]:
        # Empty proxy explicitly disables aiohttp's trust_env proxy lookup.
        proxy = self.route or ""
        if proxy.startswith(("socks5://", "socks5h://")):
            proxy = ""
        return {"connector": self.connector, "proxy": proxy}


async def async_operation(
    url: str, operation: Callable[[AiohttpConnection], Awaitable[_T]], *,
    network_policy: NetworkPolicy | None = None,
    retry_allowed: Callable[[], bool] = lambda: True,
) -> _T:
    """Execute an aiohttp operation with the same direct-first route policy.

    Operations pass connection.request_options to their library. A streaming
    caller supplies a guard to prevent replay after any output has arrived.
    """
    policy = network_policy or NetworkPolicy()
    routes = policy.routes(url)
    failures = []
    for route in routes:
        if route and route.startswith(("socks5://", "socks5h://")):
            connector = _SocksConnector.from_url(route.replace("socks5h://", "socks5://", 1), rdns=True)
        else:
            connector = _DirectConnector()
        try:
            result = await operation(AiohttpConnection(route, connector))
        except (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError,
                ProxyConnectionError, ProxyTimeoutError, ProxyError) as error:
            if isinstance(error, aiohttp.ClientSSLError) or not retry_allowed():
                raise
            failures.append((route, error))
            _fallback_log(httpx.Request("GET", url), route, error)
            continue
        finally:
            await connector.close()
        policy.succeeded(url, route, routes)
        return result
    raise NetworkConnectionError(failures) from failures[-1][1]
