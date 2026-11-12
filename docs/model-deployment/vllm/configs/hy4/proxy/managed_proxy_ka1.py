# SPDX-License-Identifier: Apache-2.0
"""Corrected task ownership around the frozen Mooncake KA1/retries0 proxy.

This entry point was not used for the historical performance tables. It starts
P/D concurrently, adds bounded upstream waits and cancels/joins unfinished work
on a failed or disconnected stream. It does not write historical timing files.
"""

import asyncio
import importlib.util
import itertools
import logging
import math
import os
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import anyio
import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from proxy_task_lifecycle import owned_pd_stream

spec = importlib.util.spec_from_file_location(
    "hy4_frozen_mooncake_proxy",
    Path(__file__).with_name("mooncake_connector_proxy_ka1.py"),
)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
UPSTREAM_TIMEOUT = float(os.environ.get("HY4_PD_PROXY_TIMEOUT_SECONDS", "300"))
if not math.isfinite(UPSTREAM_TIMEOUT) or UPSTREAM_TIMEOUT <= 0:
    raise ValueError("HY4_PD_PROXY_TIMEOUT_SECONDS must be finite and positive")
logger = logging.getLogger(__name__)


async def initialize_prefillers(app):
    for info in app.state.prefill_clients:
        while True:
            try:
                response = await info["client"].get("/health", timeout=5)
                response.raise_for_status()
                response = await info["client"].get(
                    info["bootstrap_addr"] + "/query", timeout=5
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not payload:
                    raise ValueError("bootstrap must return a nonempty engine map")
                engines = {
                    int(rank): item["engine_id"] for rank, item in payload.items()
                }
                if (
                    len(engines) != len(payload)
                    or set(engines) != set(range(len(engines)))
                    or any(
                        not isinstance(value, str) or not value
                        for value in engines.values()
                    )
                ):
                    raise ValueError(
                        "bootstrap requires contiguous ranks and engine IDs"
                    )
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                logger.warning("Waiting for prefiller %s: %s", info["url"], error)
                await asyncio.sleep(1)
                continue
            info["dp_engine_id"] = engines
            info["dp_size"] = len(engines)
            break
    app.state.ready.set()


@asynccontextmanager
async def managed_lifespan(app):
    app.state.ready = asyncio.Event()
    app.state.prefill_clients = []
    app.state.decode_clients = []
    clients = AsyncExitStack()
    initializer = None

    async def client(url):
        return await clients.enter_async_context(
            httpx.AsyncClient(
                timeout=None,
                base_url=url,
                limits=httpx.Limits(
                    max_connections=None,
                    max_keepalive_connections=None,
                    keepalive_expiry=1.0,
                ),
            )
        )

    try:
        if not proxy.global_args.prefill or not proxy.global_args.decode:
            raise ValueError("at least one prefiller and decoder are required")
        for url, port in proxy.global_args.prefill:
            host = proxy.maybe_wrap_ipv6_address(
                proxy.urllib.parse.urlparse(url).hostname
            )
            app.state.prefill_clients.append(
                {
                    "client": await client(url),
                    "url": url,
                    "bootstrap_addr": proxy.make_http_path(host, port or 8998),
                    "dp_engine_id": {},
                }
            )
        for url in proxy.global_args.decode:
            app.state.decode_clients.append({"client": await client(url)})
        app.state.prefill_iterator = proxy.prefiller_cycle(app.state.prefill_clients)
        app.state.decode_iterator = itertools.cycle(
            range(len(app.state.decode_clients))
        )
        initializer = asyncio.create_task(initialize_prefillers(app))
        yield
    finally:
        with anyio.CancelScope(shield=True):
            app.state.ready.clear()
            if initializer is not None:
                initializer.cancel()
                await asyncio.gather(initializer, return_exceptions=True)
            await clients.aclose()


proxy.app.router.lifespan_context = managed_lifespan


class OwnedStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Disconnect can arrive during send(), outside the suspended
            # iterator. Explicitly close it even under an ASGI cancel scope.
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()


async def managed_handle(api: str, request: Request):
    if not request.app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")
    data = await request.json()
    request_id = str(uuid.uuid4())
    prefill, rank = proxy.get_next_client(request.app, "prefill")
    decode = proxy.get_next_client(request.app, "decode")
    stream = owned_pd_stream(
        lambda: proxy.send_request_to_service(
            prefill, rank, api, data.copy(), request_id
        ),
        lambda: proxy.stream_service_response(
            prefill, rank, decode, api, data.copy(), request_id
        ),
        timeout=UPSTREAM_TIMEOUT,
    )
    return OwnedStreamingResponse(stream, media_type="application/json")


proxy._handle_completions = managed_handle


@proxy.app.api_route("/metrics", methods=["GET"])
@proxy.app.api_route("/v1/models", methods=["GET"])
@proxy.app.api_route("/tokenize", methods=["POST"])
@proxy.app.api_route("/detokenize", methods=["POST"])
async def forward_auxiliary(request: Request):
    client = request.app.state.decode_clients[0]["client"]
    response = await client.request(
        request.method,
        request.url.path,
        content=await request.body(),
        headers={
            "content-type": request.headers.get("content-type", "application/json")
        },
        timeout=UPSTREAM_TIMEOUT,
    )
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers={
            "content-type": response.headers.get("content-type", "application/json")
        },
    )


@proxy.app.get("/health")
async def health(request: Request):
    if not request.app.state.ready.is_set():
        return Response(status_code=503)
    try:
        for info in request.app.state.prefill_clients:
            response = await info["client"].get(
                info["bootstrap_addr"] + "/query", timeout=5
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return Response(status_code=503)
            current = {int(rank): item["engine_id"] for rank, item in payload.items()}
            if current != info["dp_engine_id"]:
                return Response(
                    content="Producer engine_id changed; restart proxy", status_code=503
                )
        response = await request.app.state.decode_clients[0]["client"].get(
            "/health", timeout=5
        )
        return Response(status_code=response.status_code)
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return Response(status_code=503)


if __name__ == "__main__":
    import uvicorn

    proxy.global_args = proxy.parse_args()
    uvicorn.run(
        proxy.app,
        host=proxy.global_args.host,
        port=proxy.global_args.port,
        access_log=False,
    )
