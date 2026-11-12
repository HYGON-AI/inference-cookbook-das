# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Standalone Mooncake P/D proxy for HY4."""

import argparse
import asyncio
import ipaddress
import itertools
import logging
import math
import os
import urllib.parse
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import anyio
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM_TIMEOUT = float(os.environ.get("HY4_PD_PROXY_TIMEOUT_SECONDS", "300"))
if not math.isfinite(UPSTREAM_TIMEOUT) or UPSTREAM_TIMEOUT <= 0:
    raise ValueError("HY4_PD_PROXY_TIMEOUT_SECONDS must be finite and positive")
logger = logging.getLogger(__name__)


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


def make_http_path(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def prefiller_cycle(prefill_clients: list[Any]):
    while True:
        for prefill_client in prefill_clients:
            for i in range(prefill_client["dp_size"]):
                yield (prefill_client, i)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_raw",
        metavar=("URL", "bootstrap_port"),
        help="Prefill server URL and optional bootstrap port. Can be specified multiple times. Format: --prefill URL [BOOTSTRAP_PORT]. BOOTSTRAP_PORT can be a port number, 'none', or omitted (defaults to none).",
    )
    parser.add_argument(
        "--decode",
        nargs=1,
        action="append",
        dest="decode_raw",
        metavar=("URL",),
        help="Decode server URL. Can be specified multiple times.",
    )
    args = parser.parse_args()
    args.prefill = _parse_prefill_urls(args.prefill_raw)
    args.decode = _parse_decode_urls(args.decode_raw)
    return args


def _parse_prefill_urls(prefill_list):
    """Parse prefill URLs from --prefill arguments.

    Format: --prefill URL [BOOTSTRAP_PORT]
    Example:
        --prefill http://prefill1:8080 9000  # With bootstrap port
        --prefill http://prefill2:8080 none  # Explicitly no bootstrap port
        --prefill http://prefill3:8080       # Defaults to no bootstrap port
    """
    if not prefill_list:
        return []
    prefill_urls = []
    for prefill_args in prefill_list:
        url = prefill_args[0]
        if len(prefill_args) >= 2:
            bootstrap_port_str = prefill_args[1]
            if bootstrap_port_str.lower() == "none":
                bootstrap_port = None
            else:
                try:
                    bootstrap_port = int(bootstrap_port_str)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bootstrap port: {bootstrap_port_str}. Must be a number or 'none'"
                    ) from e
        else:
            bootstrap_port = None
        prefill_urls.append((url, bootstrap_port))
    return prefill_urls


def _parse_decode_urls(decode_list):
    """Parse decode URLs from --decode arguments.

    Format: --decode URL
    Example: --decode http://decode1:8081 --decode http://decode2:8081
    """
    if not decode_list:
        return []
    return [url[0] for url in decode_list]


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        return next(app.state.prefill_iterator)
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(
    client_info: dict, dp_rank: int, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": f"xfer-{request_id}",
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        "X-data-parallel-rank": str(dp_rank),
    }
    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()
    await response.aclose()


async def stream_service_response(
    prefill_client_info: dict,
    prefill_dp_rank: int,
    decode_client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    req_data["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
        "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
        "transfer_id": f"xfer-{request_id}",
    }
    async with decode_client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def owned_pd_stream(prefill, decode, *, timeout=300):
    """Start P/D concurrently, propagate failures, and join tasks on exit.

    Arguments are zero-argument factories, so discarding an unconsumed response
    does not leave a prefill coroutine/task behind. The deadline bounds upstream
    waits; downstream disconnect cancellation is propagated by the ASGI server.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("P/D upstream timeout must be finite and positive")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    producer = pending = iterator = None
    try:
        producer = asyncio.create_task(prefill())
        iterator = decode().__aiter__()
        producer_pending = True
        while True:
            pending = asyncio.create_task(iterator.__anext__())
            while True:
                watched = {pending}
                if producer_pending:
                    watched.add(producer)
                (done, _) = await asyncio.wait(
                    watched,
                    timeout=max(0, deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise asyncio.TimeoutError("P/D upstream deadline exceeded")  # noqa: UP041 - Python 3.10
                if producer_pending and producer in done:
                    producer.result()
                    producer_pending = False
                if pending in done:
                    break
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                pending = None
                break
            pending = None
            yield chunk
        if producer_pending:
            await asyncio.wait_for(producer, timeout=max(0, deadline - loop.time()))
    finally:
        with anyio.CancelScope(shield=True):
            tasks = [task for task in (pending, producer) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if iterator is not None:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()


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
                    int(rank): item["engine_id"] for (rank, item) in payload.items()
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
async def lifespan(app):
    app.state.ready = asyncio.Event()
    app.state.prefill_clients = []
    app.state.decode_clients = []
    clients = AsyncExitStack()
    initializer = None

    async def client(url):
        return await clients.enter_async_context(
            httpx.AsyncClient(
                timeout=None,
                trust_env=False,
                base_url=url,
                limits=httpx.Limits(
                    max_connections=None,
                    max_keepalive_connections=None,
                    keepalive_expiry=1.0,
                ),
            )
        )

    try:
        if not global_args.prefill or not global_args.decode:
            raise ValueError("at least one prefiller and decoder are required")
        for url, port in global_args.prefill:
            host = maybe_wrap_ipv6_address(urllib.parse.urlparse(url).hostname)
            app.state.prefill_clients.append(
                {
                    "client": await client(url),
                    "url": url,
                    "bootstrap_addr": make_http_path(host, port or 8998),
                    "dp_engine_id": {},
                }
            )
        for url in global_args.decode:
            app.state.decode_clients.append({"client": await client(url)})
        app.state.prefill_iterator = prefiller_cycle(app.state.prefill_clients)
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


class OwnedStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()


async def _handle_completions(api: str, request: Request):
    if not request.app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")
    data = await request.json()
    request_id = str(uuid.uuid4())
    (prefill, rank) = get_next_client(request.app, "prefill")
    decode = get_next_client(request.app, "decode")
    stream = owned_pd_stream(
        lambda: send_request_to_service(prefill, rank, api, data.copy(), request_id),
        lambda: stream_service_response(
            prefill, rank, decode, api, data.copy(), request_id
        ),
        timeout=UPSTREAM_TIMEOUT,
    )
    return OwnedStreamingResponse(stream, media_type="application/json")


app = FastAPI(lifespan=lifespan)


@app.api_route("/metrics", methods=["GET"])
@app.api_route("/v1/models", methods=["GET"])
@app.api_route("/tokenize", methods=["POST"])
@app.api_route("/detokenize", methods=["POST"])
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


@app.get("/health")
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
            current = {int(rank): item["engine_id"] for (rank, item) in payload.items()}
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


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


if __name__ == "__main__":
    global_args = parse_args()
    uvicorn.run(
        app,
        host=global_args.host,
        port=global_args.port,
        access_log=False,
        timeout_keep_alive=60,
    )
