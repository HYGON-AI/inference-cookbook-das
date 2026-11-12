# SPDX-License-Identifier: Apache-2.0
"""Own both upstream operations for the lifetime of one downstream stream."""

import asyncio
import math

import anyio


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
                done, _ = await asyncio.wait(
                    watched,
                    timeout=max(0, deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    # Python 3.10 keeps asyncio.TimeoutError distinct from
                    # the builtin; match wait_for on the native DTK image.
                    raise asyncio.TimeoutError(  # noqa: UP041
                        "P/D upstream deadline exceeded"
                    )
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
        # Starlette can cancel a stream through an AnyIO task-group scope;
        # shielding ensures cleanup awaits are not cancelled a second time.
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
