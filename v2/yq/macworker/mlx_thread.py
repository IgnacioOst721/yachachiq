"""The ONE thread that touches MLX in the Mac worker.

MLX streams belong to the thread that created them: an array or model made on one thread and used
(or freed) on another fails with "There is no Stream(gpu, N) in current thread", and when that
happens while freeing memory the exception escapes into C++ and aborts the whole worker (seen on
2026-09-27 when voice's Whisper and art's Qwen3 shared the server). So every MLX load, inference
and unload in the worker runs here: ModelManager.get/unload, Whisper, the LLM, the VLM and the image
generator. One GPU user at a time also keeps the 16 GB Mac out of swap.

    run(fn, *args, **kwargs)   execute on the MLX thread and return the result (re-entrant)
    on_mlx_thread()            True when already running there
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_tls = threading.local()


def on_mlx_thread() -> bool:
    return getattr(_tls, "on_mlx", False)


def run(fn: Callable, *args, **kwargs):
    """Execute fn on the dedicated MLX thread and wait for the result. Calling it from the MLX
    thread itself just calls fn (no deadlock when an MLX task loads a model)."""
    if on_mlx_thread():
        return fn(*args, **kwargs)

    def call():
        _tls.on_mlx = True
        try:
            return fn(*args, **kwargs)
        finally:
            _tls.on_mlx = False

    return _executor.submit(call).result()
