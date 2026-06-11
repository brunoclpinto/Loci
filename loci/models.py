"""Model loading: context managers for embedder and chat LLM via llama.cpp."""
from __future__ import annotations

import gc
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from loci.config import Config


def _require_llama():
    try:
        from llama_cpp import Llama  # noqa: F401
        return Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for model inference.\n"
            "Install with Metal support:\n"
            "  brew install cmake\n"
            "  CMAKE_ARGS=\"-DGGML_METAL=on\" uv add llama-cpp-python"
        ) from exc


@contextmanager
def load_embedder(cfg: Config) -> Generator[Any, None, None]:
    """Load the GGUF embedding model; unload and free memory on exit."""
    Llama = _require_llama()
    model_path = cfg.paths.models_dir.expanduser() / cfg.models.embedder
    if not model_path.exists():
        raise FileNotFoundError(
            f"Embedder model not found: {model_path}\n"
            "Run: loci models pull"
        )
    model = Llama(
        model_path=str(model_path),
        embedding=True,
        n_ctx=512,
        n_gpu_layers=cfg.models.n_gpu_layers,
        use_mmap=cfg.models.use_mmap,
        verbose=False,
    )
    try:
        yield model
    finally:
        del model
        gc.collect()


@contextmanager
def load_chat(cfg: Config, *, low_mem: bool = False) -> Generator[Any, None, None]:
    """Load the GGUF chat model; unload and free memory on exit."""
    Llama = _require_llama()
    model_name = cfg.models.chat_low_mem if low_mem else cfg.models.chat
    n_ctx = 2048 if low_mem else cfg.models.n_ctx
    model_path = cfg.paths.models_dir.expanduser() / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Chat model not found: {model_path}\n"
            "Run: loci models pull"
        )
    model = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=cfg.models.n_gpu_layers,
        use_mmap=cfg.models.use_mmap,
        verbose=False,
    )
    try:
        yield model
    finally:
        del model
        gc.collect()


def embed_batch(model: Any, texts: list[str], *, normalize: bool = True) -> list[list[float]]:
    """Embed a list of texts; always returns List[List[float]]."""
    if not texts:
        return []
    result = model.embed(texts, normalize=normalize)
    # llama-cpp-python returns List[float] for a single string and
    # List[List[float]] for a list. If a list was passed but a flat list
    # came back (older API), wrap it.
    if texts and not isinstance(result[0], (list, tuple)):
        return [list(result)]
    return [list(v) for v in result]


def cosine_dist_threshold(sim_threshold: float) -> float:
    """Convert cosine similarity threshold to L2 distance threshold.

    On unit-normalized vectors, L2 distance and cosine similarity relate by:
      L2² = 2 * (1 - cos_sim)
    """
    return math.sqrt(2.0 * (1.0 - sim_threshold))
