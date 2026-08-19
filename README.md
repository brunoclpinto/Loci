# Loci

**A local assistant whose memory lives on disk, not in your RAM.**

Loci is a low-footprint, fully local RAG system for modest machines (built and
tuned on a Mac mini M1 with 8GB). The goal: a small quantized LLM paired with
a knowledge base that lives on SSD, not in RAM, so it can answer with
precision without locking up your system.

The name comes from the *method of loci*, the ancient memory-palace technique:
knowledge filed in organized locations and retrieved by walking to the right
spot.

> **Status: fresh start.** The previous implementation and build plan have
> been retired. Same goal, new approach — details to follow as the design
> takes shape.

---

## Why

Most local RAG stacks assume RAM is cheap: PyTorch runtimes, in-memory vector
indexes, embedding servers. On an 8GB machine that means swapping, beachballs,
and an unusable computer while the assistant "thinks."

Loci flips the assumption:

- **Knowledge lives on SSD**, not in RAM. Idle RAM cost: effectively zero.
- **No PyTorch anywhere.** Chat and embedding models run as GGUF files
  through llama.cpp, so weights page from disk on demand.
- **Models load only when needed** and are freed when done.
- **Knowledge grows by connecting, not rebuilding.**

## Requirements

- macOS on Apple Silicon (primary target; Linux should work, untested)
- ~8GB RAM (tighter setups are a design goal, not an afterthought)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)

## License

TBD.
