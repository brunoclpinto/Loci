from dataclasses import dataclass

# Approximate token counting: word count. Good enough for chunk-sizing
# purposes without pulling in a tokenizer that wouldn't match the local
# model's actual vocabulary anyway.


@dataclass
class TextChunk:
    text: str
    token_count: int


def split_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[TextChunk]:
    words = text.split()
    if not words:
        return []

    step = max(chunk_tokens - overlap_tokens, 1)
    chunks: list[TextChunk] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_tokens]
        chunks.append(TextChunk(text=" ".join(window), token_count=len(window)))
        if start + chunk_tokens >= len(words):
            break
        start += step
    return chunks
