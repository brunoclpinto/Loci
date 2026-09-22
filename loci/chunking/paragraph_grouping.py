from loci.chunking.splitter import TextChunk, split_text


def group_paragraphs(paragraphs: list[str], target_tokens: int) -> list[TextChunk]:
    """Greedily accumulates consecutive paragraphs into chunks up to
    target_tokens words, never splitting a paragraph across two chunks —
    unlike the word-count-only split_text(), every chunk boundary here lands
    on a paragraph boundary. Used by the relationship-extraction pass, which
    needs units small enough for precise pronoun/reference resolution
    without cutting a sentence in half.

    A single paragraph that alone exceeds target_tokens is the one case
    that can't respect a paragraph boundary; it falls back to split_text()
    on just that paragraph."""
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for paragraph in paragraphs:
        words = len(paragraph.split())
        if words > target_tokens:
            if buffer:
                chunks.append(TextChunk(text="\n\n".join(buffer), token_count=buffer_tokens))
                buffer, buffer_tokens = [], 0
            chunks.extend(split_text(paragraph, target_tokens, overlap_tokens=0))
            continue
        if buffer and buffer_tokens + words > target_tokens:
            chunks.append(TextChunk(text="\n\n".join(buffer), token_count=buffer_tokens))
            buffer, buffer_tokens = [], 0
        buffer.append(paragraph)
        buffer_tokens += words

    if buffer:
        chunks.append(TextChunk(text="\n\n".join(buffer), token_count=buffer_tokens))

    return chunks
