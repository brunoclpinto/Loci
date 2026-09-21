from pathlib import Path
from typing import Any

# QnA `book` slug -> raw filename under benchWork/raw, confirmed by reading
# each file's Project Gutenberg title header.
BOOK_FILES: dict[str, str] = {
    "a_study_in_scarlet": "pg244_a_study_in_scarlet.txt",
    "the_sign_of_four": "pg2097_the_sign_of_four.txt",
    "adventures_of_sherlock_holmes": "pg48320.txt",
    "memoirs_of_sherlock_holmes": "pg834_memoirs_of_sherlock_holmes.txt",
    "return_of_sherlock_holmes": "pg221_the_return_of_sherlock_holmes.txt",
    "hound_of_the_baskervilles": "pg2852_the_hound_of_the_baskervilles.txt",
    "valley_of_fear": "pg3289_the_valley_of_fear.txt",
    "his_last_bow": "pg2350_his_last_bow.txt",
    "case_book_of_sherlock_holmes": "pg69700_the_case_book_of_sherlock_holmes.txt",
}


def books_for_qna(qna_items: list[dict[str, Any]]) -> list[str]:
    """Distinct book slugs actually referenced by a QnA set. The special
    slug "corpus" (used by multi-book questions) expands to every book."""
    slugs = {item["book"] for item in qna_items}
    if "corpus" in slugs:
        slugs.discard("corpus")
        slugs.update(BOOK_FILES)
    return sorted(slugs)


def raw_path(corpus_dir: Path, book: str) -> Path:
    try:
        filename = BOOK_FILES[book]
    except KeyError:
        raise ValueError(f"unknown book slug {book!r}; known: {sorted(BOOK_FILES)}") from None
    return corpus_dir / filename
