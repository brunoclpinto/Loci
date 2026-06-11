"""Tests for loci/ingest.py: pipeline, dedup, chunking, fact round-trip."""
import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from loci.ingest import hash_file, ingest_file, make_chunks, read_file


# ---------------------------------------------------------------------------
# Small corpus fixture
# ---------------------------------------------------------------------------

CORPUS = """\
Sherlock Holmes took his bottle from the corner of the mantel-piece.
Watson entered the room and observed Holmes carefully.
Holmes placed the syringe in its neat morocco case.
Mrs. Hudson knocked on the door and brought some tea.
Holmes did not answer the door immediately.
Watson examined the case that Holmes had left on the table.
"""


@pytest.fixture
def corpus_file(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(CORPUS)
    return p


@pytest.fixture
def md_file(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Notes\n\n" + CORPUS)
    return p


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_reads_txt(self, corpus_file):
        text = read_file(corpus_file)
        assert "Holmes" in text

    def test_reads_md(self, md_file):
        text = read_file(md_file)
        assert "Holmes" in text

    def test_rejects_unsupported(self, tmp_path):
        p = tmp_path / "doc.xyz"
        p.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            read_file(p)

    def test_pdf_raises_import_error_without_pypdf(self, tmp_path, monkeypatch):
        """Without pypdf installed, reading a PDF raises ImportError with install hint."""
        import sys
        monkeypatch.setitem(sys.modules, "pypdf", None)
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4")
        with pytest.raises(ImportError, match="pypdf"):
            read_file(p)

    def test_epub_raises_import_error_without_ebooklib(self, tmp_path, monkeypatch):
        """Without ebooklib installed, reading an EPUB raises ImportError."""
        import sys
        monkeypatch.setitem(sys.modules, "ebooklib", None)
        p = tmp_path / "book.epub"
        p.write_bytes(b"PK")
        with pytest.raises(ImportError, match="ebooklib"):
            read_file(p)


# ---------------------------------------------------------------------------
# hash_file
# ---------------------------------------------------------------------------

class TestHashFile:
    def test_consistent(self, corpus_file):
        assert hash_file(corpus_file) == hash_file(corpus_file)

    def test_differs_on_change(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello")
        b.write_text("world")
        assert hash_file(a) != hash_file(b)


# ---------------------------------------------------------------------------
# make_chunks
# ---------------------------------------------------------------------------

class TestMakeChunks:
    def test_produces_chunks(self, nlp):
        doc = nlp(CORPUS)
        chunks = make_chunks(doc, target_tokens=50, overlap_sentences=1)
        assert len(chunks) >= 2

    def test_chunk_text_nonempty(self, nlp):
        doc = nlp(CORPUS)
        for text, sents in make_chunks(doc, target_tokens=100):
            assert text.strip()
            assert sents

    def test_overlap_sentence_appears_in_consecutive(self, nlp):
        doc = nlp(CORPUS)
        chunks = make_chunks(doc, target_tokens=80, overlap_sentences=1)
        if len(chunks) >= 2:
            # Last sentence of chunk[0] should appear at start of chunk[1]
            last_sent = chunks[0][1][-1].text.strip()
            first_sent_next = chunks[1][1][0].text.strip()
            assert last_sent == first_sent_next

    def test_single_chunk_for_short_text(self, nlp):
        doc = nlp("Holmes took the bottle.")
        chunks = make_chunks(doc, target_tokens=512)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# ingest_file — pipeline
# ---------------------------------------------------------------------------

class TestIngestFile:
    def test_returns_stats(self, tmp_db, corpus_file, fake_embedder, nlp, default_cfg):
        stats = ingest_file(
            corpus_file, cfg=default_cfg, conn=tmp_db,
            embedder=fake_embedder, spacy_nlp=nlp,
        )
        assert not stats["skipped"]
        assert stats["chunks"] > 0
        assert stats["facts"] > 0
        assert stats["entities_new"] > 0

    def test_reingest_is_noop(self, tmp_db, corpus_file, fake_embedder, nlp, default_cfg):
        ingest_file(corpus_file, cfg=default_cfg, conn=tmp_db,
                    embedder=fake_embedder, spacy_nlp=nlp)
        stats2 = ingest_file(corpus_file, cfg=default_cfg, conn=tmp_db,
                             embedder=fake_embedder, spacy_nlp=nlp)
        assert stats2["skipped"]
        assert stats2["chunks"] == 0

    def test_chunks_stored_in_db(self, tmp_db, corpus_file, fake_embedder, nlp, default_cfg):
        ingest_file(corpus_file, cfg=default_cfg, conn=tmp_db,
                    embedder=fake_embedder, spacy_nlp=nlp)
        count = tmp_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count > 0

    def test_embeddings_stored(self, tmp_db, corpus_file, fake_embedder, nlp, default_cfg):
        ingest_file(corpus_file, cfg=default_cfg, conn=tmp_db,
                    embedder=fake_embedder, spacy_nlp=nlp)
        count = tmp_db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        assert count > 0

    def test_fts_populated(self, tmp_db, corpus_file, fake_embedder, nlp, default_cfg):
        ingest_file(corpus_file, cfg=default_cfg, conn=tmp_db,
                    embedder=fake_embedder, spacy_nlp=nlp)
        from loci.store import fts_search_chunks
        results = fts_search_chunks(tmp_db, query="Holmes", k=5)
        assert results

    def test_metadata_stored(self, tmp_db, corpus_file, fake_embedder, nlp, default_cfg):
        ingest_file(corpus_file, cfg=default_cfg, conn=tmp_db,
                    embedder=fake_embedder, spacy_nlp=nlp,
                    meta={"title": "Test Book", "author": "Conan Doyle"})
        row = tmp_db.execute("SELECT title, author FROM sources").fetchone()
        assert row["title"] == "Test Book"
        assert row["author"] == "Conan Doyle"

    def test_no_embedder_still_stores_facts(self, tmp_db, corpus_file, nlp, default_cfg):
        """Without an embedder, facts and FTS still work."""
        stats = ingest_file(corpus_file, cfg=default_cfg, conn=tmp_db,
                            embedder=None, spacy_nlp=nlp)
        assert stats["facts"] > 0
        # No vec entries expected
        assert tmp_db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 0


class TestCanonicalFact:
    def test_spec_fact_in_db(self, tmp_db, fake_embedder, nlp, default_cfg, tmp_path):
        """The spec example sentence must produce (sherlock holmes, take, bottle, from=...)."""
        p = tmp_path / "spec.txt"
        p.write_text(
            "Sherlock Holmes took his bottle from the corner of the mantel-piece."
        )
        ingest_file(p, cfg=default_cfg, conn=tmp_db,
                    embedder=fake_embedder, spacy_nlp=nlp)

        facts = tmp_db.execute(
            "SELECT f.predicate, f.object_text, f.qualifiers, e.canonical_name "
            "FROM facts f JOIN entities e ON f.subject_id = e.id "
            "WHERE f.predicate = 'take'"
        ).fetchall()

        assert facts, "no 'take' facts found"
        bottle_fact = next(
            (f for f in facts if f["object_text"] == "bottle"), None
        )
        assert bottle_fact is not None, f"expected object_text='bottle' in {[dict(f) for f in facts]}"

        import json
        quals = json.loads(bottle_fact["qualifiers"] or "{}")
        assert "from" in quals
        assert "corner" in quals["from"]


class TestEntityResolutionAcrossChunks:
    def test_same_entity_linked(self, tmp_db, fake_embedder, nlp, default_cfg, tmp_path):
        """'Holmes', 'Sherlock Holmes', 'Mr. Sherlock Holmes' → one entity ≥3 aliases."""
        p = tmp_path / "multi.txt"
        p.write_text(
            "Sherlock Holmes entered the room.\n"
            "Holmes sat by the fire.\n"
            "Mr. Sherlock Holmes picked up the newspaper.\n"
        )
        ingest_file(p, cfg=default_cfg, conn=tmp_db,
                    embedder=fake_embedder, spacy_nlp=nlp)

        # All three mentions must resolve to the same entity
        entity_count = tmp_db.execute("SELECT COUNT(DISTINCT entity_id) FROM aliases "
                                      "WHERE alias IN ('sherlock holmes','holmes')").fetchone()[0]
        assert entity_count == 1, "Holmes and Sherlock Holmes must be the same entity"

        alias_count = tmp_db.execute(
            "SELECT COUNT(*) FROM aliases WHERE entity_id=("
            "  SELECT entity_id FROM aliases WHERE alias='sherlock holmes'"
            ")"
        ).fetchone()[0]
        assert alias_count >= 3, f"expected ≥3 aliases, got {alias_count}"
