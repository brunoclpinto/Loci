import hashlib
import pytest
from loci.store import open_db


@pytest.fixture
def tmp_db(tmp_path):
    conn = open_db(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def flat_embedding():
    """384-dim embedding, all 0.1."""
    return [0.1] * 384


@pytest.fixture(scope="session")
def nlp():
    """Session-scoped spaCy pipeline (loads once for all tests)."""
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "senter"])
    except OSError:
        pytest.skip("en_core_web_sm not installed — run: uv run python -m spacy download en_core_web_sm")


class FakeEmbedder:
    """Deterministic 384-dim embedder for tests — no llama-cpp-python required."""

    def embed(self, texts, normalize=False):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16)
            vec = [((seed >> (i % 128)) & 0xFF) / 255.0 for i in range(384)]
            result.append(vec)
        return result


@pytest.fixture(scope="session")
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def default_cfg():
    """Bare default config — no file loading, no path expansion."""
    from loci.config import Config
    return Config()
