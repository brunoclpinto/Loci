"""Layered configuration: CLI override > env var (LOCI_SECTION__KEY) > config.toml > default."""

import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "loci"
    password: str = ""
    dbname: str = "loci"

    @property
    def dsn(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = 6333
    prefer_grpc: bool = False


class OllamaSettings(BaseModel):
    host: str = "localhost"
    port: int = 11434
    chat_model: str = "qwen2.5:7b-instruct"
    embedding_model: str = "nomic-embed-text"
    request_timeout_s: int = 900

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class McpSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    transport: str = "streamable-http"


class RegistrySettings(BaseModel):
    schemas_dir: str = "loci/schema_registry/seed_schemas"


class ResolverSettings(BaseModel):
    name_similarity_threshold: float = 0.85
    embedding_similarity_threshold: float = 0.90


class IngestSettings(BaseModel):
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    extraction_window_tokens: int = 2500
    # Two-pass extraction's relationship pass groups paragraphs up to this
    # size (never splitting a paragraph across units) instead of reusing
    # extraction_window_tokens, since relationship extraction benefits from
    # a tighter span for precise pronoun/reference resolution.
    relationship_unit_tokens: int = 400
    embed_batch: int = 16
    extraction_json_retries: int = 5
    # Segment classification and coreference resolution are simple
    # classification/matching tasks, not full extraction — no reason to pay
    # a heavy reasoning model's latency for them regardless of which model
    # is doing extraction for a given run.
    segment_classifier_model: str = "qwen2.5:7b-instruct"
    coreference_model: str = "qwen2.5:7b-instruct"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    jsonl: bool = True


class BenchWeights(BaseModel):
    """Composite-score weights. Kept config-driven (not hardcoded) since the
    right balance between answer quality and resource cost is a judgment
    call that will change as models get trimmed down."""

    answer_quality: float = 0.6
    ingest_time: float = 0.1
    retrieval_time: float = 0.1
    knowledge_space: float = 0.05
    ram: float = 0.075
    vram: float = 0.075


class BenchSettings(BaseModel):
    corpus_dir: str = "benchWork/raw"
    qna_dir: str = "benchWork/bench"
    qna_file: str = "qna_scarlet.json"
    log_dir: str = "benchWork/logs"
    debug_dir: str = "benchWork/debug"
    # qwen2.5:14b-instruct beat deepseek-r1:14b outright as an extraction
    # model too: a full 2x2 matrix against every answer-model pairing showed
    # it winning independently in both roles (85.0 vs 59.5 quality on the
    # same-model pairing), while ingesting ~26x faster (505s vs 13347s) with
    # a comparable relationship yield (256 vs 279) and zero stale-id
    # warnings on either model.
    default_extraction_model: str = "qwen2.5:14b-instruct"
    # qwen2.5:14b-instruct beat gemma4:26b outright on this hardware (higher
    # quality score, zero empty answers vs 5/100, ~5x faster) — gemma4:26b
    # was too large to share this GPU's VRAM with the embedding model
    # without contention, silently truncating some answers to empty.
    default_answer_model: str = "qwen2.5:14b-instruct"
    sample_interval_s: float = 5.0
    weights: BenchWeights = BenchWeights()


def _config_file_path() -> Path | None:
    env_path = os.environ.get("LOCI_CONFIG_FILE")
    if env_path:
        return Path(env_path)
    default = Path.home() / ".loci" / "config.toml"
    return default if default.exists() else None


class LociSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCI_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    database: DatabaseSettings = DatabaseSettings()
    qdrant: QdrantSettings = QdrantSettings()
    ollama: OllamaSettings = OllamaSettings()
    mcp: McpSettings = McpSettings()
    registry: RegistrySettings = RegistrySettings()
    resolver: ResolverSettings = ResolverSettings()
    ingest: IngestSettings = IngestSettings()
    logging: LoggingSettings = LoggingSettings()
    bench: BenchSettings = BenchSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence, highest first: CLI-passed kwargs (init) > env vars > config.toml > defaults
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        toml_path = _config_file_path()
        if toml_path is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        return tuple(sources)


def load_settings(**cli_overrides) -> LociSettings:
    return LociSettings(**cli_overrides)
