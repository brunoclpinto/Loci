from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceDescriptor:
    """Where the data is. A file path today; a connection string or API
    endpoint for future live-DB/API adapters."""

    path: Path
    context_name: str
    forced_adapter: str | None = None


@dataclass
class ExtractedEntity:
    local_id: str
    type: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    scope: str = "context_local"
    confidence: float | None = None


@dataclass
class ExtractedRelationship:
    subject_local_id: str
    predicate: str
    object_local_id: str | None = None
    object_literal: dict | None = None
    attributes: dict = field(default_factory=dict)
    confidence: float | None = None


@dataclass
class ExtractedChunk:
    text: str
    token_count: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionBatch:
    context_name: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    chunks: list[ExtractedChunk] = field(default_factory=list)
    source_ref: str = ""


class SourceAdapter(ABC):
    """Common contract every ingestion source implements. Adapters never
    talk to the resolver, embedder, or Postgres directly — they only produce
    ExtractionBatches; IngestionPipeline owns everything downstream."""

    name: str

    @abstractmethod
    def can_handle(self, source: SourceDescriptor) -> bool: ...

    @abstractmethod
    def extract(self, source: SourceDescriptor) -> Iterable[ExtractionBatch]:
        """Lazily yield batches so large/streamed sources and future
        live-DB/API adapters aren't forced to materialize everything."""
