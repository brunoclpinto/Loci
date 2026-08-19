import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Context(Base):
    __tablename__ = "contexts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ContextRelationship(Base):
    __tablename__ = "context_relationships"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    parent_context_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("contexts.id", ondelete="CASCADE"), nullable=False)
    child_context_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("contexts.id", ondelete="CASCADE"), nullable=False)
    relation_type: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")

    __table_args__ = (
        UniqueConstraint("parent_context_id", "child_context_id", "relation_type"),
    )


class EntityType(Base):
    __tablename__ = "entity_types"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text)
    current_schema_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EntityTypeSchemaVersion(Base):
    __tablename__ = "entity_type_schema_versions"

    entity_type: Mapped[str] = mapped_column(ForeignKey("entity_types.name", ondelete="CASCADE"), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    json_schema: Mapped[dict] = mapped_column(JSONB, nullable=False)
    changelog: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PredicateVocabulary(Base):
    __tablename__ = "predicate_vocabulary"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text)
    domain_entity_types: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    range_entity_types: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    component: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    is_active: Mapped[bool] = mapped_column(nullable=False, server_default="true")
    activated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("component IN ('resolver','embedding_model','extraction_llm')", name="ck_model_registry_component"),
        UniqueConstraint("component", "name", "version"),
    )


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    type: Mapped[str] = mapped_column(ForeignKey("entity_types.name"), nullable=False)
    context_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("contexts.id"), nullable=False)
    scope: Mapped[str] = mapped_column(Text, nullable=False, server_default="context_local")
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    attribute_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    resolver_version: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model_version: Mapped[str | None] = mapped_column(Text)
    source_adapter: Mapped[str | None] = mapped_column(Text)
    source_ref: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("scope IN ('context_local','cross_context')", name="ck_entities_scope"),
        ForeignKeyConstraint(
            ["type", "attribute_schema_version"],
            ["entity_type_schema_versions.entity_type", "entity_type_schema_versions.version"],
        ),
        Index("idx_entities_type_context", "type", "context_id"),
        Index("idx_entities_scope", "scope"),
        Index("idx_entities_name_trgm", "canonical_name", postgresql_using="gin", postgresql_ops={"canonical_name": "gin_trgm_ops"}),
        Index("idx_entities_attributes", "attributes", postgresql_using="gin", postgresql_ops={"attributes": "jsonb_path_ops"}),
    )


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    subject_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    predicate: Mapped[str] = mapped_column(ForeignKey("predicate_vocabulary.name"), nullable=False)
    object_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    object_literal: Mapped[dict | None] = mapped_column(JSONB)
    context_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("contexts.id"), nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    confidence: Mapped[float | None] = mapped_column(Numeric)
    source_adapter: Mapped[str | None] = mapped_column(Text)
    source_ref: Mapped[str | None] = mapped_column(Text)
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    resolver_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("object_id IS NOT NULL OR object_literal IS NOT NULL", name="ck_relationships_object"),
        Index("idx_rel_subject", "subject_id"),
        Index("idx_rel_object", "object_id"),
        Index("idx_rel_predicate_context", "predicate", "context_id"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    context_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("contexts.id"), nullable=False)
    source_adapter: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding_model_version: Mapped[str] = mapped_column(Text, nullable=False)
    qdrant_point_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_chunks_context", "context_id"),)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    adapter: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    context_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("contexts.id"))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("status IN ('running','succeeded','failed')", name="ck_ingestion_runs_status"),
    )
