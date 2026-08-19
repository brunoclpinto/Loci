"""init core schema

Revision ID: 0001
Revises:
Create Date: 2026-08-19

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "contexts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "context_relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("parent_context_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("contexts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("child_context_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("contexts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.UniqueConstraint("parent_context_id", "child_context_id", "relation_type"),
    )

    op.create_table(
        "entity_types",
        sa.Column("name", sa.Text, primary_key=True),
        sa.Column("description", sa.Text),
        sa.Column("current_schema_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "entity_type_schema_versions",
        sa.Column("entity_type", sa.Text, sa.ForeignKey("entity_types.name", ondelete="CASCADE"), primary_key=True),
        sa.Column("version", sa.Integer, primary_key=True),
        sa.Column("json_schema", postgresql.JSONB, nullable=False),
        sa.Column("changelog", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "predicate_vocabulary",
        sa.Column("name", sa.Text, primary_key=True),
        sa.Column("description", sa.Text),
        sa.Column("domain_entity_types", postgresql.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("range_entity_types", postgresql.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "model_registry",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("component", sa.Text, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("version", sa.Text, nullable=False),
        sa.Column("config", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("component IN ('resolver','embedding_model','extraction_llm')", name="ck_model_registry_component"),
        sa.UniqueConstraint("component", "name", "version"),
    )

    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("type", sa.Text, sa.ForeignKey("entity_types.name"), nullable=False),
        sa.Column("context_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("contexts.id"), nullable=False),
        sa.Column("scope", sa.Text, nullable=False, server_default="context_local"),
        sa.Column("canonical_name", sa.Text, nullable=False),
        sa.Column("aliases", postgresql.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("attributes", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("attribute_schema_version", sa.Integer, nullable=False),
        sa.Column("resolver_version", sa.Text, nullable=False),
        sa.Column("embedding_model_version", sa.Text),
        sa.Column("source_adapter", sa.Text),
        sa.Column("source_ref", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("scope IN ('context_local','cross_context')", name="ck_entities_scope"),
        sa.ForeignKeyConstraint(
            ["type", "attribute_schema_version"],
            ["entity_type_schema_versions.entity_type", "entity_type_schema_versions.version"],
        ),
    )
    op.create_index("idx_entities_type_context", "entities", ["type", "context_id"])
    op.create_index("idx_entities_scope", "entities", ["scope"])
    op.execute("CREATE INDEX idx_entities_name_trgm ON entities USING GIN (canonical_name gin_trgm_ops)")
    op.execute("CREATE INDEX idx_entities_attributes ON entities USING GIN (attributes jsonb_path_ops)")

    op.create_table(
        "relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("predicate", sa.Text, sa.ForeignKey("predicate_vocabulary.name"), nullable=False),
        sa.Column("object_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE")),
        sa.Column("object_literal", postgresql.JSONB),
        sa.Column("context_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("contexts.id"), nullable=False),
        sa.Column("attributes", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("confidence", sa.Numeric),
        sa.Column("source_adapter", sa.Text),
        sa.Column("source_ref", sa.Text),
        sa.Column("extraction_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("resolver_version", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("object_id IS NOT NULL OR object_literal IS NOT NULL", name="ck_relationships_object"),
    )
    op.create_index("idx_rel_subject", "relationships", ["subject_id"])
    op.create_index("idx_rel_object", "relationships", ["object_id"])
    op.create_index("idx_rel_predicate_context", "relationships", ["predicate", "context_id"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("context_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("contexts.id"), nullable=False),
        sa.Column("source_adapter", sa.Text, nullable=False),
        sa.Column("source_ref", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer),
        sa.Column("embedding_model_version", sa.Text, nullable=False),
        sa.Column("qdrant_point_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_chunks_context", "chunks", ["context_id"])

    op.create_table(
        "ingestion_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("adapter", sa.Text, nullable=False),
        sa.Column("source_ref", sa.Text, nullable=False),
        sa.Column("context_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("contexts.id")),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("stats", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("error", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('running','succeeded','failed')", name="ck_ingestion_runs_status"),
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_table("chunks")
    op.drop_table("relationships")
    op.drop_table("entities")
    op.drop_table("model_registry")
    op.drop_table("predicate_vocabulary")
    op.drop_table("entity_type_schema_versions")
    op.drop_table("entity_types")
    op.drop_table("context_relationships")
    op.drop_table("contexts")
