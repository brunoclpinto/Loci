"""shared enum_vocabulary table for LLM-facing enum fields

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-23

"""
from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

# Descriptions adapted from the prose that already existed, hardcoded, in
# loci/llm/prompts/discover_entities.md (scope, identity_status) and
# loci/llm/client.py's _SEGMENT_CLASSIFICATION_SYSTEM_PROMPT (segment_type)
# — moved into the DB so they're injected dynamically per-call, same
# rationale and pattern as entity types (0005) and predicates (0006).
ROWS = [
    {
        "domain": "entity_scope",
        "value": "context_local",
        "short_description": "Specific to this document's own narrative or framing (the default).",
        "long_description": (
            "Anything specific to this passage's own narrative or framing — a fictional "
            "character, an organization invented for a story, a claim that only holds within "
            "this particular source. When in doubt, prefer context_local: it is always safe, "
            "while marking something cross_context incorrectly can wrongly merge unrelated "
            "entities across different documents."
        ),
    },
    {
        "domain": "entity_scope",
        "value": "cross_context",
        "short_description": "Exists independently of this document — a real place, organization, or historical person.",
        "long_description": (
            "Only for entities that exist independently of this passage's specific setting or "
            "narrative — real, objectively-real places, real organizations, real historical "
            "people. Safe to merge with the same entity mentioned in a different document. Use "
            "sparingly — see context_local's note on the cost of getting this wrong."
        ),
    },
    {
        "domain": "entity_identity_status",
        "value": "named",
        "short_description": "The entity has an actual stated name (the default).",
        "long_description": (
            "Leave as named once an entity has an actual name, even if it was only just "
            "resolved from an earlier unresolved reference."
        ),
    },
    {
        "domain": "entity_identity_status",
        "value": "unresolved",
        "short_description": "A distinct, real individual whose identity isn't stated here yet.",
        "long_description": (
            "Set when the passage clearly refers to a distinct, real individual whose identity "
            "isn't stated here and isn't on the known-entities list (e.g. 'the mysterious man', "
            "'the cabman') — give it a descriptive canonical_name (not a fabricated proper name) "
            "so it can be connected to their real identity later if the document reveals it. "
            "Don't create an entity at all for a one-off background mention that's never referred "
            "to again and has no relationships worth recording — this is for individuals worth "
            "tracking, not every pronoun in the text."
        ),
    },
    {
        "domain": "segment_type",
        "value": "narrative",
        "short_description": "The actual content/substance of the document.",
        "long_description": (
            "The story, article, report, or other real content — the substance of the document, "
            "not information about the document as an artifact."
        ),
    },
    {
        "domain": "segment_type",
        "value": "front_matter",
        "short_description": "Document metadata that precedes the real content.",
        "long_description": (
            "Information about the document/artifact itself that precedes its real content — a "
            "title page, table of contents, author byline, publisher/edition info."
        ),
    },
    {
        "domain": "segment_type",
        "value": "back_matter",
        "short_description": "Document metadata that follows the real content.",
        "long_description": (
            "Information about the document/artifact itself that follows its real content — a "
            "colophon, license text, appendix, index."
        ),
    },
]

enum_vocabulary_table = sa.table(
    "enum_vocabulary",
    sa.column("domain", sa.Text),
    sa.column("value", sa.Text),
    sa.column("short_description", sa.Text),
    sa.column("long_description", sa.Text),
)


def upgrade() -> None:
    op.create_table(
        "enum_vocabulary",
        sa.Column("domain", sa.Text, nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("short_description", sa.Text),
        sa.Column("long_description", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("domain", "value"),
    )
    op.bulk_insert(enum_vocabulary_table, ROWS)


def downgrade() -> None:
    op.drop_table("enum_vocabulary")
