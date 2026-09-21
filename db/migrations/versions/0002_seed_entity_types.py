"""seed entity types

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-19

"""
import json
from pathlib import Path

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

SEED_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "loci" / "schema_registry" / "seed_schemas"

ENTITY_TYPES = ["Person", "Organization", "Event", "Location"]

entity_types_table = sa.table(
    "entity_types",
    sa.column("name", sa.Text),
    sa.column("current_schema_version", sa.Integer),
)

entity_type_schema_versions_table = sa.table(
    "entity_type_schema_versions",
    sa.column("entity_type", sa.Text),
    sa.column("version", sa.Integer),
    sa.column("json_schema", postgresql.JSONB),
    sa.column("changelog", sa.Text),
)

contexts_table = sa.table(
    "contexts",
    sa.column("name", sa.Text),
    sa.column("description", sa.Text),
)


def upgrade() -> None:
    for type_name in ENTITY_TYPES:
        schema_path = SEED_SCHEMAS_DIR / f"{type_name.lower()}.v1.json"
        json_schema = json.loads(schema_path.read_text())

        op.bulk_insert(entity_types_table, [{"name": type_name, "current_schema_version": 1}])
        op.bulk_insert(
            entity_type_schema_versions_table,
            [
                {
                    "entity_type": type_name,
                    "version": 1,
                    "json_schema": json_schema,
                    "changelog": "initial version",
                }
            ],
        )

    op.bulk_insert(
        contexts_table,
        [{"name": "real_world", "description": "The default context for facts asserted as true in reality, not scoped to a fictional or source-specific universe."}],
    )


def downgrade() -> None:
    op.execute("DELETE FROM contexts WHERE name = 'real_world'")
    op.execute("DELETE FROM entity_type_schema_versions WHERE entity_type IN ('Person','Organization','Event','Location')")
    op.execute("DELETE FROM entity_types WHERE name IN ('Person','Organization','Event','Location')")
