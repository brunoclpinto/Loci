"""seed predicate vocabulary

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-19

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

predicate_vocabulary_table = sa.table(
    "predicate_vocabulary",
    sa.column("name", sa.Text),
    sa.column("description", sa.Text),
    sa.column("domain_entity_types", postgresql.ARRAY(sa.Text)),
    sa.column("range_entity_types", postgresql.ARRAY(sa.Text)),
)

PREDICATES = [
    {
        "name": "located_in",
        "description": "Subject is physically located within the object.",
        "domain_entity_types": ["Person", "Organization", "Event", "Location"],
        "range_entity_types": ["Location"],
    },
    {
        "name": "member_of",
        "description": "Subject is a member of the object organization.",
        "domain_entity_types": ["Person"],
        "range_entity_types": ["Organization"],
    },
    {
        "name": "founded",
        "description": "Subject founded the object organization.",
        "domain_entity_types": ["Person", "Organization"],
        "range_entity_types": ["Organization"],
    },
    {
        "name": "works_for",
        "description": "Subject is employed by the object organization.",
        "domain_entity_types": ["Person"],
        "range_entity_types": ["Organization"],
    },
    {
        "name": "born_in",
        "description": "Subject was born in the object location.",
        "domain_entity_types": ["Person"],
        "range_entity_types": ["Location"],
    },
    {
        "name": "died_in",
        "description": "Subject died in the object location.",
        "domain_entity_types": ["Person"],
        "range_entity_types": ["Location"],
    },
    {
        "name": "occurred_at",
        "description": "Subject event occurred at the object location.",
        "domain_entity_types": ["Event"],
        "range_entity_types": ["Location"],
    },
    {
        "name": "participant_in",
        "description": "Subject participated in the object event.",
        "domain_entity_types": ["Person", "Organization"],
        "range_entity_types": ["Event"],
    },
    {
        "name": "part_of",
        "description": "Subject is a part/subdivision of the object.",
        "domain_entity_types": ["Organization", "Location"],
        "range_entity_types": ["Organization", "Location"],
    },
    {
        "name": "related_to",
        "description": "Generic, otherwise-unclassified relationship between subject and object.",
        "domain_entity_types": [],
        "range_entity_types": [],
    },
]


def upgrade() -> None:
    op.bulk_insert(predicate_vocabulary_table, PREDICATES)


def downgrade() -> None:
    names = tuple(p["name"] for p in PREDICATES)
    op.execute(f"DELETE FROM predicate_vocabulary WHERE name IN {names}")
