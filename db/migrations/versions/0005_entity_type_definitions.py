"""entity type short/long semantic definitions

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-23

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

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
    sa.column("short_description", sa.Text),
    sa.column("long_description", sa.Text),
    sa.column("changelog", sa.Text),
)

# Carried forward unchanged from v1 (db/migrations/versions/0002_seed_entity_types.py)
# — this migration only adds semantic definitions, not attribute changes.
JSON_SCHEMAS = {
    "Person": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Person.attributes.v1",
        "type": "object",
        "properties": {
            "birth_date": {"type": "string"},
            "death_date": {"type": "string"},
            "occupation": {"type": "string"},
            "nationality": {"type": "string"},
            "description": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "Organization": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Organization.attributes.v1",
        "type": "object",
        "properties": {
            "founded_date": {"type": "string"},
            "dissolved_date": {"type": "string"},
            "industry": {"type": "string"},
            "headquarters": {"type": "string"},
            "description": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "Event": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Event.attributes.v1",
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "location": {"type": "string"},
            "description": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "Location": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Location.attributes.v1",
        "type": "object",
        "properties": {
            "location_type": {"type": "string", "description": "e.g. city, country, building, region"},
            "coordinates": {"type": "string"},
            "population": {"type": "integer"},
            "description": {"type": "string"},
        },
        "additionalProperties": True,
    },
}

# Root-caused via live bench runs (2026-09-22/23): entity discovery was
# defaulting to Organization as a catch-all whenever a type was unclear —
# tagging locations ("221B Baker Street", "Nevada Mountains") and even
# non-entities (a concert, a notebook) as Organization, which then failed
# every downstream relationship's predicate vocabulary check. The old
# prompt gave zero disambiguation guidance; these definitions are injected
# dynamically per-call (see loci/llm/client.py::discover_entities) so they
# stay in sync with whatever's live in the DB, including future custom
# types added without a code change.
DEFINITIONS = {
    "Person": {
        "short": "A human individual, real or fictional.",
        "long": (
            "An individual human being — named, described, or referred to as a specific person "
            "(including a not-yet-identified one tracked via identity_status=unresolved, e.g. "
            "'the cabman'). Never a group of people, an institution, or a non-human character "
            "unless it's personified as a single individual actor."
        ),
    },
    "Organization": {
        "short": "An institution, company, or organized group acting as a body — not a place.",
        "long": (
            "A formally or informally organized group of people acting as a single body: a "
            "company, government agency, police force, club, family unit, council, or similar "
            "institution. An Organization is the institution itself, not the building or address "
            "it operates from — a building strongly associated with an institution (a police "
            "station, a headquarters) is still a Location unless the passage is specifically "
            "referring to the body of people/authority acting, not the physical premises. When "
            "genuinely ambiguous, judge from how the entity is used in THIS passage, not from the "
            "name alone."
        ),
    },
    "Event": {
        "short": "A specific happening — something that occurs, not a place or thing.",
        "long": (
            "A specific occurrence with some temporal extent: a battle, a meeting, a concert, a "
            "disaster, a crime, a journey. If the passage describes something HAPPENING rather "
            "than a person, place, or institution, it's an Event — don't default to Organization "
            "for a happening just because it doesn't obviously fit Person or Location."
        ),
    },
    "Location": {
        "short": "A physical place — a building, address, region, or landmark.",
        "long": (
            "Any physical place: a building, room, address, city, region, country, or geographic "
            "feature. A building is a Location even when it's strongly associated with (or named "
            "after) an organization — e.g. an address where people live or work is a Location even "
            "though an institution operates there; the same name can refer to an Organization "
            "instead only when the passage is clearly talking about the institution/body acting, "
            "not the premises."
        ),
    },
}


def upgrade() -> None:
    op.add_column("entity_type_schema_versions", sa.Column("short_description", sa.Text))
    op.add_column("entity_type_schema_versions", sa.Column("long_description", sa.Text))

    for type_name, defs in DEFINITIONS.items():
        op.bulk_insert(
            entity_type_schema_versions_table,
            [
                {
                    "entity_type": type_name,
                    "version": 2,
                    "json_schema": JSON_SCHEMAS[type_name],
                    "short_description": defs["short"],
                    "long_description": defs["long"],
                    "changelog": "added short/long semantic definitions to disambiguate from "
                    "Organization, especially for buildings/addresses associated with "
                    "institutions and for non-place, non-institution happenings (Event)",
                }
            ],
        )
        op.execute(
            entity_types_table.update()
            .where(entity_types_table.c.name == type_name)
            .values(current_schema_version=2)
        )


def downgrade() -> None:
    op.execute(
        entity_types_table.update()
        .where(entity_types_table.c.name.in_(list(DEFINITIONS)))
        .values(current_schema_version=1)
    )
    op.execute(
        "DELETE FROM entity_type_schema_versions WHERE version = 2 AND entity_type IN "
        f"({','.join(repr(k) for k in DEFINITIONS)})"
    )
    op.drop_column("entity_type_schema_versions", "long_description")
    op.drop_column("entity_type_schema_versions", "short_description")
