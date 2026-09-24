"""predicate short/long definitions + evidence-driven domain/range widening

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-23

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

ALL_TYPES = ["Person", "Organization", "Event", "Location"]

# Computed from a full-corpus bench (9 books, 2 extraction models,
# ~1128+ extraction calls) — domain_entity_types/range_entity_types widened
# to the exact union of types the model actually attempted-and-got-rejected
# for each predicate/slot (not a blanket "accept everything"). 8 of 9
# predicates converge to all 4 types on both sides simply because, at this
# scale, every slot got attempted with every type at least once somewhere
# in the corpus — born_in is the one exception, with far fewer and
# narrower observed failures. Descriptions now carry the correctness
# burden the type constraint used to for those 8 predicates.
PREDICATES = {
    "located_in": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject is physically located within the object place.",
        "long": (
            "The subject exists at, is contained within, or is situated at the object, which "
            "must be understood as a physical PLACE in this usage — a building, room, city, "
            "region, or similar. Never use a Person as the object: if you mean 'X is with/"
            "serving/staying with person Y', use a different predicate (works_for for service, "
            "member_of for household/family, or related_to as a fallback) — located_in is "
            "exclusively about physical containment in a place, even though the schema no "
            "longer blocks other object types outright."
        ),
    },
    "member_of": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject is a member of the object group (an organization, institution, or family).",
        "long": (
            "Subject belongs to the object as a group — a company, police force, club, or "
            "family unit (the Organization entity type explicitly includes family/household "
            "groups). Direction: the individual/smaller entity is always the subject, the group "
            "is always the object — never reverse this. Not for physical containment (use "
            "located_in) and not for a one-off event (a witness at a crime isn't a 'member of' "
            "the crime — use participant_in or related_to)."
        ),
    },
    "founded": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject founded/established the object organization.",
        "long": (
            "Subject (a person or an existing organization) created or established the object, "
            "which should be an Organization — a company, institution, or similar. Not for "
            "physical construction of a building (omit or use related_to instead) and not for "
            "an event 'founding' something."
        ),
    },
    "works_for": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject is employed by or in service to the object — an institution or a specific person.",
        "long": (
            "Covers both institutional employment ('the constable works_for Scotland Yard') and "
            "personal service to a specific individual ('the maid works_for Mrs. Charpentier'). "
            "Direction: the employee/servant is always the subject, the employer (whether an "
            "organization or a person) is always the object — never reverse this. Not for "
            "describing where someone lives (use located_in) or their involvement in an event "
            "(use participant_in)."
        ),
    },
    "born_in": {
        "domain": ["Person", "Event"],
        "range": ["Location", "Person", "Organization"],
        "short": "Subject (a person) was born in the object location.",
        "long": (
            "Strictly for a person's birthplace — subject is a Person, object is a Location "
            "(a city, country, or similar). Do not use for an event's origin or an "
            "organization's founding location (use related_to or a literal attribute instead)."
        ),
    },
    "died_in": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject (a person) died in the object location.",
        "long": (
            "Strictly for where a person died — subject is a Person, object is a Location. Do "
            "not use the circumstances or event of death as the object (not 'died_in [the "
            "murder]') — that's better captured as an attribute or via related_to; died_in's "
            "object must be a place, not an event or another person."
        ),
    },
    "occurred_at": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject event occurred at the object location.",
        "long": (
            "Subject is an Event, object is the Location where it took place. Do not use a "
            "Person or Organization as the object — if you want to say who was involved, use "
            "participant_in instead (with the event as its object, not this predicate's)."
        ),
    },
    "participant_in": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject participated in, witnessed, or was otherwise involved in the object event.",
        "long": (
            "Broadly covers anyone connected to an event, not just active doers — a witness, "
            "victim, suspect, or investigator all count as a participant. Direction is fixed: "
            "the person/organization is always the subject, the EVENT is always the object — "
            "never the reverse (an event cannot 'participate in' something else; if one event "
            "is nested inside a larger one, use part_of instead)."
        ),
    },
    "part_of": {
        "domain": ALL_TYPES,
        "range": ALL_TYPES,
        "short": "Subject is a component, sub-division, or sub-incident of the object.",
        "long": (
            "Covers an organization being part of a larger one, a location being part of a "
            "larger region, and a smaller incident/event being part of a larger case or "
            "sequence of events (e.g. 'the theft part_of the wider investigation'). Direction: "
            "the smaller/more specific thing is always the subject, the larger/containing thing "
            "is always the object."
        ),
    },
    "related_to": {
        "domain": [],
        "range": [],
        "short": "Generic, otherwise-unclassified relationship between subject and object.",
        "long": (
            "Use only when no other predicate fits, after checking each one's definition — this "
            "is the fallback, not a default. Any entity type may appear as subject or object."
        ),
    },
}


predicate_vocabulary_table = sa.table(
    "predicate_vocabulary",
    sa.column("name", sa.Text),
    sa.column("domain_entity_types", postgresql.ARRAY(sa.Text)),
    sa.column("range_entity_types", postgresql.ARRAY(sa.Text)),
    sa.column("short_description", sa.Text),
    sa.column("long_description", sa.Text),
)


def upgrade() -> None:
    op.drop_column("predicate_vocabulary", "description")
    op.add_column("predicate_vocabulary", sa.Column("short_description", sa.Text))
    op.add_column("predicate_vocabulary", sa.Column("long_description", sa.Text))

    for name, defs in PREDICATES.items():
        op.execute(
            predicate_vocabulary_table.update()
            .where(predicate_vocabulary_table.c.name == name)
            .values(
                domain_entity_types=defs["domain"],
                range_entity_types=defs["range"],
                short_description=defs["short"],
                long_description=defs["long"],
            )
        )


def downgrade() -> None:
    op.drop_column("predicate_vocabulary", "long_description")
    op.drop_column("predicate_vocabulary", "short_description")
    op.add_column("predicate_vocabulary", sa.Column("description", sa.Text))

    # Restore original domain/range from db/migrations/versions/0003_seed_predicate_vocabulary.py
    originals = {
        "located_in": (["Person", "Organization", "Event", "Location"], ["Location"]),
        "member_of": (["Person"], ["Organization"]),
        "founded": (["Person", "Organization"], ["Organization"]),
        "works_for": (["Person"], ["Organization"]),
        "born_in": (["Person"], ["Location"]),
        "died_in": (["Person"], ["Location"]),
        "occurred_at": (["Event"], ["Location"]),
        "participant_in": (["Person", "Organization"], ["Event"]),
        "part_of": (["Organization", "Location"], ["Organization", "Location"]),
        "related_to": ([], []),
    }
    for name, (domain, rng) in originals.items():
        op.execute(
            predicate_vocabulary_table.update()
            .where(predicate_vocabulary_table.c.name == name)
            .values(domain_entity_types=domain, range_entity_types=rng)
        )
