"""Tests for loci/resolve.py: entity resolution steps 1-5."""
import pytest
from loci.resolve import normalize_mention, resolve_entity


class TestNormalize:
    def test_lowercase(self):
        assert normalize_mention("Sherlock Holmes") == "sherlock holmes"

    def test_strips_mr(self):
        assert normalize_mention("Mr. Sherlock Holmes") == "sherlock holmes"

    def test_strips_mrs(self):
        assert normalize_mention("Mrs. Hudson") == "hudson"

    def test_strips_dr(self):
        assert normalize_mention("Dr. Watson") == "watson"

    def test_strips_punctuation(self):
        assert normalize_mention("Holmes,") == "holmes"

    def test_empty_after_strip(self):
        assert normalize_mention("Mr.") == ""

    def test_multiword(self):
        assert normalize_mention("Mr. Sherlock Holmes") == "sherlock holmes"


class TestExactAlias:
    def test_exact_hit(self, tmp_db):
        from loci.store import insert_alias, insert_entity
        eid = insert_entity(tmp_db, canonical_name="Sherlock Holmes")
        insert_alias(tmp_db, entity_id=eid, alias="sherlock holmes")

        result = resolve_entity(tmp_db, "Sherlock Holmes")
        assert result == eid

    def test_exact_hit_with_title_stripped(self, tmp_db):
        from loci.store import insert_alias, insert_entity
        eid = insert_entity(tmp_db, canonical_name="Sherlock Holmes")
        insert_alias(tmp_db, entity_id=eid, alias="sherlock holmes")

        # "Mr. Sherlock Holmes" normalizes to "sherlock holmes" → exact hit
        result = resolve_entity(tmp_db, "Mr. Sherlock Holmes")
        assert result == eid


class TestFuzzyMatch:
    def test_subset_match_holmes_in_sherlock_holmes(self, tmp_db):
        """'holmes' ⊂ 'sherlock holmes' → same entity."""
        from loci.store import insert_alias, insert_entity
        eid = insert_entity(tmp_db, canonical_name="Sherlock Holmes")
        insert_alias(tmp_db, entity_id=eid, alias="sherlock holmes")

        result = resolve_entity(tmp_db, "Holmes")
        assert result == eid

    def test_fuzzy_adds_new_alias(self, tmp_db):
        from loci.store import insert_alias, insert_entity
        eid = insert_entity(tmp_db, canonical_name="Sherlock Holmes")
        insert_alias(tmp_db, entity_id=eid, alias="sherlock holmes")

        resolve_entity(tmp_db, "Holmes")

        aliases = {r[0] for r in tmp_db.execute(
            "SELECT alias FROM aliases WHERE entity_id=?", [eid]
        )}
        assert "holmes" in aliases

    def test_ambiguous_records_pending_link(self, tmp_db):
        from loci.store import insert_alias, insert_entity
        # Two entities both containing "holmes" → ambiguous fuzzy match
        e1 = insert_entity(tmp_db, canonical_name="Sherlock Holmes")
        insert_alias(tmp_db, entity_id=e1, alias="sherlock holmes")
        e2 = insert_entity(tmp_db, canonical_name="Mycroft Holmes")
        insert_alias(tmp_db, entity_id=e2, alias="mycroft holmes")

        resolve_entity(tmp_db, "Holmes")

        pending = tmp_db.execute(
            "SELECT mention FROM pending_links WHERE mention='holmes'"
        ).fetchone()
        assert pending is not None


class TestCreateNewEntity:
    def test_new_entity_created(self, tmp_db):
        eid = resolve_entity(tmp_db, "Irene Adler")
        assert eid is not None
        row = tmp_db.execute(
            "SELECT canonical_name FROM entities WHERE id=?", [eid]
        ).fetchone()
        assert row is not None
        assert "Irene" in row["canonical_name"]

    def test_new_entity_alias_stored(self, tmp_db):
        eid = resolve_entity(tmp_db, "Irene Adler")
        aliases = {r[0] for r in tmp_db.execute(
            "SELECT alias FROM aliases WHERE entity_id=?", [eid]
        )}
        assert "irene adler" in aliases


class TestAcceptanceAliases:
    def test_three_mentions_one_entity(self, tmp_db):
        """Holmes, Sherlock Holmes, Mr. Sherlock Holmes → ONE entity, ≥3 aliases."""
        id1 = resolve_entity(tmp_db, "Sherlock Holmes")
        id2 = resolve_entity(tmp_db, "Holmes")
        id3 = resolve_entity(tmp_db, "Mr. Sherlock Holmes")

        assert id1 == id2 == id3, "all three mentions must resolve to the same entity"

        aliases = {r[0] for r in tmp_db.execute(
            "SELECT alias FROM aliases WHERE entity_id=?", [id1]
        )}
        assert len(aliases) >= 3, f"expected ≥3 aliases, got {aliases}"
        assert "sherlock holmes" in aliases
        assert "holmes" in aliases
