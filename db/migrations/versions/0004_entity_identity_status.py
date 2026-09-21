"""entity identity_status + merged_into for deferred coreference resolution

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-19

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "entities",
        sa.Column("identity_status", sa.Text, nullable=False, server_default="named"),
    )
    op.add_column(
        "entities",
        sa.Column("merged_into", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id"), nullable=True),
    )
    op.create_check_constraint(
        "ck_entities_identity_status", "entities", "identity_status IN ('named','unresolved')"
    )
    op.create_index("idx_entities_identity_status", "entities", ["identity_status"])


def downgrade() -> None:
    op.drop_index("idx_entities_identity_status", table_name="entities")
    op.drop_constraint("ck_entities_identity_status", "entities", type_="check")
    op.drop_column("entities", "merged_into")
    op.drop_column("entities", "identity_status")
