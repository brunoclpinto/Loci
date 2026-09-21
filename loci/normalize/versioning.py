from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.db.models import ModelRegistryEntry

# Bumping any of these is a deliberate, explicit act — it changes what
# "the same entity" or "the same vector space" means, so old data doesn't
# silently become incomparable to new data. See Entity.resolver_version /
# Chunk.embedding_model_version, which stamp every row with the version
# active at the time it was written.
RESOLVER_VERSION = "trigram-v0"


def ensure_registered(session: Session, component: str, name: str, version: str, config: dict | None = None) -> None:
    """Idempotently record that (component, name, version) is the active
    implementation, so drift (e.g. swapping embedding models later) is
    detectable rather than silent."""
    existing = session.scalars(
        select(ModelRegistryEntry).where(
            ModelRegistryEntry.component == component,
            ModelRegistryEntry.name == name,
            ModelRegistryEntry.version == version,
        )
    ).first()
    if existing is not None:
        return
    session.add(
        ModelRegistryEntry(component=component, name=name, version=version, config=config or {}, is_active=True)
    )
