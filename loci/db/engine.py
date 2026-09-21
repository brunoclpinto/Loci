from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from loci.config import LociSettings

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine(settings: LociSettings) -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database.dsn, pool_pre_ping=True)
    return _engine


def get_session_factory(settings: LociSettings) -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(settings), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope(settings: LociSettings) -> Iterator[Session]:
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
