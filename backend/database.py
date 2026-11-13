"""
SQLAlchemy 2.0 конфигурация: engine, session factory и транзакционни контексти.

Гаранции за ACID:
* `transactional_session()` отваря ЕДНА транзакция (`session.begin()`):
  при успешен изход -> COMMIT, при всяко изключение -> ROLLBACK.
  Така `SimulationRun` и всички негови `AuditLog` записи се записват атомарно.
* `get_db()` е FastAPI dependency за read-only/кратки заявки.
* Всички времена в базата са UTC (сесийна настройка `timezone=UTC`).
* Драйвер: psycopg2 (`postgresql+psycopg2`), както е в общия requirements.txt.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

from sqlalchemy import MetaData, create_engine, event, text
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_settings

logger = logging.getLogger(__name__)

IsolationLevel = Literal["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"]

# Детерминирани имена на ограниченията -> стабилни Alembic миграции
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Базов клас за всички ORM модели."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ==============================================================================
# Engine / Session factory
# ==============================================================================


def create_db_engine(url: str | URL | None = None, *, echo: bool | None = None) -> Engine:
    """Създава Engine. За PostgreSQL – connection pool, pre-ping и UTC сесия."""
    settings = get_settings()
    db_url = make_url(url) if url is not None else settings.sqlalchemy_url
    echo = settings.db_echo if echo is None else echo

    if db_url.get_backend_name() == "postgresql":
        connect_args: dict[str, object] = {"options": "-c timezone=UTC"}
        if settings.db_statement_timeout_ms > 0:
            connect_args["options"] += f" -c statement_timeout={settings.db_statement_timeout_ms}"
        engine = create_engine(
            db_url,
            echo=echo,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
            isolation_level="READ COMMITTED",
            connect_args=connect_args,
        )
    elif db_url.get_backend_name() == "sqlite":
        # Използва се само в unit тестове
        engine = create_engine(
            db_url, echo=echo, connect_args={"check_same_thread": False}
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - тривиално
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    else:
        engine = create_engine(db_url, echo=echo, pool_pre_ping=True)

    logger.info("DB engine създаден за %s", db_url.render_as_string(hide_password=True))
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,  # обектите остават достъпни след COMMIT (за API отговори)
        class_=Session,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Lazy singleton – engine се създава при първа нужда, не при импорт."""
    return create_db_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return create_session_factory(get_engine())


# ==============================================================================
# Транзакционни контексти
# ==============================================================================


@contextmanager
def transactional_session(
    session_factory: sessionmaker[Session] | None = None,
    *,
    isolation_level: IsolationLevel | None = None,
) -> Iterator[Session]:
    """
    Unit of Work: една атомарна транзакция.

        with transactional_session() as s:
            s.add(run)                 # SimulationRun
            s.add_all(audit_logs)      # всички AuditLog записи
        # -> COMMIT; при изключение -> ROLLBACK и нищо не се записва

    Не извиквай `session.commit()` вътре в блока – контекстът го прави.
    """
    factory = session_factory or get_session_factory()
    session = factory()
    try:
        if isolation_level is not None:
            session.connection(execution_options={"isolation_level": isolation_level})
        with session.begin():
            yield session
    except Exception:
        logger.exception("Транзакцията е върната (ROLLBACK)")
        raise
    finally:
        session.close()


@contextmanager
def nested_savepoint(session: Session) -> Iterator[Session]:
    """SAVEPOINT в рамките на текуща транзакция (частичен rollback)."""
    with session.begin_nested():
        yield session


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency. Всяка заявка получава собствена сесия;
    при грешка – ROLLBACK, накрая сесията винаги се затваря.
    """
    session = get_session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ==============================================================================
# Инициализация / здраве
# ==============================================================================


def init_db(engine: Engine | None = None) -> None:
    """Създава всички таблици (за dev/тестове; в продукция – Alembic)."""
    from backend import models  # noqa: F401  – регистрира моделите в metadata

    Base.metadata.create_all(bind=engine or get_engine())


def drop_db(engine: Engine | None = None) -> None:
    from backend import models  # noqa: F401

    Base.metadata.drop_all(bind=engine or get_engine())


def check_connection(engine: Engine | None = None) -> bool:
    try:
        with (engine or get_engine()).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Няма връзка с базата данни")
        return False


__all__ = [
    "Base",
    "NAMING_CONVENTION",
    "check_connection",
    "create_db_engine",
    "create_session_factory",
    "drop_db",
    "get_db",
    "get_engine",
    "get_session_factory",
    "init_db",
    "nested_savepoint",
    "transactional_session",
]
