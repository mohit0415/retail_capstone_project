import logging
import time
from contextlib import contextmanager

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from configs.settings import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _redacted_dsn(dsn: str) -> str:
    """The connection string without its password, for log lines."""
    if "@" not in dsn or "://" not in dsn:
        return dsn

    scheme, rest = dsn.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    user = credentials.split(":", 1)[0]

    return f"{scheme}://{user}:***@{host}"


def open_pool() -> ConnectionPool:
    global _pool

    if _pool is None:
        started = time.perf_counter()

        try:
            _pool = ConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                timeout=settings.db_connect_timeout_seconds,
                kwargs={"row_factory": dict_row, "autocommit": True},
                open=True,
            )
        except Exception as exc:
            logger.error(
                "database pool failed to open dsn=%s error=%s", _redacted_dsn(settings.database_url), exc
            )

            raise

        logger.info(
            "database pool opened dsn=%s min=%d max=%d connect_timeout_s=%.1f statement_timeout_ms=%d in %.0fms",
            _redacted_dsn(settings.database_url),
            settings.db_pool_min_size,
            settings.db_pool_max_size,
            settings.db_connect_timeout_seconds,
            settings.sql_statement_timeout_ms,
            (time.perf_counter() - started) * 1000,
        )

    return _pool


def close_pool() -> None:
    global _pool

    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("database pool closed")


def get_pool() -> ConnectionPool:
    if _pool is None:
        logger.debug("database pool requested before startup; opening lazily")

        return open_pool()

    return _pool


@contextmanager
def read_only_connection():
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {settings.sql_statement_timeout_ms}")

        # the pool opens connections in autocommit mode, where "SET TRANSACTION READ ONLY" on its
        # own only covers that one SET statement and the next query could still write. Inside an
        # explicit transaction block the database itself refuses any write the block attempts.
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")

            yield conn


@contextmanager
def writable_connection() -> Connection:
    with get_pool().connection() as conn:
        yield conn
