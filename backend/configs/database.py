from contextlib import contextmanager

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from configs.settings import settings

_pool: ConnectionPool | None = None


def open_pool() -> ConnectionPool:
    global _pool

    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            timeout=settings.db_connect_timeout_seconds,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )

    return _pool


def close_pool() -> None:
    global _pool

    if _pool is not None:
        _pool.close()
        _pool = None


def get_pool() -> ConnectionPool:
    if _pool is None:
        return open_pool()

    return _pool


@contextmanager
def read_only_connection():
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET statement_timeout = {settings.sql_statement_timeout_ms}")

        yield conn


@contextmanager
def writable_connection() -> Connection:
    with get_pool().connection() as conn:
        yield conn
