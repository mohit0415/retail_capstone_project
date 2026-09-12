import json
import logging
import time
import types

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text

from configs.settings import settings
from src.index.models import active_embed_model_name, configure_llama_settings, get_embed_model

logger = logging.getLogger(__name__)

_index: VectorStoreIndex | None = None

_lexical_corpus: list[BaseNode] | None = None


def _vector_table_name() -> str:
    return f"data_{settings.vector_table_name}".lower()


def _patch_docstore_missing_nodes(index: VectorStoreIndex) -> None:
    store = index.docstore

    def get_nodes(self, node_ids, raise_error=True):  # noqa: ARG001
        found = []

        for node_id in node_ids:
            node = self.get_node(node_id, raise_error=False)

            if node is not None:
                found.append(node)

        return found

    async def aget_nodes(self, node_ids, raise_error=True):  # noqa: ARG001
        found = []

        for node_id in node_ids:
            node = await self.aget_node(node_id, raise_error=False)

            if node is not None:
                found.append(node)

        return found

    store.get_nodes = types.MethodType(get_nodes, store)
    store.aget_nodes = types.MethodType(aget_nodes, store)


# pgvector cannot build an HNSW index on a vector column wider than this
PGVECTOR_HNSW_MAX_DIMENSIONS = 2000


def hnsw_settings(dimensions: int) -> dict | None:
    """HNSW index settings, or None when the embedding is too wide for one.

    text-embedding-3-large has 3072 dimensions, so every store build tried CREATE INDEX, failed and
    logged "Error creating HNSW index" - which reads like a broken retrieval. Without the index
    pgvector runs an exact search, which is correct and quick for a corpus of a few hundred chunks.
    """
    if dimensions > PGVECTOR_HNSW_MAX_DIMENSIONS:
        return None

    return {
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    }


def build_vector_store() -> PGVectorStore:
    url = settings.database_url

    return PGVectorStore.from_params(
        connection_string=url,
        async_connection_string=url.replace("postgresql://", "postgresql+asyncpg://"),
        table_name=settings.vector_table_name,
        embed_dim=settings.embedding_dimensions,
        hnsw_kwargs=hnsw_settings(settings.embedding_dimensions),
    )


def table_has_rows() -> bool:
    try:
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            row = conn.execute(text(f'SELECT 1 FROM "{_vector_table_name()}" LIMIT 1')).fetchone()

        return row is not None
    except Exception:
        logger.debug("table_has_rows probe failed (table may not exist yet)", exc_info=True)

        return False


def load_or_create_index() -> VectorStoreIndex:
    global _index

    if _index is not None:
        return _index

    configure_llama_settings()

    vector_store = build_vector_store()
    embed_model = get_embed_model()

    if table_has_rows():
        _index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=embed_model)

    _patch_docstore_missing_nodes(_index)

    return _index


def reset_index() -> None:
    global _index, _lexical_corpus

    _index = None
    _lexical_corpus = None


def _row_to_node(node_id: str, body: str | None, metadata) -> BaseNode:
    """Rebuild the node llama-index stored, so the BM25 leg carries the same metadata as the vector leg."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {}

    metadata = dict(metadata or {})

    try:
        node = metadata_dict_to_node(metadata, text=body or "")
    except Exception:
        plain = {k: v for k, v in metadata.items() if not k.startswith("_")}
        node = TextNode(id_=node_id, text=body or "", metadata=plain)

    if not node.node_id:
        node.id_ = node_id

    return node


def load_lexical_corpus() -> list[BaseNode]:
    """All nodes in the vector table, as in-memory nodes for the BM25 retriever.

    ``VectorStoreIndex.from_vector_store`` leaves the in-memory docstore empty
    (PGVectorStore keeps the text in Postgres), so after a restart the BM25 leg
    had nothing to search and hybrid retrieval was silently vector-only. This
    reads ``node_id, text, metadata_`` straight from the table instead. The
    corpus is a few hundred clause-level nodes for this project, so holding it
    in memory is cheap; it is reloaded after any ingest, rebuild or clear.
    """
    global _lexical_corpus

    if _lexical_corpus is not None:
        return _lexical_corpus

    started = time.perf_counter()

    try:
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            rows = conn.execute(
                text(f'SELECT node_id, text, metadata_ FROM "{_vector_table_name()}"')
            ).fetchall()
    except Exception as exc:
        logger.warning("lexical corpus could not be read from %s: %s", _vector_table_name(), exc)

        return []

    nodes = [_row_to_node(row[0], row[1], row[2]) for row in rows]
    _lexical_corpus = nodes

    logger.info(
        "lexical corpus loaded nodes=%d table=%s elapsed_ms=%.0f",
        len(nodes),
        _vector_table_name(),
        (time.perf_counter() - started) * 1000,
    )

    return nodes


def clear_vector_table() -> int:
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        row = conn.execute(text(f'SELECT COUNT(*) FROM "{_vector_table_name()}"')).fetchone()
        removed = int(row[0]) if row else 0

        conn.execute(text(f'TRUNCATE TABLE "{_vector_table_name()}"'))
        conn.commit()

    reset_index()

    return removed


def insert_nodes(nodes: list[BaseNode]) -> int:
    if not nodes:
        return 0

    global _lexical_corpus

    index = load_or_create_index()
    index.insert_nodes(nodes)
    _lexical_corpus = None

    return len(nodes)


def stored_embed_models() -> dict:
    result = {"models": set(), "untagged": 0}

    try:
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT metadata_->>'embed_model' AS model, COUNT(*) "
                    f'FROM "{_vector_table_name()}" GROUP BY model'
                )
            ).fetchall()

        for model_name, count in rows:
            if model_name:
                result["models"].add(model_name)
            else:
                result["untagged"] = int(count)
    except Exception:
        logger.debug("stored_embed_models probe failed", exc_info=True)

    return result


def check_embed_model_compatibility() -> tuple[bool, str]:
    active = active_embed_model_name()
    stored = stored_embed_models()

    models = stored["models"]
    untagged = stored["untagged"]

    if not models and untagged == 0:
        return True, ""

    foreign = models - {active}

    if not foreign and untagged == 0:
        return True, ""

    parts = []

    if foreign:
        parts.append(f"chunks embedded with {sorted(foreign)}")

    if untagged:
        parts.append(f"{untagged} chunk(s) carrying no embed_model stamp")

    message = (
        f"Embedding model mismatch: the vector table holds {' and '.join(parts)}, "
        f"but the active embedding model is '{active}'. Vectors produced by different "
        f"embedding models are not comparable, so retrieval would return confident "
        f"nonsense rather than failing. Clear the vector table and re-ingest the corpus "
        f"with '{active}'."
    )

    return False, message


def indexed_document_summary() -> list[dict]:
    try:
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT metadata_->>'doc_type' AS doc_type, "
                    f"metadata_->>'document_title' AS document_title, "
                    f"metadata_->>'version' AS version, "
                    f"metadata_->>'parsed_with' AS parsed_with, "
                    f"COUNT(*) AS node_count "
                    f'FROM "{_vector_table_name()}" '
                    f"GROUP BY 1, 2, 3, 4 ORDER BY 2"
                )
            ).fetchall()

        return [
            {
                "doc_type": row[0],
                "document_title": row[1],
                "version": row[2],
                "parsed_with": row[3],
                "node_count": int(row[4]),
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning("could not summarise the indexed corpus: %s", exc)
        return []


def clause_number_exists(clause_number: str) -> bool:
    try:
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    f'SELECT 1 FROM "{_vector_table_name()}" '
                    f"WHERE metadata_->>'clause_number' = :clause LIMIT 1"
                ),
                {"clause": clause_number},
            ).fetchone()

        return row is not None
    except Exception:
        logger.debug("clause_number_exists probe failed for %s", clause_number, exc_info=True)

        return False


def file_hash_exists(file_hash: str) -> bool:
    try:
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT 1 FROM \"{_vector_table_name()}\" WHERE metadata_->>'file_hash' = :value LIMIT 1"
                ),
                {"value": file_hash},
            ).fetchone()

        return row is not None
    except Exception:
        logger.debug("file_hash_exists probe failed", exc_info=True)

        return False


def supersede_previous_versions(doc_type: str, version: str) -> int:
    try:
        engine = create_engine(settings.database_url)

        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f'UPDATE "{_vector_table_name()}" '
                    f"SET metadata_ = jsonb_set(metadata_::jsonb, '{{is_current}}', 'false')::json "
                    f"WHERE metadata_->>'doc_type' = :doc_type "
                    f"AND metadata_->>'version' <> :version"
                ),
                {"doc_type": doc_type, "version": version},
            )

        return result.rowcount or 0
    except Exception as exc:
        logger.error(
            "supersession update failed for %s: %s. Superseded versions of this document are "
            "still marked is_current=true and remain retrievable alongside version %s.",
            doc_type,
            exc,
            version,
        )
        return 0


def table_embedding_dimensions() -> int | None:
    """Width of the ``embedding`` column of the live vector table, or None.

    pgvector stores the declared dimension in ``atttypmod``. Returns None when
    the table does not exist yet (nothing has been ingested), which is not a
    mismatch - the table is then created at the currently configured width.
    """
    try:
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT a.atttypmod FROM pg_attribute a "
                    "JOIN pg_class c ON c.oid = a.attrelid "
                    "WHERE c.relname = :table AND a.attname = 'embedding' AND a.attnum > 0"
                ),
                {"table": _vector_table_name()},
            ).fetchone()

        if row is None or row[0] is None or row[0] <= 0:
            return None

        return int(row[0])
    except Exception:
        logger.debug("vector table dimension probe failed", exc_info=True)

        return None
