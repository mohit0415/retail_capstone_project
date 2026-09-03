import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.database import close_pool, open_pool
from src.index.vector_index import check_embed_model_compatibility
from src.ingestion.bootstrap import corpus_dir
from src.ingestion.pipeline import ingest_directory, ingest_file

POLICY_DIR = corpus_dir()


def print_result(result) -> None:
    if result.skipped:
        print(f"  {result.file_name}: skipped - {result.reason}")
        return

    print(
        f"  {result.file_name}: {result.total_nodes} nodes "
        f"(text {result.text_nodes}, tables {result.table_nodes}, images {result.image_nodes}) "
        f"via {result.parsed_with}, doc_type={result.doc_type} v{result.version}"
        + (f", superseded {result.superseded} older node(s)" if result.superseded else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and index the policy corpus")
    parser.add_argument("--path", default=str(POLICY_DIR), help="file or directory to ingest")
    parser.add_argument("--force", action="store_true", help="re-ingest even when the file hash is known")
    parser.add_argument("--check", action="store_true", help="only report embedding model compatibility")
    args = parser.parse_args()

    open_pool()

    try:
        compatible, message = check_embed_model_compatibility()

        if not compatible:
            print(message)
            sys.exit(1)

        if args.check:
            print("embedding model is consistent with what is already indexed")
            return

        target = Path(args.path)

        if target.is_file():
            print(f"ingesting {target.name}")
            print_result(ingest_file(str(target), target.name, force=args.force))
            return

        if not target.exists():
            print(f"no such path: {target}")
            sys.exit(1)

        report = ingest_directory(str(target), force=args.force)

        if not report.results:
            print(f"no documents found in {target}. See data/policies/README.md for the expected files.")
            return

        print(f"ingesting from {target}")

        for result in report.results:
            print_result(result)

        print(f"done, {report.total_nodes} nodes indexed across {len(report.results)} file(s)")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
