import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.database import close_pool, open_pool, read_only_connection, writable_connection

SQL_DIR = Path(__file__).resolve().parents[1] / "data" / "sql"

DOMAIN_TABLES = ["vendors", "audit_logs", "retention_records", "compliance_reviews"]

COUNT = "SELECT COUNT(*) AS n FROM {table}"


def apply_schema() -> None:
    statement = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")

    with writable_connection() as conn:
        conn.execute(statement)

    print("applied schema.sql")


def report_row_counts() -> None:
    print("\ndomain table row counts:")

    empty = []

    with read_only_connection() as conn:
        for table in DOMAIN_TABLES:
            try:
                row = conn.execute(COUNT.format(table=table)).fetchone()
                count = int(row["n"])
            except Exception as exc:
                print(f"  {table}: unreadable ({exc})")
                continue

            print(f"  {table}: {count}")

            if count == 0:
                empty.append(table)

    if empty:
        print(
            "\nEmpty: "
            + ", ".join(empty)
            + "\nRun your dataset generator to populate them, then re-run this script to confirm."
        )


def main() -> None:
    open_pool()

    try:
        apply_schema()
        report_row_counts()
    finally:
        close_pool()


if __name__ == "__main__":
    main()
