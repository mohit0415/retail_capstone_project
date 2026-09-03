from src.ingestion.policy_metadata import infer_doc_type, parse_frontmatter
from src.ingestion.splitters import build_recursive_splitter, split_by_clause
from src.ingestion.table_nodes import extract_markdown_tables

SAMPLE = """---
doc_type: retention_policy
title: Records Retention Policy
version: 3.2
effective_date: 2025-04-01
---

## 4.1 Transaction records

Point of sale transaction records are retained for seven financial years.

## 4.2 Customer profiles

Loyalty profiles are retained for twenty-four months after the last interaction.

### A.8.15 Logging

Access logs are retained for two years.
"""


def test_frontmatter_is_parsed_and_stripped():
    meta, body = parse_frontmatter(SAMPLE)

    assert meta["doc_type"] == "retention_policy"
    assert meta["version"] == "3.2"
    assert body.lstrip().startswith("## 4.1")


def test_clause_numbers_survive_splitting():
    _, body = parse_frontmatter(SAMPLE)

    sections = split_by_clause(body)
    numbers = [clause for _, clause, _ in sections]

    assert numbers == ["4.1", "4.2", "A.8.15"]


def test_clause_headings_are_captured():
    _, body = parse_frontmatter(SAMPLE)

    headings = [heading for heading, _, _ in split_by_clause(body)]

    assert "Transaction records" in headings


def test_unnumbered_document_still_produces_one_section():
    sections = split_by_clause("Just a paragraph with no headings at all.")

    assert len(sections) == 1
    assert sections[0][1] == "1"


def test_doc_type_is_inferred_from_the_filename():
    assert infer_doc_type("vendor_policy.md") == "vendor_policy"
    assert infer_doc_type("gdpr_articles.md") == "gdpr"
    assert infer_doc_type("iso_27001.md") == "iso_27001"
    assert infer_doc_type("anti_bribery_policy.md") == "anti_bribery_policy"


def test_recursive_splitter_respects_the_chunk_ceiling():
    from configs.settings import settings

    splitter = build_recursive_splitter()
    text = "This is a sentence about retention. " * 400

    pieces = splitter.split_text(text)

    assert len(pieces) > 1
    assert all(len(piece) <= settings.max_chunk_chars + settings.chunk_overlap_chars for piece in pieces)


def test_markdown_tables_are_extracted():
    markdown = """Some prose here.

| Record type | Period |
| --- | --- |
| Transactions | 7 years |
| Profiles | 24 months |

More prose.
"""

    tables = extract_markdown_tables(markdown)

    assert len(tables) == 1
    assert "Transactions" in tables[0]


def test_prose_without_a_table_yields_nothing():
    assert extract_markdown_tables("No tables in this document at all.") == []
