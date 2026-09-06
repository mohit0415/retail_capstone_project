import pytest

from src.ingestion.elements import ClauseSection, RawImage, RawTable
from src.ingestion.extractors.clause_extraction import extract_clauses
from src.ingestion.extractors.table_extraction import (
    extract_tables,
    find_markdown_tables,
    has_data_rows,
    normalize_markdown_table,
)
from src.ingestion.metadata.schema import (
    CONTENT_TYPE_IMAGE,
    CONTENT_TYPE_TABLE,
    CONTENT_TYPE_TEXT,
    validate_document_metadata,
    validate_node_metadata,
)
from src.ingestion.metadata.stamping import (
    image_node_metadata,
    table_node_metadata,
    text_node_metadata,
)
from src.ingestion.metadata.structural import infer_doc_type, parse_frontmatter
from src.ingestion.router import choose_route

SAMPLE = """---
doc_type: retention_policy
title: Records Retention Policy
version: 3.2
effective_date: 2025-04-01
---

## 4.1 Transaction records

Point of sale transaction records are retained for seven financial years.

| Record type | Period |
| --- | --- |
| Transactions | 7 years |
| Profiles | 24 months |

## 4.2 Customer profiles

Loyalty profiles are retained for twenty-four months after the last interaction.

### A.8.15 Logging

Access logs are retained for two years.
"""

DOCUMENT_METADATA = {
    "domain": "retail_compliance",
    "doc_type": "retention_policy",
    "document_title": "Records Retention Policy",
    "version": "3.2",
    "effective_date": "2025-04-01",
    "is_current": True,
    "owning_department": "legal",
    "topic": "records retention",
    "keywords": ["retention", "records"],
    "summary": "How long each record type is kept.",
    "content_domain": "obligation",
    "intended_route": "rag",
    "original_file_name": "Data_Retention_Policy.pdf",
    "file_name": "policy_upload_x1.pdf",
    "file_type": "pdf",
    "file_size_kb": 12.4,
    "file_hash": "a" * 32,
    "ingested_at": "2026-09-03T09:00:00+00:00",
    "parsed_with": "llamaparse",
    "parse_reason": "2 page(s) carry a table, so the layout has to survive the parse",
    "embed_model": "text-embedding-3-small",
    "level": "document",
}


def test_frontmatter_is_parsed_and_stripped():
    frontmatter, body = parse_frontmatter(SAMPLE)

    assert frontmatter["doc_type"] == "retention_policy"
    assert frontmatter["version"] == "3.2"
    assert body.lstrip().startswith("## 4.1")


def test_clause_numbers_survive_splitting():
    _, body = parse_frontmatter(SAMPLE)

    assert [section.clause_number for section in extract_clauses(body)] == ["4.1", "4.2", "A.8.15"]


def test_clause_headings_are_captured():
    _, body = parse_frontmatter(SAMPLE)

    assert "Transaction records" in [section.heading for section in extract_clauses(body)]


def test_unnumbered_document_still_produces_one_section():
    sections = extract_clauses("Just a paragraph with no headings at all.")

    assert len(sections) == 1
    assert sections[0].clause_number == "1"


def test_doc_type_is_inferred_from_the_filename():
    assert infer_doc_type("vendor_policy.md") == "vendor_policy"
    assert infer_doc_type("gdpr_articles.md") == "gdpr"
    assert infer_doc_type("iso_27001.md") == "iso_27001"
    assert infer_doc_type("anti_bribery_policy.md") == "anti_bribery_policy"


def test_an_unrecognisable_filename_yields_no_doc_type():
    assert infer_doc_type("scan_0042_final_v2.pdf") == ""


def test_markdown_tables_are_found():
    tables = find_markdown_tables(SAMPLE)

    assert len(tables) == 1
    assert "Transactions" in tables[0]


def test_prose_without_a_table_yields_nothing():
    assert find_markdown_tables("No tables in this document at all.") == []


def test_a_promoted_table_carries_the_clause_it_sits_in():
    _, body = parse_frontmatter(SAMPLE)

    tables, reduced = extract_tables(extract_clauses(body))

    assert len(tables) == 1
    assert tables[0].clause_number == "4.1"
    assert tables[0].heading == "Transaction records"
    assert "|" not in reduced[0].body
    assert "indexed as its own node" in reduced[0].body


def test_a_header_only_table_is_dropped():
    block = "| Step | Owner |\n| --- | --- |\n|  |  |\n"

    assert has_data_rows(block) is False
    assert extract_tables([ClauseSection("Process", "5.1", block, 0)])[0] == []


def test_ragged_rows_are_padded_to_the_header_width():
    ragged = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 |\n"

    rebuilt = normalize_markdown_table(ragged)

    assert rebuilt.splitlines()[2] == "| 1 | 2 |  |"


def test_a_markdown_file_never_goes_to_llamaparse(tmp_path):
    path = tmp_path / "vendor_policy.md"
    path.write_text(SAMPLE, encoding="utf-8")

    route = choose_route(str(path))

    assert route.parser == "llamaindex"
    assert route.is_multimodal is False
    assert "already structured text" in route.reason


def test_the_document_metadata_fixture_satisfies_the_contract():
    assert validate_document_metadata(DOCUMENT_METADATA) == []


def test_every_node_kind_satisfies_the_node_contract():
    section = ClauseSection("Transaction records", "4.1", "body", 0)
    table = RawTable("| A |\n| --- |\n| 1 |", 0, "Transaction records", "4.1")
    image = RawImage("/tmp/page2_img0.png", 2, 0)

    text_metadata = text_node_metadata(DOCUMENT_METADATA, section, 0)
    table_metadata = table_node_metadata(DOCUMENT_METADATA, table)
    image_metadata = image_node_metadata(DOCUMENT_METADATA, image, "/store/page2_img0.png")

    assert validate_node_metadata(text_metadata) == []
    assert validate_node_metadata(table_metadata) == []
    assert validate_node_metadata(image_metadata) == []

    assert text_metadata["content_type"] == CONTENT_TYPE_TEXT
    assert table_metadata["content_type"] == CONTENT_TYPE_TABLE
    assert image_metadata["content_type"] == CONTENT_TYPE_IMAGE


def test_an_image_node_records_how_the_picture_was_obtained():
    raster = RawImage("/tmp/page2_img0.png", 2, 0)
    figure = RawImage("/tmp/page2_fig0.png", 2, 0, kind="figure")

    raster_metadata = image_node_metadata(DOCUMENT_METADATA, raster, "/store/a.png")
    figure_metadata = image_node_metadata(DOCUMENT_METADATA, figure, "/store/b.png")

    assert raster_metadata["image_kind"] == "raster"
    assert raster_metadata["element_label"] == "page2_img0"
    assert figure_metadata["image_kind"] == "figure"
    assert figure_metadata["element_label"] == "page2_fig0"
    assert validate_node_metadata(figure_metadata) == []


def test_a_drawn_flowchart_is_detected_and_a_bordered_text_box_is_not(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")

    from src.ingestion.extractors.figure_detection import detect_figures

    document = pymupdf.open()
    page = document.new_page()

    for step in range(3):
        top = 200 + step * 60
        page.draw_rect(pymupdf.Rect(120, top, 320, top + 40), fill=(0.1, 0.2, 0.6))

    for connector in range(2):
        top = 240 + connector * 60
        page.draw_line((220, top), (220, top + 20))

    page.draw_rect(pymupdf.Rect(60, 600, 540, 650))

    path = tmp_path / "drawn.pdf"
    document.save(path)
    document.close()

    document = pymupdf.open(path)
    boxes = detect_figures(document[0])
    document.close()

    assert len(boxes) == 1
    assert boxes[0].y0 < 260 and boxes[0].y1 > 300
    assert boxes[0].y1 < 600


def test_citations_name_the_clause_or_the_page():
    section = ClauseSection("Transaction records", "4.1", "body", 0)
    image = RawImage("/tmp/page2_img0.png", 2, 0)

    assert text_node_metadata(DOCUMENT_METADATA, section, 0)["citation"] == (
        "Records Retention Policy §4.1"
    )
    assert image_node_metadata(DOCUMENT_METADATA, image, "/store/x.png")["citation"] == (
        "Records Retention Policy, page 2"
    )


def test_a_node_with_a_mismatched_modality_is_rejected():
    section = ClauseSection("Transaction records", "4.1", "body", 0)

    metadata = text_node_metadata(DOCUMENT_METADATA, section, 0)
    metadata["modality"] = "table"

    assert validate_node_metadata(metadata) != []


def test_a_node_missing_the_doc_type_is_rejected():
    section = ClauseSection("Transaction records", "4.1", "body", 0)

    metadata = text_node_metadata(DOCUMENT_METADATA, section, 0)
    metadata.pop("doc_type")

    assert "missing node key 'doc_type'" in validate_node_metadata(metadata)


def test_recursive_splitter_respects_the_chunk_ceiling():
    from configs.settings import settings
    from src.ingestion.processors.splitters import build_recursive_splitter

    splitter = build_recursive_splitter()
    text = "This is a sentence about retention. " * 400

    pieces = splitter.split_text(text)

    assert len(pieces) > 1
    assert all(len(piece) <= settings.max_chunk_chars + settings.chunk_overlap_chars for piece in pieces)


def test_a_document_with_no_recognisable_type_is_refused(tmp_path, monkeypatch):
    from src.ingestion.elements import ParsedDocument, ParseRoute
    from src.ingestion.metadata import document as document_module

    path = tmp_path / "scan_0042.md"
    path.write_text("Some prose with no clause numbers.", encoding="utf-8")

    monkeypatch.setattr(document_module, "extract_content_metadata", lambda _body: {
        "topic": "unknown",
        "keywords": [],
        "owning_department": "all",
        "content_domain": "mixed",
        "intended_route": "rag",
        "summary": "",
    })

    parsed = ParsedDocument(text="body", route=ParseRoute(parser="llamaindex", reason="text only"))

    with pytest.raises(document_module.MetadataContractError):
        document_module.build_document_metadata(
            file_path=str(path),
            original_filename=path.name,
            parsed=parsed,
            frontmatter={},
            body="body",
            embed_model="text-embedding-3-small",
        )
