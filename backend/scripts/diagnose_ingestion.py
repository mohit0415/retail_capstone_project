import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.bootstrap import corpus_dir
from src.ingestion.extractors.clause_extraction import extract_clauses
from src.ingestion.extractors.figure_detection import detect_figures
from src.ingestion.extractors.pdf_backend import open_pdf
from src.ingestion.extractors.table_extraction import extract_tables, find_markdown_tables
from src.ingestion.loaders.dispatch import load_document
from src.ingestion.router import choose_route


def picture_counts(path: Path) -> tuple[int, int, int]:
    _pymupdf, document = open_pdf(str(path))

    if document is None:
        return -1, -1, -1

    rasters = 0
    drawings = 0
    figures = 0

    for index in range(len(document)):
        page = document[index]
        rasters += len(page.get_images(full=True))
        drawings += len(page.get_drawings())
        figures += len(detect_figures(page))

    document.close()

    return rasters, drawings, figures


def survey(directory: Path) -> None:
    print(f"corpus: {directory}\n")

    for path in sorted(directory.glob("*.pdf")):
        rasters, drawings, figures = picture_counts(path)
        route = choose_route(str(path), ".pdf")

        print(f"{path.name}")
        print(f"  embedded images {rasters}")
        print(f"  vector drawings {drawings}")
        print(f"  drawn figures   {figures}")
        print(f"  route           {route.parser}")
        print(f"  because         {route.reason}\n")

    print(
        "embedded images = raster pictures stored in the file.\n"
        "vector drawings = every line, box and curve, table borders included.\n"
        "drawn figures   = clusters of those drawings that look like a diagram once table\n"
        "                  borders, rules and single bordered text boxes are removed. Both\n"
        "                  embedded images and drawn figures become image_caption nodes."
    )


def inspect(path: Path, show_chars: int) -> None:
    print(f"parsing {path.name}\n")

    parsed = load_document(str(path))
    sections = extract_clauses(parsed.text)
    tables, _reduced = extract_tables(sections)

    print(f"  parsed_with      {parsed.route.parser}")
    print(f"  parse_reason     {parsed.route.reason}")
    print(f"  characters       {len(parsed.text)}")
    print(f"  clause sections  {len(sections)}")
    print(f"  tables promoted  {len(tables)}")

    if not parsed.route.is_multimodal:
        print("\n  this document went through the text reader, so no table or image node is built")
        print("  for it even if the markdown below happens to contain a pipe table.\n")

    if tables:
        first = tables[0]

        print(f"\nfirst table, attributed to clause {first.clause_number} ({first.heading}):\n")
        print(first.markdown[:1200])
        return

    raw = find_markdown_tables(parsed.text)

    if raw:
        print(f"\n{len(raw)} table-shaped block(s) matched but carried no data rows, so they were")
        print("dropped as mis-parsed diagrams. First block:\n")
        print(raw[0][:1200])
        return

    lines = [line for line in parsed.text.splitlines() if "|" in line or "<table" in line.lower()]

    if not lines:
        print("\nthe parsed markdown contains no '|' and no '<table' at all, so the parser")
        print("flattened the tables into prose before the extractor ever ran.")
        print(f"\nfirst {show_chars} characters of the parsed markdown:\n")
        print(parsed.text[:show_chars])
        return

    print("\nno table matched. First pipe-bearing lines of the parsed markdown:\n")

    for line in lines[:25]:
        print("  " + repr(line))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show why table and image nodes were or were not built")
    parser.add_argument("--path", default=None, help="one file to parse and inspect")
    parser.add_argument("--chars", type=int, default=1500, help="characters of markdown to show")
    args = parser.parse_args()

    if args.path:
        inspect(Path(args.path), args.chars)
        return

    survey(corpus_dir())


if __name__ == "__main__":
    main()
