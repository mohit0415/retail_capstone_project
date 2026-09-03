# Policy corpus

Drop the seven mandated source documents here. The ingestion script reads every `.md`, `.pdf`,
`.docx` and `.txt` in this folder, so nothing else needs changing when you add them.

## Required files

| File | `doc_type` | What it is |
|---|---|---|
| `privacy_policy.*` | `privacy_policy` | Customer and employee personal data handling |
| `retention_policy.*` | `retention_policy` | How long each record class is kept, and disposal |
| `vendor_policy.*` | `vendor_policy` | Third-party onboarding, due diligence, approval |
| `anti_bribery_policy.*` | `anti_bribery_policy` | Gifts, hospitality, facilitation payments |
| `infosec_policy.*` | `infosec_policy` | Access control, encryption, incident handling |
| `gdpr_articles.*` | `gdpr` | GDPR Articles 5, 6, 17 and 32 |
| `iso_27001.*` | `iso_27001` | The ISO 27001 Annex A controls the policies point at |

The `doc_type` values matter — `src/auth/rbac.py` grants document access by exactly these strings,
so a typo silently hides a document from every role. `doc_type` is taken from the frontmatter when
present, and otherwise inferred from the filename by `src/ingestion/policy_metadata.py`.

## Which parser each file gets

`src/ingestion/parser.py` opens PDFs with pdfplumber and DOCX with python-docx and checks whether
any page holds an extractable table or an image.

- **Tables or images found** → LlamaParse, which returns layout-aware markdown with tables intact.
  Needs `LLAMAPARSE_API_KEY` in `.env`; without it, ingestion of that file fails with a clear error
  rather than silently producing a worse parse.
- **Neither found** → parsed locally. Markdown and text files are read directly; other formats go
  through `SimpleDirectoryReader`. No API key needed.

So a markdown-only corpus needs no LlamaParse key at all.

## File format for markdown

```markdown
---
doc_type: retention_policy
title: Records Retention Policy
version: 3.2
effective_date: 2025-04-01
department: legal
---

## 4.1 Transaction records

Point of sale transaction records are retained for seven years from the end of the
financial year in which the transaction occurred...

## 4.2 Customer profiles

Loyalty programme profiles are retained for the duration of the membership and for
twenty-four months after the last recorded interaction...
```

Rules the ingester relies on:

- **The heading must carry a clause number as its first token** — `## 4.1 Transaction records`,
  `### A.8.15 Logging`, or `## §17.1 Right to erasure`. That number becomes `clause_number` and it
  is what every citation in every answer points at, so it must be stable across versions. A document
  with no numbered headings still ingests, but the whole file becomes one section numbered `1` and
  its citations are useless.
- Heading depth can be `#` to `####`. Depth is not significant; the clause number is.
- `effective_date` must be ISO format. Retrieval drops anything effective after the pinned
  `AS_OF_DATE`, so a document dated in the future is invisible to the system.
- Bump `version` when you re-publish. Ingestion marks every other version of that `doc_type` as
  `is_current: false`, which is what keeps a superseded clause out of an answer while leaving it in
  the table for audit.

For PDFs and DOCX there is no frontmatter, so `doc_type` is inferred from the filename and
`version` defaults to `1.0`. Name those files to match the table above.

## Running the ingest

```bash
uv run python scripts/ingest_policies.py                 # everything in this folder
uv run python scripts/ingest_policies.py --path data/policies/gdpr_articles.pdf
uv run python scripts/ingest_policies.py --check         # embedding compatibility only
uv run python scripts/ingest_policies.py --force         # re-ingest despite a known file hash
```

Re-running is safe. Each file is hashed and an unchanged file is skipped unless `--force` is given.

`--check` is worth running before any re-ingest after changing the embedding deployment: vectors
from two different embedding models are not comparable, and mixing them makes retrieval return
confident nonsense rather than failing. The check refuses the run and tells you to clear the table.

## What comes out

Each document produces three kinds of node, all in the same vector table:

- **text** — clause-scoped, split by `SemanticSplitterNodeParser`, with LangChain's
  `RecursiveCharacterTextSplitter` bounding anything oversized
- **table_summary** — one per markdown table; the summary is embedded, the original table is kept in
  metadata and handed to the answering model
- **image_caption** — one per extracted diagram; the image is copied to `IMAGES_STORAGE_DIR` and its
  path stored, so a UI can render it beside the answer

The ingest script prints the per-file counts, and `SELECT * FROM policy_kb_stats` shows them
afterwards.
