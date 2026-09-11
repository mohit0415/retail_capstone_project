"""Figures are citable, a gap with an adverb is still a gap, and the PII analyzer loads at startup.

Read out of the log after the project was copied to a new folder:

* the first question stalled for two minutes while Presidio downloaded spaCy's model inside the
  request (the rebuilt virtual environment did not have it), ran out of its deadline and escalated
* "Tell about the investigation and escalation principles" escalated because the only extract that
  matched - a flowchart's caption - had no clause number to cite, and "are not explicitly detailed
  in the provided extracts" was read as a claim that needed one
"""

from types import SimpleNamespace

import pytest

from src.retrieval.adapter import to_retrieved_chunk
from src.retrieval.citations import cite_uncited_sentences, is_gap_statement, uncited_sentences
from src.schemas.models import RetrievedChunk

TITLE = "Anti Bribery Ethical Conduct Policy"

FLOWCHART = (
    "The image is a flowchart outlining the process for handling complaints. It begins with initial "
    "triage, then investigation, then escalation to the ethics committee."
)


def _scored(metadata: dict, text: str = FLOWCHART) -> SimpleNamespace:
    node = SimpleNamespace(metadata=metadata, node_id="figure-node", get_content=lambda: text)

    return SimpleNamespace(node=node, score=0.97)


def _figure(label: str, **extra) -> dict:
    return {
        "document_title": TITLE,
        "doc_type": "anti_bribery_policy",
        "section": "Figures",
        "clause_number": "",
        "content_type": "image_caption",
        "element_label": label,
        **extra,
    }


# ---------------------------------------------------------------------------------------------
# a figure is cited by the page it sits on
# ---------------------------------------------------------------------------------------------


def test_a_figure_is_cited_by_the_page_it_sits_on():
    chunk = to_retrieved_chunk(_scored(_figure("page3_fig0")), fused=True, reranked=True)

    assert chunk.clause_number == "Fig-p3"
    assert chunk.citation == f"{TITLE} §Fig-p3"


def test_a_second_figure_on_a_page_gets_its_own_marker():
    chunk = to_retrieved_chunk(_scored(_figure("page2_fig1")), fused=True, reranked=True)

    assert chunk.clause_number == "Fig-p2-2"


def test_a_caption_without_a_page_label_is_still_citable():
    chunk = to_retrieved_chunk(_scored(_figure("")), fused=True, reranked=True)

    assert chunk.citation == f"{TITLE} §Fig"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"document_title": TITLE, "clause_number": "7.1", "content_type": "text"}, "7.1"),
        ({"document_title": TITLE, "clause_number": "", "content_type": "text"}, ""),
        ({"document_title": TITLE, "content_type": "text"}, "0"),
    ],
)
def test_a_text_extract_keeps_the_clause_number_it_was_indexed_with(metadata, expected):
    assert to_retrieved_chunk(_scored(metadata, "text"), fused=True).clause_number == expected


def test_a_claim_only_the_flowchart_supports_gets_the_figure_citation():
    figure = to_retrieved_chunk(_scored(_figure("page2_fig0")), fused=True, reranked=True)
    answer = "The flowchart outlines a process for handling complaints, which includes initial triage, investigation and escalation."

    _fixed, added = cite_uncited_sentences(answer, [], [figure])

    assert added == [f"{TITLE} §Fig-p2"]


def test_an_extract_cited_by_its_title_alone_is_never_placed_on_a_sentence():
    untitled = RetrievedChunk(
        chunk_id="untitled",
        doc_type="anti_bribery_policy",
        document_title=TITLE,
        section="Notes",
        clause_number="",
        version="1.0",
        content="Complaints go through initial triage, investigation and escalation.",
        rerank_score=0.9,
    )
    answer = "Complaints go through initial triage, investigation and escalation."

    assert cite_uncited_sentences(answer, [], [untitled]) == (answer, [])


# ---------------------------------------------------------------------------------------------
# a gap written with an adverb is still a gap
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "The investigation principles in the anti-bribery policy are not explicitly detailed in the provided extracts.",
        "The policy does not specifically address escalation timelines.",
        "Escalation timelines are not clearly stated in the policy.",
    ],
)
def test_a_gap_with_an_adverb_is_still_a_gap(sentence):
    assert is_gap_statement(sentence)
    assert uncited_sentences(sentence) == []


def test_a_rule_with_an_adverb_is_still_a_rule():
    sentence = "Gifts must not be explicitly solicited from suppliers."

    assert not is_gap_statement(sentence)
    assert uncited_sentences(sentence) == [sentence]


# ---------------------------------------------------------------------------------------------
# the reranker reads the section heading, so a bare bullet list is not scored 0.00
# ---------------------------------------------------------------------------------------------

BULLETS = "- Confidentiality\n- Fairness and neutrality\n- Timely resolution\n- Evidence-based conclusions"


def test_the_reranker_reads_the_heading_of_a_bare_list():
    from src.retrieval.postprocessors import with_heading

    assert with_heading("Investigation Principles", BULLETS) == f"Investigation Principles\n{BULLETS}"


@pytest.mark.parametrize(
    ("heading", "content"),
    [
        ("Figures", "The image is a flowchart outlining the process for handling complaints."),
        ("", "Some text"),
        ("Purpose", "Purpose\nThis policy defines standards to prevent bribery."),
    ],
)
def test_an_uninformative_or_repeated_heading_is_not_added(heading, content):
    from src.retrieval.postprocessors import with_heading

    assert with_heading(heading, content) == content


def test_flashrank_is_given_the_heading_of_each_extract():
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from src.retrieval.postprocessors import FlashRankRerank

    seen: list[str] = []

    class _Ranker:
        def rerank(self, request):
            seen.extend(passage["text"] for passage in request.passages)

            return [{"id": 0, "score": 0.9}]

    node = TextNode(text=BULLETS, metadata={"section": "Investigation Principles"})

    FlashRankRerank(ranker=_Ranker(), top_n=1).postprocess_nodes(
        [NodeWithScore(node=node, score=0.1)], query_bundle=QueryBundle("investigation principles")
    )

    assert seen == [f"Investigation Principles\n{BULLETS}"]


def test_the_chunk_reranker_is_given_the_heading_too(monkeypatch):
    import sys
    import types

    from src.retrieval import reranker

    seen: list[str] = []

    class _Ranker:
        def __init__(self, **kwargs):
            pass

        def rerank(self, request):
            seen.extend(passage["text"] for passage in request.passages)

            return [{"id": 0, "score": 0.8}]

    fake = types.SimpleNamespace(Ranker=_Ranker, RerankRequest=lambda query, passages: types.SimpleNamespace(passages=passages))

    monkeypatch.setitem(sys.modules, "flashrank", fake)
    monkeypatch.setattr(reranker.settings, "enable_flashrank_rerank", True)

    chunk = RetrievedChunk(
        chunk_id="7.1",
        doc_type="anti_bribery_policy",
        document_title=TITLE,
        section="Investigation Principles",
        clause_number="7.1",
        version="1.0",
        content=BULLETS,
    )

    ranked = reranker.rerank("investigation principles", [chunk], top_n=1)

    assert seen == [f"Investigation Principles\n{BULLETS}"]
    assert ranked[0].rerank_score == 0.8


# ---------------------------------------------------------------------------------------------
# the PII analyzer loads at startup, and says how to install the model when it is missing
# ---------------------------------------------------------------------------------------------


def test_the_pii_warm_up_names_the_install_command_when_the_model_is_missing(monkeypatch):
    import importlib.util

    from src.guardrails import pii

    warnings: list[str] = []

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None)
    monkeypatch.setattr(pii, "_load_presidio", lambda: (False, False))
    monkeypatch.setattr(pii.logger, "warning", lambda message, *args: warnings.append(message % args))

    assert pii.warm_up() is False
    assert any("uv run python -m spacy download en_core_web_lg" in line for line in warnings)


def test_the_pii_warm_up_reports_a_loaded_analyzer(monkeypatch):
    import importlib.util

    from src.guardrails import pii

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: object())
    monkeypatch.setattr(pii, "_load_presidio", lambda: (object(), object()))

    assert pii.warm_up() is True
