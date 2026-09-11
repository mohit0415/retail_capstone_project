"""The multi-agent workflow as people read it.

Every node span in the graph trace carries a readable agent name, a one-line
summary of what that step decided, and the model tier it used (``None`` when the
step made no model call). ``/ask`` returns these as ``trace.agent_steps`` so the
chat can draw the workflow, and the reviewer package gets them through the
reasoning trace.

The names follow the five agents the capstone brief asks for (Intent
Classification, Retrieval, Compliance Validation, Risk Assessment, Escalation
Manager) plus the supporting steps around them.
"""

from typing import Any

AGENT_FOR_NODE: dict[str, tuple[str, str]] = {
    "input_guardrail": ("Input Guardrail", "guard"),
    "query_rewrite": ("Query Rewrite", "step"),
    "intent_classification": ("Intent Classification Agent", "agent"),
    "entity_resolution": ("Entity Resolution", "step"),
    "risk_assessment": ("Risk Assessment Agent", "agent"),
    "planner": ("Planner", "plan"),
    "rag_path": ("Retrieval Agent · RAG", "agent"),
    "nl2sql_path": ("Retrieval Agent · SQL", "agent"),
    "hybrid_path": ("Retrieval Agent · Hybrid", "agent"),
    "agentic_rag": ("Retrieval Agent · Agentic", "agent"),
    "multi_agent_panel": ("High-Risk Panel", "agent"),
    "compliance_validation": ("Compliance Validation Agent", "agent"),
    "reflection": ("Reflection (self-correction)", "reflect"),
    "confidence_scoring": ("Confidence Scoring", "step"),
    "no_answer": ("Not-Found Responder", "step"),
    "output_guardrail": ("Output Guardrail", "guard"),
    "escalation_manager": ("Escalation Manager Agent", "agent"),
    "safe_refusal": ("Safe Refusal", "guard"),
    "clarification": ("Clarification", "step"),
}

PANEL_AGENT_LABELS = {
    "policy_interpreter": "Policy Interpreter",
    "data_verifier": "Data Verifier",
    "challenger": "Challenger",
}

SUMMARY_LIMIT = 220


def _value(item: Any, name: str, default: Any = None) -> Any:
    if item is None:
        return default

    if isinstance(item, dict):
        return item.get(name, default)

    return getattr(item, name, default)


def _enum_text(item: Any) -> str:
    return str(getattr(item, "value", item) if item is not None else "")


def _short(text: str, limit: int = SUMMARY_LIMIT) -> str:
    text = " ".join(str(text or "").split())

    return text if len(text) <= limit else text[: limit - 1] + "…"


def _best_score(chunks: list) -> float | None:
    scores = []

    for chunk in chunks[:5]:
        rerank = _value(chunk, "rerank_score")
        dense = _value(chunk, "dense_score")
        score = rerank if rerank is not None else dense

        if score is not None:
            scores.append(float(score))

    return max(scores) if scores else None


def _tier(node: str, merged: dict, result: dict) -> str | None:
    routing = result.get("model_routing") or []

    if routing:
        return str(_value(routing[-1], "tier") or "") or None

    if node == "query_rewrite":
        return "small" if merged.get("rewrite_mode") in ("rewritten", "unchanged") else None

    if node == "intent_classification":
        return "small"

    if node == "risk_assessment":
        return "small (ran with intent)" if merged.get("risk_l2") else "small"

    if node == "planner":
        planner = _value(merged.get("path_decision"), "planner")

        return "small" if planner == "llm" else None

    if node == "reflection":
        return "small" if merged.get("repair_strategy") == "llm_replan" else None

    if node == "nl2sql_path":
        return "small"

    if node == "multi_agent_panel":
        return "strong"

    return None


def _summary(node: str, merged: dict, result: dict) -> str:
    if node == "input_guardrail":
        if merged.get("terminal_outcome") == "refused":
            return f"refused: {merged.get('refusal_reason') or 'blocked'}"

        return "passed (injection, PII and scope checks)"

    if node == "query_rewrite":
        mode = merged.get("rewrite_mode")

        return {
            "first_turn": "first question in the chat, nothing to rewrite",
            "standalone": "follow-up already reads as a full question, model call skipped",
            "rewritten": f"rewritten to: {merged.get('standalone_query')}",
            "unchanged": "model kept the question as it was",
        }.get(mode, "checked the question against the chat history")

    if node == "intent_classification":
        intent = merged.get("intent")
        entities = _value(intent, "entities") or []
        text = f"intent {_enum_text(_value(intent, 'intent')) or 'unknown'}"

        if entities:
            text += " · entities " + ", ".join(str(_value(e, "text")) for e in entities[:4])

        if merged.get("risk_l2"):
            text += " · risk check started in parallel"

        return text

    if node == "entity_resolution":
        resolved = merged.get("resolved_entities") or []

        if not resolved:
            return "no vendor or department named"

        return ", ".join(
            f"{_value(e, 'surface_form')} → {_enum_text(_value(e, 'status'))}" for e in resolved[:4]
        )

    if node == "risk_assessment":
        risk = merged.get("risk")
        level = _enum_text(_value(risk, "final_level")) or "unknown"
        scenario = _value(risk, "scenario_id") or "none"
        ran = _value(risk, "probes_run") or []
        skipped = _value(risk, "probes_skipped") or []

        if ran:
            probes = f"DB probes ran: {', '.join(ran)}"
        elif skipped:
            probes = "DB probes skipped (the question names no vendor or department)"
        else:
            probes = "no DB probe for this scenario"

        return f"risk {level} · scenario {scenario} · {probes}"

    if node == "planner":
        decision = merged.get("path_decision") or {}
        path = _value(decision, "path") or merged.get("routed_path") or "?"
        mode = _value(decision, "planner")
        how = {
            "deterministic": "only one path possible, planner model skipped",
            "vetted_query": "a vetted query answers it directly, planner model skipped",
            "reflection": "re-plan from reflection",
            "llm": "planned by the model",
        }.get(mode, "")

        return f"path {path}" + (f" · {how}" if how else "")

    if node in ("rag_path", "hybrid_path", "agentic_rag"):
        chunks = result.get("retrieved_chunks") or []
        best = _best_score(chunks)
        draft = result.get("draft")
        cited = len(_value(draft, "cited_clauses") or [])
        text = f"{len(chunks)} extracts"

        if best is not None:
            text += f" · best match {best:.2f}"

        evidence = result.get("sql_evidence")
        route = result.get("evidence_path") or ""

        if evidence is not None:
            text += f" · {_value(evidence, 'row_count', 0)} rows"

        if node == "rag_path" and route in ("hybrid", "high_risk_panel"):
            handed = "the multi-agent panel" if route == "high_risk_panel" else "the records stage"
            text += f" · clauses handed to {handed} (stage 1 of {route})"
        elif draft is None and node == "rag_path":
            if result.get("not_found_reason") == "not_in_extracts":
                text += " · no draft written (the extracts do not answer the question)"
            else:
                text += " · no draft written (nothing matched the question)"
        else:
            text += f" · draft cites {cited} clause(s)"

        if node == "rag_path" and merged.get("reflection_count") and merged.get("replan_directive"):
            text += " · repair pass (wider search + validator feedback)"

        return text

    if node == "nl2sql_path":
        evidence = result.get("sql_evidence")
        route = result.get("evidence_path") or ""

        if evidence is None:
            failure = result.get("sql_failure")
            text = "no records returned" + (f" ({failure})" if failure else "")
        else:
            text = f"template {_value(evidence, 'template_id')} · {_value(evidence, 'row_count', 0)} rows"
            how = {
                "matched": "matched without a model call",
                "selector": "picked by the template selector",
                "generated": "no vetted query fitted, SQL generated",
            }.get(_value(evidence, "selection") or "")

            if how:
                text += f" · {how}"

            if _value(evidence, "scope_note"):
                text += f" · scope: {_value(evidence, 'scope_note')}"

        if route == "hybrid":
            clauses = len(merged.get("retrieved_chunks") or [])
            cited = len(_value(result.get("draft"), "cited_clauses") or [])
            text += f" · reconciled with {clauses} clause(s) from the RAG stage, draft cites {cited}"
        elif route == "high_risk_panel":
            text += " · rows handed to the multi-agent panel"

        if merged.get("repair_strategy") in ("sql_renarrate", "hybrid_redraft") and merged.get("reflection_count"):
            text += " · repair pass (same rows, answer rewritten with validator feedback)"

        return text

    if node == "multi_agent_panel":
        verdict = result.get("panel_verdict")
        dissent = len(_value(verdict, "dissent") or [])
        repairs = result.get("panel_repair_count") or 0
        text = f"interpreter + verifier (parallel) → challenger → consensus · {dissent} dissent"

        if repairs:
            text += f" · {repairs} repair pass"

        if _value(verdict, "unresolved_conflict"):
            text += " · unresolved conflict"

        return text

    if node == "compliance_validation":
        validation = result.get("validation")
        defects = _value(validation, "defects") or []

        if _value(validation, "passed"):
            text = "passed" + (f" · {len(defects)} minor note(s)" if defects else "")
        else:
            kinds = sorted({_enum_text(_value(d, "defect_type")) for d in defects})
            text = "failed: " + (", ".join(kinds) or "defects found")

        skipped = result.get("validation_llm_skipped")

        if skipped:
            text += f" · model call skipped ({skipped})"

        return text

    if node == "reflection":
        strategy = merged.get("repair_strategy")
        attempt = merged.get("reflection_count") or 0

        if merged.get("escalation_reason") and not strategy:
            return f"not repairable: {merged.get('escalation_reason')}"

        if strategy == "rag_widen_and_fix":
            return f"attempt {attempt}: search wider and send the defects back to the writer (no model call)"

        if strategy == "sql_renarrate":
            return f"attempt {attempt}: keep the rows and send the defects back to the answer writer (no model call)"

        if strategy == "hybrid_redraft":
            return (
                f"attempt {attempt}: keep the rows, search the policies wider and send the defects back to the "
                "writer (no model call)"
            )

        return f"attempt {attempt}: model re-planned the evidence"

    if node == "confidence_scoring":
        confidence = merged.get("confidence")
        score = _value(confidence, "final_score")

        return f"confidence {score:.2f}" if isinstance(score, int | float) else "scored"

    if node == "no_answer":
        reason = result.get("not_found_reason") or merged.get("not_found_reason")

        return {
            "not_in_extracts": "the extracts do not answer the question · replied 'I don't know' instead of guessing",
            "no_records": "the records query produced no evidence · replied 'I don't know' instead of guessing",
            "no_access": "the only source that answers this is outside the role's grant · said so instead of guessing",
        }.get(reason, "nothing in the searchable documents matches · replied 'I don't know' instead of guessing")

    if node == "output_guardrail":
        if merged.get("terminal_outcome") == "answered":
            return "released (PII scrub + citation check)"

        return f"rejected: {merged.get('escalation_reason') or 'failed the release checks'}"

    if node == "escalation_manager":
        reference = merged.get("escalation_reference")

        return "sent to human review" + (f" · ref {reference}" if reference else "")

    if node == "safe_refusal":
        return f"refused: {merged.get('refusal_reason') or ''}"

    if node == "clarification":
        return f"asked: {merged.get('clarification_question') or ''}"

    return ""


def describe_step(node: str, merged: dict, result: dict | None = None, status: str = "completed") -> dict:
    """The readable fields added to one trace span."""
    agent, kind = AGENT_FOR_NODE.get(node, (node.replace("_", " ").title(), "step"))
    result = result or {}

    if status == "budget_stopped":
        stops = result.get("budget_stops") or []
        reason = _value(stops[-1], "reason") if stops else "budget"

        return {"agent": agent, "kind": kind, "summary": f"stopped by the budget guard ({reason})", "tier": None}

    try:
        summary = _short(_summary(node, merged, result))
    except Exception:  # a summary must never break the request
        summary = ""

    return {"agent": agent, "kind": kind, "summary": summary, "tier": _tier(node, merged, result)}


def _panel_children(final_state: dict) -> list[dict]:
    verdict = final_state.get("panel_verdict")

    if verdict is None:
        return []

    children = []

    for opinion in _value(verdict, "opinions") or []:
        name = str(_value(opinion, "agent") or "")
        objections = _value(opinion, "objections") or []
        citations = _value(opinion, "supporting_citations") or []
        detail = _short(_value(opinion, "position") or "", 180)

        if citations:
            detail += f" · cites {len(citations)}"

        if objections:
            detail += f" · {len(objections)} objection(s)"

        children.append({"agent": PANEL_AGENT_LABELS.get(name, name), "summary": detail})

    consensus = _value(verdict, "consensus")

    if consensus:
        dissent = _value(verdict, "dissent") or []
        children.append(
            {
                "agent": "Consensus",
                "summary": _short(consensus, 180) + (f" · {len(dissent)} dissent" if dissent else ""),
            }
        )

    return children


def build_agent_steps(final_state: dict) -> list[dict]:
    """The ordered workflow for the UI, one entry per node run (repairs appear twice)."""
    steps: list[dict] = []
    seen: dict[str, int] = {}
    spans = final_state.get("trace") or []
    last_panel = max((i for i, span in enumerate(spans) if _value(span, "node") == "multi_agent_panel"), default=-1)

    for index, span in enumerate(spans):
        node = str(_value(span, "node") or "")
        seen[node] = seen.get(node, 0) + 1
        agent, kind = AGENT_FOR_NODE.get(node, (node.replace("_", " ").title(), "step"))

        step = {
            "node": node,
            "agent": _value(span, "agent") or agent,
            "kind": _value(span, "kind") or kind,
            "status": _value(span, "status") or "completed",
            "elapsed_ms": float(_value(span, "elapsed_ms") or 0.0),
            "since_start_ms": float(_value(span, "since_start_ms") or 0.0),
            "summary": _value(span, "summary") or "",
            "tier": _value(span, "tier"),
            "attempt": seen[node],
            "children": [],
        }

        if node == "multi_agent_panel" and index == last_panel:
            step["children"] = _panel_children(final_state)

        steps.append(step)

    return steps
