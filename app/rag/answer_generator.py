import re

from app.rag.llm_client import safe_completion_json, safe_completion_text

# Bump when any generation/verification prompt changes semantically. It is passed
# to the LLM client as an explicit cache namespace so cached responses from an
# older prompt contract can never be served (see llm_client._cache_path).
ANSWER_PROMPT_VERSION = "answer-gen-v6"

_SERVICE_FALLBACK = "I could not generate an answer due to a temporary service issue."


def _is_consolidated_group(value: object) -> bool:
    """Return True when a value matches the Phase 2 consolidation contract."""
    return isinstance(value, dict) and any(
        key in value
        for key in (
            "status",
            "resolved_fact",
            "resolution_basis",
            "raw_candidates",
        )
    )


def _format_fact(fact: dict) -> str:
    """Format one normalized fact without inventing information."""
    subject = fact.get("subject", "")
    attribute = fact.get("attribute", "")
    value = fact.get("value", "")
    statement = fact.get("statement", "")
    value_type = fact.get("value_type", "unknown")
    temporal_scope = fact.get("temporal_scope", "unknown")
    section_type = fact.get("section_type", "unknown")

    return (
        f"Subject: {subject}\n"
        f"Attribute: {attribute}\n"
        f"Value: {value}\n"
        f"Statement: {statement}\n"
        f"Value type: {value_type}\n"
        f"Temporal scope: {temporal_scope}\n"
        f"Section type: {section_type}"
    )


def _format_consolidated_group(
    group_key: str,
    result: dict,
) -> tuple[str, list[str]]:
    """Build generation context from a Phase 2 consolidated requirement."""
    lines = [f"GROUP: {group_key}"]
    sources: list[str] = []

    status = result.get("status", "insufficient")
    resolution_basis = result.get("resolution_basis")
    target_entities = result.get("target_entities") or []

    lines.append(f"STATUS: {status}")
    if target_entities:
        lines.append("VERIFIED TARGET ENTITIES: " + ", ".join(str(x) for x in target_entities))
        lines.append("ENTITY SCOPE: Only evidence already admitted for these targets is usable.")

    if resolution_basis:
        lines.append(f"RESOLUTION BASIS: {resolution_basis}")

    resolved_fact = result.get("resolved_fact")

    if isinstance(resolved_fact, dict):
        lines.append("RESOLVED FACT (AUTHORITATIVE for this attribute; other requested details may still be needed):")
        lines.append(_format_fact(resolved_fact))

    facts = result.get("facts", [])

    # Once consolidation has resolved a requirement, the resolved fact is the
    # application-level source of truth. Do not expose competing normalized
    # alternatives to the generator because that invites the model to
    # re-resolve a conflict that Python has already resolved.
    if status != "resolved" and facts:
        lines.append("ALL NORMALIZED FACTS:")
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                continue
            lines.append(f"FACT {index}:")
            lines.append(_format_fact(fact))

    evidence = result.get("evidence", result.get("raw_candidates", []))

    # Raw evidence is shown for unresolved/supported groups as normal grounding
    # context. For resolved groups, expose it only as supplementary evidence:
    # the resolved fact remains authoritative, while non-conflicting requested
    # details that were not selected as the primary resolved value may still be
    # needed to answer the user's full question.
    expose_raw_evidence = status not in {"insufficient", "resolved"}
    expose_supplementary_evidence = status == "resolved"

    if evidence and (expose_raw_evidence or expose_supplementary_evidence):
        if expose_supplementary_evidence:
            lines.append(
                "ADDITIONAL IN-SCOPE EVIDENCE (use it to answer OTHER parts of the question; "
                "never to change the RESOLVED FACT):"
            )
        else:
            lines.append("RAW SUPPORTING EVIDENCE:")

    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue

        nested = item.get("result", item)
        if not isinstance(nested, dict):
            continue

        metadata = nested.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        source = metadata.get("source") or metadata.get("filename") or "Unknown source"

        if source not in sources:
            sources.append(source)

        if expose_raw_evidence or expose_supplementary_evidence:
            document = nested.get("document", "")
            lines.append(
                f"Candidate {index} | Source: {source}\n"
                f"Evidence:\n{document}"
            )

    return "\n".join(lines), sources


def _format_raw_group(
    group_key: str,
    items: list[dict],
) -> tuple[str, list[str]]:
    """Backward-compatible formatting for the pre-Phase-2 evidence shape."""
    lines = [f"GROUP: {group_key}"]
    sources: list[str] = []

    for item in items:
        subquery = item.get("subquery", "")
        result = item.get("result", {})
        if not isinstance(result, dict):
            continue

        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        source = metadata.get("source", "Unknown source")
        document = result.get("document", "")

        lines.append(
            f"Subquery:\n{subquery}\n\n"
            f"Source:\n{source}\n\n"
            f"Evidence:\n{document}"
        )

        if source not in sources:
            sources.append(source)

    return "\n".join(lines), sources



def _normalized_text(value: object) -> str:
    """Normalize text for conservative answer-consistency checks."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _token_set(value: object) -> set[str]:
    return set(_normalized_text(value).split())


def _contains_value(answer: str, value: object) -> bool:
    """Order-insensitive check that every token of ``value`` occurs in ``answer``.

    A verbatim-substring check is too brittle for multi-item values (an answer
    may legitimately re-list the same items in another order or grouping), but a
    token-set check still catches a changed or replaced value because a different
    number/word is simply absent.
    """
    value_tokens = _token_set(value)
    if not value_tokens:
        return True
    return value_tokens <= _token_set(answer)


def _resolved_facts(evidence: dict) -> list[dict]:
    """Return authoritative resolved facts from consolidated groups."""
    if not isinstance(evidence, dict):
        return []

    facts: list[dict] = []
    for group_value in evidence.values():
        if not isinstance(group_value, dict):
            continue
        if group_value.get("status") != "resolved":
            continue
        fact = group_value.get("resolved_fact")
        if isinstance(fact, dict) and str(fact.get("value", "")).strip():
            facts.append(fact)
    return facts


def _sole_resolved_fact(evidence: dict) -> dict | None:
    """The resolved fact, only when the question has exactly ONE group and it is resolved.

    The resolved-value guard is meaningful only for a single-requirement
    question. With several groups, a resolved fact answers just one part of the
    question, so requiring it to carry the whole answer is wrong.
    """
    if not isinstance(evidence, dict) or len(evidence) != 1:
        return None
    facts = _resolved_facts(evidence)
    return facts[0] if len(facts) == 1 else None


def _answer_preserves_resolved_facts(answer: str, resolved_facts: list[dict]) -> bool:
    """True when every resolved value is present in the answer (token-tolerant)."""
    if not resolved_facts:
        return True
    return all(_contains_value(answer, fact.get("value", "")) for fact in resolved_facts)


def _deterministic_resolved_answer(resolved_facts: list[dict]) -> str:
    """Produce a grounded fallback using only application-resolved facts."""
    statements = []
    for fact in resolved_facts:
        statement = str(fact.get("statement", "")).strip()
        if statement:
            statements.append(statement)

    if not statements:
        return _SERVICE_FALLBACK

    return " ".join(statements)


# ---------------------------------------------------------------------------
# Answer-obligation verification (generate -> verify -> append only)
# ---------------------------------------------------------------------------

_MAX_POOL_CHARS = 14000
_MAX_DOC_CHARS = 3500
_MAX_APPENDED_ITEMS = 6
_MAX_QUOTE_CHARS = 300


def _doc_key(item: dict) -> tuple[str, str] | None:
    nested = item.get("result", item) if isinstance(item, dict) else None
    if not isinstance(nested, dict):
        return None
    metadata = nested.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    source = str(metadata.get("source") or metadata.get("filename") or "Unknown source")
    document = str(nested.get("document", ""))
    if not document.strip():
        return None
    return source, document


def _verification_pool(evidence: dict) -> list[dict]:
    """Collect the evidence the verifier may quote from.

    Only groups the consolidator left `resolved` or `supported` contribute.
    `insufficient` stays insufficient and `contradictory` stays contradictory,
    so neither group may be used to add content. Candidates the entity firewall
    rejected are never included. Candidates that were retrieved but not cited by
    an extracted fact are included only when the group was explicitly entity
    scoped (the firewall actually ran); for unscoped groups only cited evidence
    is trusted.
    """
    pool: list[dict] = []
    seen: set[tuple[str, str]] = set()
    total = 0

    for group_key, group in evidence.items():
        if not isinstance(group, dict):
            continue
        if group.get("status") not in {"resolved", "supported"}:
            continue

        rejected = {
            key
            for key in (_doc_key(i) for i in group.get("rejected_candidates") or [])
            if key
        }
        candidates = list(group.get("evidence") or [])
        if group.get("entity_scope_status") == "explicit":
            candidates += list(group.get("raw_candidates") or [])

        for item in candidates:
            key = _doc_key(item)
            if not key or key in rejected or key in seen:
                continue
            source, document = key
            document = document[:_MAX_DOC_CHARS]
            if total + len(document) > _MAX_POOL_CHARS:
                continue
            seen.add(key)
            total += len(document)
            pool.append({
                "id": f"E{len(pool)}",
                "group": group_key,
                "source": source,
                "document": document,
            })

    return pool


def _clean_quote(text: str) -> str:
    cleaned = re.sub(r"\*+", "", text)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _conflicts_with_resolved(quote: str, resolved_facts: list[dict]) -> bool:
    """Reject a quote that restates an authoritative attribute with another value."""
    # Whole-word phrase match on normalized text, never a raw substring match:
    # a short attribute label must not fire inside unrelated words.
    padded_quote = f" {_normalized_text(quote)} "
    for fact in resolved_facts:
        attribute = _normalized_text(fact.get("attribute", ""))
        if len(attribute) < 3:
            continue
        if f" {attribute} " in padded_quote and not _contains_value(
            quote, fact.get("value", "")
        ):
            return True
    return False


def _content_tokens(value: object) -> set[str]:
    """Tokens of length > 2, so connectives do not dominate overlap checks."""
    return {tok for tok in _normalized_text(value).split() if len(tok) > 2}


def _coverage(quote: str, answer: str) -> float:
    """Fraction of the quote's content tokens that already occur in the answer."""
    tokens = _content_tokens(quote)
    if not tokens:
        return 1.0
    return len(tokens & _content_tokens(answer)) / len(tokens)


def _layout_blocks(document: str) -> list[str]:
    """Split a document into contiguous blocks (normalized text).

    A block is a run of consecutive non-empty lines. Blank lines and headings end
    a block. This is a purely structural notion of "the same list/paragraph" and
    carries no domain vocabulary.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in document.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append(current)
    return [_normalized_text(" ".join(block)) for block in blocks]


def _same_block(document: str, first: str, second: str) -> bool:
    first_norm, second_norm = _normalized_text(first), _normalized_text(second)
    if not first_norm or not second_norm:
        return False
    return any(
        first_norm in block and second_norm in block
        for block in _layout_blocks(document)
    )


def _build_verifier_prompt(query: str, answer: str, evidence: dict, pool: list[dict]) -> str:
    fact_lines: list[str] = []
    for group_key, group in evidence.items():
        if not isinstance(group, dict):
            continue
        status = group.get("status", "insufficient")
        fact_lines.append(f"GROUP: {group_key}\nSTATUS: {status}")
        fact = group.get("resolved_fact")
        if status == "resolved" and isinstance(fact, dict):
            fact_lines.append(
                f"AUTHORITATIVE RESOLVED FACT: {fact.get('statement', '')} "
                f"(attribute: {fact.get('attribute', '')}; value: {fact.get('value', '')})"
            )
    evidence_blocks = [
        f"[{p['id']}] Source: {p['source']}\n{p['document']}" for p in pool
    ]

    return f"""
You audit a drafted answer for COMPLETENESS against a fixed evidence pool.
You do not rewrite the answer and you never add knowledge of your own.

User question:
{query}

Drafted answer:
{answer}

Group statuses (application outputs; never contradict them):
{chr(10).join(fact_lines)}

Evidence pool (already restricted to the correct entities):
{chr(10).join(evidence_blocks)}

Step 1 - obligations. Using ONLY the question text, list the distinct pieces of
information it asks for. Give each a "kind":
- "single_value": it asks for one specific value or fact.
- "collection": it asks for a set of things (what something manages, oversees,
  uses, includes, consists of, is responsible for, etc.), where several separate
  items may each answer it.

Step 2 - status of each obligation against the drafted answer:
- "answered": the answer gives supported information for it (it may still be
  incomplete; completeness is judged item by item in Step 3).
- "declared_unavailable": the answer explicitly says it is unavailable/insufficient.
- "omitted": the answer says nothing about it.

Step 3 - enumerate items. For every obligation that is NOT "declared_unavailable",
list EACH distinct item in the evidence pool that directly answers it, one item
per entry, and decide for each item on its own whether the drafted answer already
conveys it ("conveyed_by_draft"). Do this even when the answer looks complete
or is a single sentence: judge every item separately, do not judge the whole
request at once.
An item counts only if ALL hold:
- its "quote" is contiguous text copied character for character from ONE
  evidence item, and "evidence_id" names that item;
- it is about the same entity, and within the same scope, role or time period
  that the question asks about;
- it itself answers what the question asks (for example, describes what the
  question asks about), as opposed to merely describing the entity's other
  attributes, relationships, measurements, results or history that the question
  did not ask for;
- it does not restate an authoritative resolved attribute with a different value.
Never list an item from another entity, scope or period. If nothing in the pool
answers an obligation, list no items for it.

Return ONLY JSON in exactly this form:
{{
  "obligations": [
    {{"obligation": "short phrase", "kind": "single_value|collection", "status": "answered|declared_unavailable|omitted"}}
  ],
  "items": [
    {{"obligation": "same phrase as above", "evidence_id": "E0", "quote": "exact text", "conveyed_by_draft": true}}
  ]
}}
"""


def _append_missing_obligations(
    query: str,
    answer: str,
    evidence: dict,
    audit: dict,
) -> str:
    """Append evidence-backed items the draft omitted. Never replaces anything.

    Item-level rules (all enforced in Python, never delegated to the model):
      * the quote must occur verbatim in an entity-scoped evidence item;
      * `declared_unavailable` obligations never receive items;
      * items for an `omitted` obligation are accepted on their own merits;
      * items for an already `answered` obligation are accepted only for
        collection requests AND only when they sit in the same contiguous layout
        block as an anchor, i.e. text the draft demonstrably conveys (or a resolved
        fact's value). This stops expansion into unrelated parts of a document;
      * no conflict with an authoritative attribute, no duplicates, hard cap.
    """
    pool = _verification_pool(evidence)
    if not pool:
        audit["verifier"] = "skipped_no_usable_evidence"
        return answer

    prompt = _build_verifier_prompt(query, answer, evidence, pool)
    data = safe_completion_json(
        prompt,
        max_tokens=1536,
        max_retries=2,
        fallback={},
        cache_namespace=ANSWER_PROMPT_VERSION,
    )
    if not isinstance(data, dict):
        audit["verifier"] = "invalid_response"
        return answer

    obligations = data.get("obligations")
    items = data.get("items")
    if not isinstance(obligations, list) or not isinstance(items, list):
        audit["verifier"] = "invalid_response"
        return answer

    info: dict[str, dict] = {}
    for entry in obligations:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("obligation", "")).strip().casefold()
        if name:
            info[name] = {
                "status": str(entry.get("status", "")).strip().casefold(),
                "kind": str(entry.get("kind", "")).strip().casefold(),
            }
    audit["obligations"] = {name: dict(value) for name, value in info.items()}

    pool_by_id = {p["id"]: p for p in pool}
    resolved = _resolved_facts(evidence)

    # Anchors: (evidence id, quote) pairs the draft demonstrably conveys.
    anchors: list[tuple[str, str]] = []
    for fact in resolved:
        value = str(fact.get("value", "")).strip()
        if value and _coverage(value, answer) >= 0.6:
            for p in pool:
                if _normalized_text(value) in _normalized_text(p["document"]):
                    anchors.append((p["id"], value))
    for item in items:
        if not isinstance(item, dict) or item.get("conveyed_by_draft") is not True:
            continue
        source = pool_by_id.get(str(item.get("evidence_id", "")).strip())
        quote = str(item.get("quote", "")).strip()
        if (
            source is not None
            and quote
            and _normalized_text(quote) in _normalized_text(source["document"])
            and _coverage(quote, answer) >= 0.6
        ):
            anchors.append((source["id"], quote))

    accepted: list[str] = []
    decisions: list[dict] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("conveyed_by_draft") is not False:
            continue  # conveyed items are anchors only; never appended

        obligation = str(item.get("obligation", "")).strip().casefold()
        quote = str(item.get("quote", "")).strip()
        source = pool_by_id.get(str(item.get("evidence_id", "")).strip())
        meta = info.get(obligation, {})

        reason = None
        if len(accepted) >= _MAX_APPENDED_ITEMS:
            reason = "cap_reached"
        elif not meta:
            reason = "unknown_obligation"
        elif meta["status"] == "declared_unavailable":
            reason = "obligation_declared_unavailable"
        elif meta["status"] not in {"answered", "omitted"}:
            reason = "unknown_obligation_status"
        elif not quote or len(quote) > _MAX_QUOTE_CHARS or source is None:
            reason = "malformed"
        elif _normalized_text(quote) not in _normalized_text(source["document"]):
            reason = "quote_not_in_evidence"
        elif _conflicts_with_resolved(quote, resolved):
            reason = "conflicts_with_resolved_attribute"
        else:
            cleaned = _clean_quote(quote)
            if not cleaned or _coverage(cleaned, answer) >= 0.8:
                reason = "already_in_answer"
            elif any(_coverage(cleaned, a) >= 0.8 for a in accepted):
                reason = "duplicate"
            elif meta["status"] == "answered":
                if meta["kind"] != "collection":
                    reason = "single_value_obligation_answered"
                elif not any(
                    anchor_id == source["id"]
                    and _same_block(source["document"], anchor_quote, quote)
                    for anchor_id, anchor_quote in anchors
                ):
                    reason = "outside_anchor_block"
            if reason is None:
                accepted.append(cleaned)

        decisions.append({
            "quote": quote[:100],
            "obligation": obligation,
            "decision": reason or "appended",
        })

    audit["appended"] = list(accepted)
    audit["item_decisions"] = decisions
    audit["rejected_additions"] = [d for d in decisions if d["decision"] != "appended"]
    audit["anchors"] = [quote[:100] for _, quote in anchors]
    audit["verifier"] = "ran"

    if not accepted:
        return answer

    bullets = "\n".join(f"- {text}" for text in accepted)
    return f"{answer.rstrip()}\n\nAdditional directly supported details:\n{bullets}"


def generate_answer(
    query: str,
    evidence: dict,
) -> dict:
    """
    Generate an answer from retrieved/consolidated evidence.

    Pipeline:
      1. generate from the consolidated evidence;
      2. (single-requirement questions only) make sure the authoritative
         resolved value is present, correcting with the FULL context;
      3. verify answer obligations and append only missing, evidence-backed,
         entity-scoped items. A draft is never replaced by the verifier.

    The function also accepts the pre-Phase-2 raw evidence shape.
    The returned dict carries an `audit` entry describing what each stage did.
    """

    if not evidence:
        return {
            "answer": "I could not find sufficient evidence to answer the question.",
            "sources": [],
            "audit": {"stage": "no_evidence"},
        }

    evidence_context: list[str] = []
    sources: list[str] = []
    consolidated_mode = all(
        _is_consolidated_group(value)
        for value in evidence.values()
    )

    for group_key, group_value in evidence.items():
        if consolidated_mode and isinstance(group_value, dict):
            formatted, group_sources = _format_consolidated_group(
                group_key,
                group_value,
            )
        else:
            formatted, group_sources = _format_raw_group(
                group_key,
                group_value if isinstance(group_value, list) else [],
            )

        evidence_context.append(formatted)

        for source in group_sources:
            if source not in sources:
                sources.append(source)

    context = "\n\n".join(evidence_context)

    if consolidated_mode:
        resolution_instructions = """
IMPORTANT EVIDENCE-RESOLUTION CONTRACT:
The evidence consolidation layer has already performed deterministic fact
resolution. Treat STATUS, RESOLVED FACT, and RESOLUTION BASIS as application
outputs, not suggestions to reconsider.

- If STATUS is `resolved`, the RESOLVED FACT is authoritative for the attribute
  it states. It is one required value, not necessarily the whole answer.
- If STATUS is `supported`, answer from the supported fact(s).
- If STATUS is `contradictory`, explicitly report the conflicting facts and do
  not choose one yourself.
- If STATUS is `insufficient`, state that the available evidence is
  insufficient; do not invent or infer a value.
- Do not independently re-resolve a `resolved` requirement using other evidence.
  Other evidence has already passed entity scoping and may be used for OTHER
  parts of the question, but it MUST NOT replace or override the RESOLVED FACT.
"""
    else:
        resolution_instructions = """
This evidence has not necessarily been passed through deterministic
consolidation. Do not invent facts, and do not silently resolve genuine
conflicts in raw evidence.
"""

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided evidence.
Evidence is grouped by independently retrievable requirement.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Every factual claim must be supported by the provided evidence.
4. Carefully synthesize evidence from different groups only when the user's
   question requires combining independently supported facts.
5. Keep different requirement groups distinct; do not use one group's fact to
   satisfy another group's requirement.
6. Be concise and directly answer the user's question.
7. Do not generate citations or source references.
8. Before answering, identify every distinct item the question asks for (each
   attribute, entity, period, condition or outcome). Answer all of them. When the
   question asks what something manages, oversees, uses or includes, state every
   directly responsive supported item for the role/scope the question names, and
   leave out other roles, other time periods and unrelated details. A resolved
   fact is one required value, not a reason to stop early. Do not add unrelated
   evidence merely to be exhaustive.
9. When a resolved group is accompanied by ADDITIONAL IN-SCOPE EVIDENCE, the
   RESOLVED FACT stays authoritative for its own attribute. The additional
   evidence may be used only to add directly supported, non-conflicting details
   that answer other parts of the question.
10. When the user asks for a calculation (for example a total, difference,
    change, average, percentage, comparison, or ranking), perform that calculation
    from the explicit numeric values in the provided evidence when the required
    inputs are available. State the resulting value clearly. Never invent a
    numeric component that is not present in the evidence.
11. If a requested total includes a component that has no stated numeric value,
    do not present a sum as the complete total. State the sum of the explicitly
    quantified components, label it as excluding the unquantified component by
    name, and say that component has no stated value.

{resolution_instructions}

User question:
{query}

Retrieved and consolidated evidence:
{context}

Now provide the final answer.
"""

    answer = safe_completion_text(
        prompt,
        max_tokens=512,
        fallback=_SERVICE_FALLBACK,
        cache_namespace=ANSWER_PROMPT_VERSION,
    )

    audit: dict = {"prompt_version": ANSWER_PROMPT_VERSION}
    authoritative_facts = _resolved_facts(evidence) if consolidated_mode else []

    # Provider failure: only an application-resolved fact may stand in.
    if answer.strip() == _SERVICE_FALLBACK:
        if authoritative_facts:
            audit["stage"] = "deterministic_fallback_after_provider_failure"
            return {
                "answer": _deterministic_resolved_answer(authoritative_facts),
                "sources": sources,
                "audit": audit,
            }
        audit["stage"] = "provider_failure"
        return {"answer": answer, "sources": sources, "audit": audit}

    # Resolved-value guard: single-requirement questions only, and it corrects
    # with the full context so other supported details are not lost.
    sole = _sole_resolved_fact(evidence) if consolidated_mode else None
    if sole is not None and not _answer_preserves_resolved_facts(answer, [sole]):
        correction_prompt = f"""{prompt}

CORRECTION REQUIRED:
The previous draft did not state the RESOLVED FACT value "{sole.get('value', '')}".
Write the final answer again so that it states the RESOLVED FACT exactly and keeps
every other directly supported detail the question asks for. Do not mention this
correction.

Previous draft:
{answer}
"""
        corrected = safe_completion_text(
            correction_prompt,
            max_tokens=512,
            max_retries=2,
            fallback="",
            cache_namespace=ANSWER_PROMPT_VERSION,
        )
        if corrected.strip() and _answer_preserves_resolved_facts(corrected, [sole]):
            answer = corrected.strip()
            audit["resolved_guard"] = "corrected_with_full_context"
        else:
            answer = _deterministic_resolved_answer([sole])
            audit["resolved_guard"] = "deterministic_statement"

    if consolidated_mode:
        try:
            answer = _append_missing_obligations(query, answer, evidence, audit)
        except Exception as error:  # verification must never break generation
            audit["verifier"] = f"error:{type(error).__name__}"

    return {
        "answer": answer,
        "sources": sources,
        "audit": audit,
    }