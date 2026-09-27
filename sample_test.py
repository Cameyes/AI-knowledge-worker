# """
# Synthetic, domain-agnostic tests for verification_context.py.

# These tests use generic documents (a product spec, a project
# report, a policy) — deliberately NOT employee/HR data — to prove
# the fixes generalize rather than being tuned to one test dataset.

# The cross-encoder is stubbed with a deterministic scoring function
# so these tests are fast, reproducible, and don't require the real
# model weights. The stub is intentionally biased AGAINST tabular/
# terse text and FOR fluent prose, to reproduce the real failure mode
# observed with the actual cross-encoder.

# Run with: python test_compaction_generic.py
# """

# from unittest.mock import patch

# import app.rag.verification_context as vc


# # ---------------------------------------------------------------
# # A controllable, deterministic stand-in for the real cross-encoder.
# # ---------------------------------------------------------------

# def _term_overlap_score(query: str, text: str) -> float:
#     """
#     Base relevance signal: fraction of query terms present in text.
#     """
#     import re

#     query_terms = set(re.findall(r"[a-z0-9']+", query.lower()))
#     text_lower = text.lower()

#     if not query_terms:
#         return 0.0

#     hits = sum(1 for term in query_terms if term in text_lower)

#     return hits / len(query_terms)


# def _biased_predict(pairs, batch_size=32, show_progress_bar=False):
#     """
#     Stub for reranker.predict(). Scores term overlap, but applies
#     a penalty to short, tabular-looking lines (few words, contains
#     a colon or digit-heavy content) to reproduce the real observed
#     bias of prose-trained cross-encoders against terse key-value
#     text — even when that text contains the exact requested fact.
#     """
#     scores = []

#     for query, text in pairs:
#         score = _term_overlap_score(query, text)

#         word_count = len(text.split())
#         looks_tabular = (
#             word_count <= 8
#             and (":" in text or any(c.isdigit() for c in text))
#         )

#         if looks_tabular:
#             score *= 0.5

#         scores.append(score)

#     return scores


# # ---------------------------------------------------------------
# # Test 1: heading context must persist across ALL sibling blocks,
# # not just the first one after the heading.
# # ---------------------------------------------------------------

# def test_heading_persists_across_siblings():

#     document = """Quarterly Results

# Q1 Revenue: $2.1M
# Grew steadily on strong renewal rates.

# Q2 Revenue: $2.4M
# Driven by a large enterprise contract signed in May.

# Q3 Revenue: $1.9M
# Declined due to a temporary supply disruption.
# """

#     units = vc._split_units(document)

#     # There should be 3 substantive units (one per quarter). All
#     # three must carry "Quarterly Results" as scoring context —
#     # not just the first one immediately after the heading.
#     substantive = [u for u in units if "Revenue" in u["text"]]

#     assert len(substantive) == 3, (
#         f"Expected 3 quarterly units, got {len(substantive)}"
#     )

#     for unit in substantive:
#         assert unit["context"] == "Quarterly Results", (
#             f"Unit missing persisted heading context: {unit}"
#         )

#     # The heading itself should only be merged verbatim into the
#     # FIRST unit's text (its immediate neighbor) — later siblings
#     # keep their own original text unmodified, with the heading
#     # available only as scoring context, not duplicated into text.
#     assert "Quarterly Results" in substantive[0]["text"]
#     assert "Quarterly Results" not in substantive[1]["text"]
#     assert "Quarterly Results" not in substantive[2]["text"]

#     print("PASS: test_heading_persists_across_siblings")


# # ---------------------------------------------------------------
# # Test 2: context is used for scoring but never appears in the
# # reconstructed output (no rewriting).
# # ---------------------------------------------------------------

# def test_context_not_duplicated_into_nonadjacent_siblings():
#     """
#     A fragment directly preceding a block is correctly merged into
#     that block's own text (expected — it preserves adjacent
#     label/fact pairs verbatim). But a LATER sibling that only
#     inherits the heading as persisted scoring `context` must NOT
#     have that heading duplicated into its own returned text.
#     """

#     document = """Quarterly Results

# Q1 Revenue: $2.1M
# Grew steadily on strong renewal rates.

# Q2 Revenue: $2.4M
# Driven by a large enterprise contract signed in May.
# """

#     with patch.object(vc.reranker, "predict", _biased_predict):
#         units = vc._split_units(document)
#         scored = vc._score_units(
#             "What was Q2 revenue?",
#             units,
#         )
#         # Force selection of the Q2 (non-adjacent-to-heading) unit
#         # specifically, regardless of how the stub scores it.
#         q2_only = [
#             item for item in scored
#             if "Q2 Revenue" in item[2]["text"]
#         ]

#         compacted = vc._compact_scored_units(
#             q2_only,
#             max_units=1,
#             max_chars=1000,
#         )

#     assert "Quarterly Results" not in compacted, (
#         "Heading was duplicated into a non-adjacent sibling's output"
#     )
#     assert "Q2 Revenue: $2.4M" in compacted

#     print("PASS: test_context_not_duplicated_into_nonadjacent_siblings")


# # ---------------------------------------------------------------
# # Test 3: top-1 selection loses a terse fact-bearing unit to a
# # fluent but irrelevant prose unit; top-k recovers it.
# # ---------------------------------------------------------------

# def test_topk_recovers_terse_unit_lost_by_topk1():

#     document = """The operations team performed extensive downtime \
# analysis this month, reviewing every downtime incident from the \
# previous quarter in detail.

# Engineers discussed downtime trends across multiple environments \
# during the retrospective meeting held last week.

# Facilities scheduled a routine snack budget review for next quarter.

# Downtime: 45 minutes
# """

#     query = "How long was the downtime?"

#     with patch.object(vc.reranker, "predict", _biased_predict):

#         units = vc._split_units(document)
#         scored = vc._score_units(query, units)

#         top1 = vc._compact_scored_units(scored, max_units=1, max_chars=1000)
#         top3 = vc._compact_scored_units(scored, max_units=3, max_chars=1000)

#     # With this stub's penalty against terse, tabular-looking text,
#     # top-1 is expected to select a fluent-but-nonspecific paragraph
#     # over the short line containing the actual figure.
#     assert "45 minutes" not in top1, (
#         "Test setup assumption failed: top-1 unexpectedly recovered "
#         "the fact on its own — adjust the stub/fixture."
#     )

#     # top-3 should recover the fact...
#     assert "Downtime: 45 minutes" in top3, (
#         "top-k retention failed to recover the fact-bearing unit"
#     )
#     # ...while still excluding the clearly irrelevant 4th paragraph,
#     # showing top-k isn't just "return everything".
#     assert "snack budget" not in top3, (
#         "top-k included a clearly irrelevant unit — k is too loose "
#         "or scoring/exclusion logic is broken"
#     )

#     print("PASS: test_topk_recovers_terse_unit_lost_by_topk1")


# # ---------------------------------------------------------------
# # Test 4: no-rewriting invariant — every unit of text returned by
# # compact_evidence must be a verbatim substring of the original
# # document, for an arbitrary synthetic document.
# # ---------------------------------------------------------------

# def test_output_is_verbatim_substring_of_original():

#     document = """Section A

# Item one has some descriptive text here.

# Item two has some other descriptive text.

# Section B

# Final note about the process.
# """

#     with patch.object(vc.reranker, "predict", _biased_predict):
#         compacted = vc.compact_evidence(
#             "descriptive text",
#             document,
#             max_units=2,
#         )

#     for fragment in compacted.split("\n\n"):
#         assert fragment in document, (
#             f"Output fragment is not a verbatim substring of the "
#             f"original document: {fragment!r}"
#         )

#     print("PASS: test_output_is_verbatim_substring_of_original")


# # ---------------------------------------------------------------
# # Test 5: source metadata must NOT be part of the scoring input
# # (regression guard against reintroducing document-level bias).
# # ---------------------------------------------------------------

# def test_source_not_used_in_scoring_pairs():

#     captured_pairs = []

#     def _capturing_predict(pairs, batch_size=32, show_progress_bar=False):
#         captured_pairs.extend(pairs)
#         return _biased_predict(pairs)

#     results = [
#         {
#             "document": "Policy Overview\n\nAll requests must be logged.",
#             "metadata": {"source": "policy_internal_v3_final.md"},
#         }
#     ]

#     with patch.object(vc.reranker, "predict", _capturing_predict):
#         vc.prepare_verification_context("logging requirement", results)

#     for _, scoring_text in captured_pairs:
#         assert "policy_internal_v3_final" not in scoring_text, (
#             "Source filename leaked into cross-encoder scoring input"
#         )

#     print("PASS: test_source_not_used_in_scoring_pairs")


# def main():

#     tests = [
#         test_heading_persists_across_siblings,
#         test_context_not_duplicated_into_nonadjacent_siblings,
#         test_topk_recovers_terse_unit_lost_by_topk1,
#         test_output_is_verbatim_substring_of_original,
#         test_source_not_used_in_scoring_pairs,
#     ]

#     failures = 0

#     for test in tests:
#         try:
#             test()
#         except AssertionError as e:
#             failures += 1
#             print(f"FAIL: {test.__name__} — {e}")

#     print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")


# if __name__ == "__main__":
#     main()
#=================================================================================#

# """
# Diagnostic: is David Kim's compensation chunk missing at the
# vector-retrieval stage, or is it retrieved but reranked out?

# This is a debugging tool, not a pipeline fix — it deliberately
# bypasses multi_evidence_retrieve to inspect each stage separately.
# """

# from app.rag.retriever import retrieve
# from app.rag.reranker import rerank


# SUBQUERIES = [
#     "What was David Kim's 2022 performance rating?",
#     "What was David Kim's bonus in 2022?",
# ]


# def inspect(subquery: str, retrieval_top_k: int = 20):

#     print("\n" + "=" * 80)
#     print(f"SUBQUERY: {subquery}")
#     print("=" * 80)

#     # ---------------------------------------------------------
#     # Stage 1: raw vector retrieval, BEFORE reranking, with a
#     # generously large top_k so we can see if David Kim's chunk
#     # is present anywhere in the embedding-similarity ranking,
#     # even if it wouldn't normally survive down to top_k=10.
#     # ---------------------------------------------------------

#     raw_results = retrieve(subquery, top_k=retrieval_top_k)

#     print(f"\n--- RAW RETRIEVAL (top {retrieval_top_k}, pre-rerank) ---")

#     for i, result in enumerate(raw_results):
#         source = result.get("metadata", {}).get("source", "?")
#         is_david = "David Kim" in source
#         marker = " <-- DAVID KIM" if is_david else ""
#         distance = result.get("distance", result.get("score", "?"))
#         print(f"{i:2d}. {source}  (score/distance: {distance}){marker}")

#     david_in_raw = [
#         r for r in raw_results
#         if "David Kim" in r.get("metadata", {}).get("source", "")
#     ]

#     print(f"\nDavid Kim chunks in raw retrieval: {len(david_in_raw)}")
#     for r in david_in_raw:
#         preview = r.get("document", "")[:150].replace("\n", " ")
#         print(f"  - {preview}...")

#     # ---------------------------------------------------------
#     # Stage 2: reranked results, at the top_k your pipeline
#     # actually uses (8), to see if David Kim survives reranking.
#     # ---------------------------------------------------------

#     reranked = rerank(subquery, raw_results, top_k=8)

#     print("\n--- RERANKED (top 8, what the pipeline actually uses) ---")

#     for i, result in enumerate(reranked):
#         source = result.get("metadata", {}).get("source", "?")
#         is_david = "David Kim" in source
#         marker = " <-- DAVID KIM" if is_david else ""
#         print(f"{i:2d}. {source}{marker}")

#     david_in_reranked = [
#         r for r in reranked
#         if "David Kim" in r.get("metadata", {}).get("source", "")
#     ]

#     # ---------------------------------------------------------
#     # Diagnosis
#     # ---------------------------------------------------------

#     print("\n--- DIAGNOSIS ---")

#     if not david_in_raw:
#         print(
#             "David Kim's chunk(s) are ABSENT even from raw vector "
#             "retrieval at top_k={0}. This points to an embedding/"
#             "chunking-level problem: either the chunk doesn't exist "
#             "as expected, or its embedding doesn't score well against "
#             "this query. Increasing retrieval_top_k won't fix this — "
#             "check how David Kim.md was chunked.".format(retrieval_top_k)
#         )
#     elif not david_in_reranked:
#         print(
#             "David Kim's chunk(s) ARE present in raw retrieval but "
#             "did NOT survive reranking into the top 8. This points to "
#             "a reranking-precision problem, not a recall problem — "
#             "consider raising rerank_top_k, or inspect why the "
#             "cross-encoder scores this chunk low for this query."
#         )
#     else:
#         contains_compensation = any(
#             "Compensation History" in r.get("document", "")
#             for r in david_in_reranked
#         )
#         if contains_compensation:
#             print(
#                 "David Kim's chunk(s) with Compensation History DID "
#                 "survive to the reranked top 8. If the pipeline still "
#                 "fails, the issue is downstream (compaction/"
#                 "verification) after all — re-check that stage."
#             )
#         else:
#             print(
#                 "David Kim's chunk(s) survived reranking, but NONE of "
#                 "them contain 'Compensation History' — retrieval found "
#                 "*a* David Kim chunk, just not the one with the fact "
#                 "this subquery needs. Check how David Kim.md was "
#                 "chunked: is Compensation History in its own separate "
#                 "chunk, and if so, why doesn't THAT chunk rank in the "
#                 "top 8 for this query?"
#             )


# def main():
#     for subquery in SUBQUERIES:
#         inspect(subquery)


# if __name__ == "__main__":
#     main()

#==============================================================
# """
# Generic chunk-size audit.

# Reports, per source document, how many chunks it produced and
# their character-length distribution. Flags any source whose
# chunks are unusually large relative to the rest of the corpus —
# a strong signal that LLM-based semantic chunking fell back to a
# single oversized chunk for that document (see chunker.py's
# find_boundaries fallback), which in turn risks silent truncation
# during cross-encoder reranking (see reranker.py).

# No document text is printed — only lengths and counts — so this
# is safe to run and share output from even for sensitive corpora.
# """

# import statistics

# from app.rag.vectorstore import ChromaVectorStore


# # A rough, conservative proxy for "likely to hit the cross-encoder's
# # truncation limit". ms-marco-MiniLM-L-6-v2 truncates at a default
# # max sequence length (commonly 512 tokens); English text averages
# # roughly 4-5 characters per token, so ~2000-2500 characters is a
# # reasonable warning threshold without needing to load a tokenizer
# # just for this audit.
# TRUNCATION_RISK_CHARS = 2000


# def audit():

#     vectorstore = ChromaVectorStore()

#     all_data = vectorstore.collection.get(
#         include=["documents", "metadatas"]
#     )

#     documents = all_data["documents"]
#     metadatas = all_data["metadatas"]

#     by_source: dict[str, list[int]] = {}

#     for document, metadata in zip(documents, metadatas):
#         source = metadata.get("source", "unknown")
#         by_source.setdefault(source, []).append(len(document))

#     print(f"Total chunks in collection: {len(documents)}")
#     print(f"Total distinct sources: {len(by_source)}")

#     all_lengths = [length for lengths in by_source.values() for length in lengths]

#     if all_lengths:
#         corpus_mean = statistics.mean(all_lengths)
#         corpus_median = statistics.median(all_lengths)
#         print(
#             f"\nCorpus-wide chunk length — mean: {corpus_mean:.0f} "
#             f"chars, median: {corpus_median:.0f} chars"
#         )

#     print("\n" + "=" * 80)
#     print(
#         f"{'Source':<45} {'#chunks':>8} {'avg len':>10} "
#         f"{'max len':>10} {'flag':>6}"
#     )
#     print("=" * 80)

#     rows = []

#     for source, lengths in by_source.items():
#         avg_len = sum(lengths) / len(lengths)
#         max_len = max(lengths)
#         flagged = max_len > TRUNCATION_RISK_CHARS or len(lengths) == 1
#         rows.append((source, len(lengths), avg_len, max_len, flagged))

#     # Sort so flagged / largest-max-length sources appear first —
#     # these are the ones worth investigating.
#     rows.sort(key=lambda r: (-r[4], -r[3]))

#     for source, count, avg_len, max_len, flagged in rows:
#         flag_str = "⚠️ " if flagged else ""
#         print(
#             f"{source:<45} {count:>8} {avg_len:>10.0f} "
#             f"{max_len:>10} {flag_str:>6}"
#         )

#     flagged_sources = [r for r in rows if r[4]]

#     print("\n" + "=" * 80)
#     if flagged_sources:
#         print(
#             f"{len(flagged_sources)} source(s) flagged: either a "
#             f"single chunk (possible chunking fallback to the whole "
#             f"document) or a chunk exceeding "
#             f"{TRUNCATION_RISK_CHARS} characters (truncation risk "
#             f"during reranking)."
#         )
#         print(
#             "\nIf a source you'd expect to retrieve correctly is "
#             "flagged here, that's a strong candidate explanation for "
#             "retrieval/reranking misses on that document — the fix "
#             "would be re-chunking it (e.g. re-running chunker.py for "
#             "that file, or lowering MAX_BATCH_CHARS) rather than "
#             "anything in retriever.py or reranker.py."
#         )
#     else:
#         print(
#             "No sources flagged — chunk sizes look consistent across "
#             "the corpus. If retrieval is still missing expected "
#             "content, the issue is more likely embedding quality or "
#             "reranking behavior itself, not chunk granularity."
#         )


# if __name__ == "__main__":
#     audit()

# from app.rag.vectorstore import ChromaVectorStore


# def main():
#     vectorstore = ChromaVectorStore()

#     results = vectorstore.collection.get(
#         include=["documents", "metadatas"]
#     )

#     print("\n" + "=" * 80)
#     print("DAVID KIM CHUNKS IN CHROMA")
#     print("=" * 80)

#     count = 0

#     for document, metadata in zip(
#         results["documents"],
#         results["metadatas"],
#     ):
#         source = metadata.get("source", "")

#         if "David Kim" not in source:
#             continue

#         count += 1

#         print(f"\n--- CHUNK {count} ---")
#         print(f"Source: {source}")
#         print(document)

#     print("\n" + "=" * 80)
#     print(f"TOTAL DAVID KIM CHUNKS: {count}")
#     print("=" * 80)


# if __name__ == "__main__":
#     main()

#========================================================

# """
# Diagnostic: is David Kim's compensation chunk missing at the
# vector-retrieval stage, or is it retrieved but reranked out — and
# if reranked out, by how much, and is the reranker itself sane?

# This is a debugging tool, not a pipeline fix — it deliberately
# bypasses multi_evidence_retrieve to inspect each stage separately.
# """

# from app.rag.retriever import retrieve
# from app.rag.reranker import rerank, reranker


# SUBQUERIES = [
#     "What was David Kim's 2022 performance rating?",
#     "What was David Kim's bonus in 2022?",
# ]


# def sanity_check_reranker() -> None:
#     """
#     Runs the model card's own minimal example. If this doesn't
#     clearly rank the Mars passage highest, the problem is the
#     environment/model loading, not your pipeline or corpus —
#     stop debugging the pipeline and check package versions instead.
#     """

#     print("\n" + "=" * 80)
#     print("RERANKER SANITY CHECK (official minimal example)")
#     print("=" * 80)

#     query = "Which planet is known as the Red Planet?"
#     passages = [
#         "Venus is often called Earth's twin because of its similar "
#         "size and proximity.",
#         "Mars, known for its reddish appearance, is often referred "
#         "to as the Red Planet.",
#         "Jupiter, the largest planet in our solar system, has a "
#         "prominent red spot.",
#         "Saturn, famous for its rings, is sometimes mistaken for "
#         "the Red Planet.",
#     ]

#     scores = reranker.predict([(query, p) for p in passages])

#     ranked = sorted(
#         zip(scores, passages),
#         key=lambda x: x[0],
#         reverse=True,
#     )

#     for score, passage in ranked:
#         marker = " <-- EXPECTED WINNER (Mars)" if "Mars" in passage else ""
#         print(f"{float(score):8.4f}  {passage[:60]}...{marker}")

#     top_passage = ranked[0][1]

#     if "Mars" in top_passage:
#         print(
#             "\nPASS: Mars passage ranked highest, as expected. The "
#             "model and environment are working correctly — the issue "
#             "is specific to the pipeline or corpus, not a broken "
#             "install."
#         )
#     else:
#         print(
#             "\nFAIL: Mars passage did NOT rank highest on this "
#             "textbook example. This points to an environment/version "
#             "problem (outdated transformers/sentence-transformers, "
#             "or a bad model load) rather than anything specific to "
#             "your pipeline. Check `pip show sentence-transformers "
#             "transformers` versions before debugging further."
#         )


# def inspect(subquery: str, retrieval_top_k: int = 20):

#     print("\n" + "=" * 80)
#     print(f"SUBQUERY: {subquery}")
#     print("=" * 80)

#     # ---------------------------------------------------------
#     # Stage 1: raw vector retrieval, BEFORE reranking, with a
#     # generously large top_k so we can see if David Kim's chunk
#     # is present anywhere in the embedding-similarity ranking,
#     # even if it wouldn't normally survive down to top_k=10.
#     # ---------------------------------------------------------

#     raw_results = retrieve(subquery, top_k=retrieval_top_k)

#     print(f"\n--- RAW RETRIEVAL (top {retrieval_top_k}, pre-rerank) ---")

#     for i, result in enumerate(raw_results):
#         source = result.get("metadata", {}).get("source", "?")
#         is_david = "David Kim" in source
#         marker = " <-- DAVID KIM" if is_david else ""
#         distance = result.get("distance", result.get("score", "?"))
#         print(f"{i:2d}. {source}  (score/distance: {distance}){marker}")

#     david_in_raw = [
#         r for r in raw_results
#         if "David Kim" in r.get("metadata", {}).get("source", "")
#     ]

#     print(f"\nDavid Kim chunks in raw retrieval: {len(david_in_raw)}")
#     for r in david_in_raw:
#         preview = r.get("document", "")[:150].replace("\n", " ")
#         print(f"  - {preview}...")

#     # ---------------------------------------------------------
#     # Stage 2: rerank ALL raw results (not just top 8), so we can
#     # see David Kim's actual score and rank even if it doesn't
#     # make the pipeline's top_k=8 cutoff — "lost by a little" and
#     # "lost by a lot" point to very different problems.
#     # ---------------------------------------------------------

#     reranked_full = rerank(
#         subquery,
#         raw_results,
#         top_k=len(raw_results),
#     )

#     print(
#         f"\n--- FULL RERANKED ORDER (all {len(raw_results)}, "
#         f"with scores) ---"
#     )

#     for i, result in enumerate(reranked_full):
#         source = result.get("metadata", {}).get("source", "?")
#         score = result.get("rerank_score", "?")
#         is_david = "David Kim" in source
#         marker = " <-- DAVID KIM" if is_david else ""
#         cutoff_marker = "  [pipeline cutoff: top 8]" if i == 8 else ""
#         print(
#             f"{i:2d}. {source:<45} score={score:>10.4f}{marker}"
#             if isinstance(score, float)
#             else f"{i:2d}. {source:<45} score={score}{marker}"
#         )
#         if cutoff_marker:
#             print("    " + "-" * 60 + cutoff_marker)

#     reranked = reranked_full[:8]

#     david_in_reranked = [
#         r for r in reranked
#         if "David Kim" in r.get("metadata", {}).get("source", "")
#     ]

#     david_ranks = [
#         (i, r.get("rerank_score"))
#         for i, r in enumerate(reranked_full)
#         if "David Kim" in r.get("metadata", {}).get("source", "")
#     ]

#     if david_ranks and not david_in_reranked:
#         print(
#             f"\nDavid Kim's chunk(s) fell just outside the top 8 — "
#             f"actual rank(s)/score(s): {david_ranks}. Compare these "
#             f"scores to the score at rank 7 (the last item that made "
#             f"the cut) to see how close a call this was."
#         )

#     # ---------------------------------------------------------
#     # Diagnosis
#     # ---------------------------------------------------------

#     print("\n--- DIAGNOSIS ---")

#     if not david_in_raw:
#         print(
#             "David Kim's chunk(s) are ABSENT even from raw vector "
#             "retrieval at top_k={0}. This points to an embedding/"
#             "chunking-level problem: either the chunk doesn't exist "
#             "as expected, or its embedding doesn't score well against "
#             "this query. Increasing retrieval_top_k won't fix this — "
#             "check how David Kim.md was chunked.".format(retrieval_top_k)
#         )
#     elif not david_in_reranked:
#         print(
#             "David Kim's chunk(s) ARE present in raw retrieval but "
#             "did NOT survive reranking into the top 8. See the score "
#             "comparison above: if David Kim's score is close to the "
#             "rank-7/8 cutoff, this may just be a genuinely close call "
#             "for this reranker. If it's far below (or negative/odd "
#             "looking) while raw retrieval had it near rank 0, that's "
#             "a stronger signal of a reranker-side problem — run "
#             "sanity_check_reranker() if you haven't already."
#         )
#     else:
#         contains_compensation = any(
#             "Compensation History" in r.get("document", "")
#             for r in david_in_reranked
#         )
#         if contains_compensation:
#             print(
#                 "David Kim's chunk(s) with Compensation History DID "
#                 "survive to the reranked top 8. If the pipeline still "
#                 "fails, the issue is downstream (compaction/"
#                 "verification) after all — re-check that stage."
#             )
#         else:
#             print(
#                 "David Kim's chunk(s) survived reranking, but NONE of "
#                 "them contain 'Compensation History' — retrieval found "
#                 "*a* David Kim chunk, just not the one with the fact "
#                 "this subquery needs. Check how David Kim.md was "
#                 "chunked: is Compensation History in its own separate "
#                 "chunk, and if so, why doesn't THAT chunk rank in the "
#                 "top 8 for this query?"
#             )


# def main():
#     sanity_check_reranker()

#     for subquery in SUBQUERIES:
#         inspect(subquery)


# if __name__ == "__main__":
#     main()

#=====================================================

from app.rag.retriever import retrieve_by_source

results = retrieve_by_source("Daniel Park.md")

print(len(results))

for result in results:
    print(result["metadata"])