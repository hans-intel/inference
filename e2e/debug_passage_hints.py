import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

DEFAULT_DATASET = Path("data/frames_dataset.tsv")
DEFAULT_PASSAGES = Path("passages/doc_html_len2048_overlap32_word.json")
DEFAULT_OUTPUT = Path("debug_passage_hints_len2048.json")
DEFAULT_JUDGE_URL = "http://127.0.0.1:8123/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_CONTEXT_LIMIT = 40960


def _load_passages(path: Path) -> Dict[str, List[Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for item in raw:
        url = item.get("original_url")
        if not url:
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(url, []).append(
            {
                "index": index,
                "passage": item.get("passage", ""),
                "base_filename": item.get("base_filename"),
            }
        )
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry["index"])
    return grouped


def _load_dataset(path: Path) -> List[Tuple[int, str, str, List[str]]]:
    frame = pd.read_csv(path, sep="\t")
    rows: List[Tuple[int, str, str, List[str]]] = []
    for idx, (_, row) in enumerate(frame.iterrows(), start=1):
        question = str(row.get("Prompt", "")).strip()
        answer = str(row.get("Answer", "")).strip()
        links: List[str] = []
        seen = set()
        for col in frame.columns:
            if not col.startswith("wikipedia_link_"):
                continue
            value = row.get(col)
            if isinstance(value, str):
                url = value.strip()
                if url and url not in seen:
                    links.append(url)
                    seen.add(url)
        rows.append((idx, question, answer, links))
    return rows


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 0.5
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
        try:
            return float(lowered) >= 0.5
        except ValueError:
            return False
    return False


def _extract_json(content: str) -> Optional[Dict[str, object]]:
    if not content:
        return None
    stripped = content.strip()
    candidates: List[str] = []
    if stripped.startswith("```"):
        remainder = stripped.split("```", 1)[1].strip()
        if remainder.lower().startswith("json"):
            remainder = remainder[4:].strip()
        closing = remainder.find("```")
        if closing != -1:
            remainder = remainder[:closing]
        candidates.append(remainder.strip())
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    candidates.append(stripped)
    for candidate in candidates:
        sample = candidate.strip()
        if not sample:
            continue
        try:
            parsed = json.loads(sample)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def build_context_snippets(
    documents: Sequence[Dict[str, object]],
    max_entries: int,
    char_limit: int,
) -> Optional[str]:
    snippets: List[str] = []
    for doc in documents:
        url = doc.get("url", "")
        passages = doc.get("relevant_passages", [])
        for passage in passages[-max_entries:]:
            excerpt = _truncate(passage.get("content", ""), char_limit)
            comment = passage.get("comment", "")
            snippets.append(
                f"- {url} | idx {passage.get('index')} | iter {passage.get('iteration')}\n"
                f"  Comment: {comment}\n  Excerpt: {excerpt}"
            )
    if not snippets:
        return None
    return "\n\n".join(snippets[-max_entries:])


def _short_url_name(url: str) -> str:
    """Extract a short name from Wikipedia URL."""
    if "wikipedia.org/wiki/" in url:
        return url.split("/wiki/")[-1].replace("_", " ").replace("%C3%AB", "e")[:50]
    return url[:50]


def plan_link_visit_order(
    session: requests.Session,
    service_url: str,
    model: str,
    question: str,
    answer: str,
    links: Sequence[str],
    iteration: int,
    documents: Sequence[Dict[str, object]],
    max_output_tokens: int,
) -> Tuple[List[str], str]:
    """Use LLM to strategically order wiki links based on the question."""
    # Create short names for readability
    link_map = {f"Link{i+1}({_short_url_name(url)})": url for i, url in enumerate(links)}
    
    if iteration == 1:
        prompt = f"""You are a strategic retrieval planner.
Given a multi-hop question and a list of Wikipedia URLs, decide the optimal order to visit them.
Explain your reasoning concisely: which link provides the foundation, which builds on it, etc.
Use the short link names (Link1, Link2, etc.) in your reasoning.

Question: {question}
Expected Answer: {answer}

Available Wikipedia URLs:
{chr(10).join(f"{name}: {url}" for name, url in link_map.items())}

Respond with JSON: {{"ordered_urls": [list of full URLs in visit order], "reasoning": "concise step-by-step explanation using Link1, Link2, etc."}}"""
    else:
        progress = []
        for doc in documents:
            rel = len(doc.get("relevant_passages", []))
            rem = len(doc.get("remaining_indices", []))
            short = _short_url_name(doc['url'])
            progress.append(f"{short}: {rel} relevant, {rem} remaining")
        progress_str = "\n".join(progress)
        prompt = f"""You are a strategic retrieval planner.
Based on progress so far, decide the order to revisit Wikipedia URLs.
Use short link names in your reasoning.

Question: {question}
Expected Answer: {answer}
Iteration: {iteration}

Progress so far:
{progress_str}

Respond with JSON: {{"ordered_urls": [list of full URLs in visit order], "reasoning": "concise explanation using short names"}}"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You plan optimal search strategies for multi-hop questions."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }
    try:
        response = session.post(service_url, json=payload, timeout=120)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"].strip()
        parsed = _extract_json(message)
        if parsed and "ordered_urls" in parsed:
            ordered = parsed["ordered_urls"]
            reasoning = parsed.get("reasoning", "")
            # Validate URLs are in the original list
            valid_ordered = [url for url in ordered if url in links]
            # Add any missing URLs at the end
            for url in links:
                if url not in valid_ordered:
                    valid_ordered.append(url)
            return valid_ordered, reasoning
    except Exception:
        pass
    # Fallback: default order
    return list(links), "Default order (planner failed)"


def format_evidence(documents: Sequence[Dict[str, object]], char_limit: int) -> str:
    lines: List[str] = []
    for doc in documents:
        url = doc.get("url", "")
        for passage in doc.get("relevant_passages", []):
            excerpt = _truncate(passage.get("content", ""), char_limit)
            comment = passage.get("comment", "")
            lines.append(
                f"URL: {url}\nIndex: {passage.get('index')} | Iteration: {passage.get('iteration')}\n"
                f"Comment: {comment}\nExcerpt: {excerpt}"
            )
    return "\n\n".join(lines)


def call_passage_judge(
    session: requests.Session,
    service_url: str,
    model: str,
    question: str,
    answer: str,
    passage_text: str,
    iteration: int,
    context_text: Optional[str],
    prior_iteration_comments: Sequence[str],
    prior_passage_comment: Optional[str],
    max_output_tokens: int,
) -> Tuple[bool, str, str]:
    prompt_parts: List[str] = [
        "You are a strict retrieval judge.",
        "Mark contains_hint=true ONLY if this passage provides a NEW concrete fact directly needed to answer the question.",
        "A passage is relevant only if removing it would make the answer impossible to construct.",
        "If similar information was already found in prior passages (shown in context), mark contains_hint=false.",
        "Background context, lists without specific needed data, or vaguely related content should be marked contains_hint=false.",
        "For lists/tables: only mark relevant if this passage contains the SPECIFIC entry needed (not just another row of the same list).",
        "Respond strictly with JSON: {\"contains_hint\": boolean, \"comment\": string}.",
        f"\nQuestion: {question}",
        f"\nExpected Answer: {answer}",
        f"\nIteration: {iteration}",
    ]
    if prior_iteration_comments:
        bullet = "\n".join(f"- {comment}" for comment in prior_iteration_comments if comment.strip())
        if bullet:
            prompt_parts.append("\nEarlier iteration rationales (avoid repeating):\n" + bullet)
    if prior_passage_comment:
        prompt_parts.append("\nEarlier judgement for this passage:\n- " + prior_passage_comment)
    if context_text:
        prompt_parts.append("\nPreviously confirmed relevant passages:\n" + context_text)
    prompt_parts.append("\nPassage:\n" + passage_text)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You identify whether passages contain answer-bearing evidence."},
            {"role": "user", "content": "".join(prompt_parts)},
        ],
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }
    response = session.post(service_url, json=payload, timeout=120)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]["content"].strip()
    parsed = _extract_json(message)
    if parsed is not None:
        contains = _parse_bool(parsed.get("contains_hint"))
        comment = parsed.get("comment")
        if not isinstance(comment, str):
            comment = json.dumps(comment, ensure_ascii=False) if comment is not None else ""
    else:
        lowered = message.lower()
        contains = "no" not in lowered and "not" not in lowered
        comment = message
    return contains, comment or message, message


def call_iteration_judge(
    session: requests.Session,
    service_url: str,
    model: str,
    question: str,
    answer: str,
    documents: Sequence[Dict[str, object]],
    iteration: int,
    char_limit: int,
    prior_comments: Sequence[str],
    max_output_tokens: int,
) -> Tuple[bool, str, List[Dict[str, object]]]:
    evidence = format_evidence(documents, char_limit)
    prompt_lines: List[str] = [
        "You assess whether the collected passages justify the answer.",
        "IMPORTANT: Only set can_answer=true if the passages explicitly contain ALL facts needed to derive the answer.",
        "Do NOT rely on external knowledge or assumptions - every fact must be explicitly stated in the passages.",
        "Provide a step-by-step logical flow explaining how the passages combine to produce the answer.",
        "For each component of the answer, cite the specific passage using short format: [URL_short_name, idx=N].",
        "Use concise URL names (e.g., 'Jane_Eyre' instead of full URL).",
        "If can_answer=true, provide 'minimal_passages': the SMALLEST set of passages that, when combined following your logical flow, are sufficient to reconstruct the answer.",
        "Each passage in minimal_passages must contribute a unique, essential fact; remove any redundant passages.",
        "If insufficient, explain exactly which missing fact prevents completion.",
        "Respond with JSON: {can_answer: boolean, comment: string with step-by-step flow using short names, minimal_passages: [list of {url_short: str, index: int}] if can_answer=true}.",
        f"\nIteration: {iteration}",
        f"\nQuestion: {question}",
        f"\nExpected Answer: {answer}",
        "\nCollected passages:\n" + (evidence if evidence else "None yet."),
    ]
    if prior_comments:
        bullet = "\n".join(f"- {comment}" for comment in prior_comments if comment.strip())
        if bullet:
            prompt_lines.append("\nEarlier iteration comments (do not repeat verbatim; refine them):\n" + bullet)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You verify that gathered evidence answers the question."},
            {"role": "user", "content": "".join(prompt_lines)},
        ],
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }
    response = session.post(service_url, json=payload, timeout=120)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]["content"].strip()
    parsed = _extract_json(message)
    if parsed is not None:
        can_answer = _parse_bool(parsed.get("can_answer"))
        comment = parsed.get("comment")
        if not isinstance(comment, str):
            comment = json.dumps(comment, ensure_ascii=False) if comment is not None else ""
        # Extract minimal_passages if present and return separately
        minimal_passages = parsed.get("minimal_passages", [])
    else:
        lowered = message.lower()
        can_answer = "cannot" not in lowered and "not" not in lowered
        comment = message
        minimal_passages = []
    final_comment = (comment or message).strip()
    if not can_answer:
        final_comment = (
            final_comment
            + "\nRe-evaluate the cited passages—together they must support the answer. Explain the linkage explicitly."
        ).strip()
    if not final_comment:
        final_comment = (
            f"Iteration {iteration}: restate the reasoning with new phrasing and cite the specific passages that justify the answer."
        )
    return can_answer, final_comment, minimal_passages


class RunState:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.metadata = payload.get("metadata", {}) or {}
            questions = payload.get("questions")
            if questions is None:
                questions = payload.get("results")
            self.questions: List[Dict[str, object]] = questions or []
        else:
            self.metadata = {}
            self.questions = []
        self._by_dataset_index: Dict[int, Dict[str, object]] = {
            int(item.get("dataset_index", 0)): item for item in self.questions if item.get("dataset_index")
        }

    def ensure_question(
        self,
        dataset_index: int,
        run_order: int,
        question: str,
        answer: str,
        links: Sequence[str],
    ) -> Dict[str, object]:
        entry = self._by_dataset_index.get(dataset_index)
        if entry is None:
            entry = {
                "dataset_index": dataset_index,
                "run_order": run_order,
                "question": question,
                "answer": answer,
                "links": list(links),
                "documents": [],
                "iterations": [],
                "complete": False,
                "final_comment": "",
                "last_iteration": 0,
                "total_relevant_passages": 0,
                "total_remaining_passages": 0,
                "total_processed_passages": 0,
            }
            self.questions.append(entry)
            self._by_dataset_index[dataset_index] = entry
        else:
            entry["run_order"] = run_order
            entry["question"] = question
            entry["answer"] = answer
            entry["links"] = list(links)
        entry.setdefault("documents", [])
        entry.setdefault("iterations", [])
        entry.setdefault("complete", False)
        entry.setdefault("final_comment", "")
        entry.setdefault("last_iteration", 0)
        entry.setdefault("total_relevant_passages", 0)
        entry.setdefault("total_remaining_passages", 0)
        entry.setdefault("total_processed_passages", 0)
        return entry

    def reset_question(self, dataset_index: int) -> None:
        entry = self._by_dataset_index.get(dataset_index)
        if entry is None:
            return
        entry["documents"] = []
        entry["iterations"] = []
        entry["complete"] = False
        entry["final_comment"] = ""
        entry["last_iteration"] = 0
        entry["total_relevant_passages"] = 0
        entry["total_remaining_passages"] = 0
        entry["total_processed_passages"] = 0

    def save(self) -> None:
        meta = dict(self.metadata)
        meta["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        meta["questions_completed"] = sum(1 for q in self.questions if q.get("complete"))
        meta["processed_questions"] = max((q.get("run_order", 0) for q in self.questions), default=0)
        payload = {"metadata": meta, "questions": self.questions}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        tmp.replace(self.path)


def ensure_document(
    entry: Dict[str, object],
    url: str,
    passages_by_url: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    documents: List[Dict[str, object]] = entry.setdefault("documents", [])  # type: ignore[assignment]
    for doc in documents:
        if doc.get("url") == url:
            break
    else:
        doc = {
            "url": url,
            "source": None,
            "total_passages": 0,
            "remaining_indices": [],
            "relevant_passages": [],
        }
        documents.append(doc)
    passages = passages_by_url.get(url, [])
    indices = sorted({int(p.get("index")) for p in passages if p.get("index") is not None})
    if not doc.get("remaining_indices"):
        doc["remaining_indices"] = indices.copy()
    doc.setdefault("relevant_passages", [])
    doc["total_passages"] = len(indices)
    if doc.get("source") is None and passages:
        doc["source"] = passages[0].get("base_filename")
    # keep only valid remaining indices
    doc["remaining_indices"] = sorted(i for i in doc.get("remaining_indices", []) if i in indices)
    if not doc["remaining_indices"] and indices:
        doc["remaining_indices"] = indices.copy()
    return doc


def update_entry_totals(entry: Dict[str, object]) -> None:
    documents: Sequence[Dict[str, object]] = entry.get("documents", [])  # type: ignore[assignment]
    total_relevant = sum(len(doc.get("relevant_passages", [])) for doc in documents)
    total_remaining = sum(len(doc.get("remaining_indices", [])) for doc in documents)
    total_passages = sum(doc.get("total_passages", 0) for doc in documents)
    entry["total_relevant_passages"] = total_relevant
    entry["total_remaining_passages"] = total_remaining
    entry["total_processed_passages"] = max(total_passages - total_remaining, 0)


def process_question(
    state: RunState,
    dataset_index: int,
    run_order: int,
    question: str,
    answer: str,
    links: Sequence[str],
    passages_by_url: Dict[str, List[Dict[str, object]]],
    session: requests.Session,
    service_url: str,
    model: str,
    *,
    total_questions: int,
    include_context: bool,
    context_limit: int,
    context_char_limit: int,
    max_iterations: int,
    judge_char_limit: int,
    max_output_tokens: int,
) -> None:
    entry = state.ensure_question(dataset_index, run_order, question, answer, links)
    for iteration in range(entry.get("last_iteration", 0) + 1, max_iterations + 1):
        iteration_record = {
            "iteration": iteration,
            "processed_passages": 0,
            "new_relevant": 0,
            "can_answer": False,
            "comment": "",
            "link_visit_order": [],
            "link_visit_reasoning": "",
        }
        state.metadata["last_question_index"] = dataset_index
        state.metadata["last_question"] = question
        state.metadata["last_iteration"] = iteration

        # --- STRATEGIC LINK ORDERING VIA LLM PLANNER ---
        ordered_links, reasoning = plan_link_visit_order(
            session,
            service_url,
            model,
            question,
            answer,
            links,
            iteration,
            entry.get("documents", []),
            max_output_tokens,
        )
        iteration_record["link_visit_order"] = ordered_links
        iteration_record["link_visit_reasoning"] = reasoning

        # Track early termination
        can_answer_now = False
        
        for link_idx, url in enumerate(ordered_links, start=1):
            doc = ensure_document(entry, url, passages_by_url)
            passages = passages_by_url.get(url, [])
            remaining = set(doc.get("remaining_indices", []))
            if not remaining:
                continue

            for passage in passages:
                try:
                    idx = int(passage.get("index"))
                except (TypeError, ValueError):
                    continue
                if idx not in remaining:
                    continue

                context_text = None
                if include_context:
                    context_text = build_context_snippets(
                        entry.get("documents", []), context_limit, context_char_limit
                    )

                previous_comments = [it.get("comment", "") for it in entry.get("iterations", []) if it.get("comment")]
                prior_iteration_tail = previous_comments[-3:]
                prior_passage_comment = next(
                    (p.get("comment") for p in doc.get("relevant_passages", []) if p.get("index") == idx and p.get("comment")),
                    None,
                )

                contains, comment_text, raw_message = call_passage_judge(
                    session,
                    service_url,
                    model,
                    question,
                    answer,
                    passage.get("passage", ""),
                    iteration,
                    context_text,
                    prior_iteration_tail,
                    prior_passage_comment,
                    max_output_tokens,
                )

                remaining.discard(idx)
                doc["remaining_indices"] = sorted(remaining)
                iteration_record["processed_passages"] += 1

                if contains and not any(p.get("index") == idx for p in doc.get("relevant_passages", [])):
                    stored_comment = comment_text.strip() or f"Iteration {iteration} marked passage {idx} as relevant."
                    doc.setdefault("relevant_passages", []).append(
                        {
                            "index": idx,
                            "iteration": iteration,
                            "comment": stored_comment,
                            "content": passage.get("passage", ""),
                        }
                    )
                    iteration_record["new_relevant"] += 1
                    
                    # --- EARLY TERMINATION CHECK AFTER FINDING RELEVANT PASSAGE ---
                    # Check if we can answer now with the passages collected so far
                    update_entry_totals(entry)
                    can_answer_now, temp_comment, temp_minimal = call_iteration_judge(
                        session,
                        service_url,
                        model,
                        question,
                        answer,
                        entry.get("documents", []),
                        iteration,
                        judge_char_limit,
                        [it.get("comment", "") for it in entry.get("iterations", []) if it.get("comment")],
                        max_output_tokens,
                    )
                    if can_answer_now:
                        print(
                            f"  Early termination: Can answer with {entry['total_relevant_passages']} passages!",
                            flush=True,
                        )
                        # Exit passage loop and proceed to iteration judge
                        remaining.clear()
                        doc["remaining_indices"] = []
                        break

                update_entry_totals(entry)
                hit_label = "hit" if contains else "miss"
                print(
                    f"Q{run_order}[{iteration}] {hit_label} doc{link_idx}/{len(ordered_links)} idx{idx} "
                    f"new={iteration_record['new_relevant']} rel={entry['total_relevant_passages']} rem={entry['total_remaining_passages']}",
                    flush=True,
                )
                if iteration == 1 and iteration_record["processed_passages"] <= 2:
                    preview = raw_message.replace("\n", " ")
                    print(f"  judge: {preview[:220]}", flush=True)

                state.save()
            
            # Break out of link loop if early termination triggered
            if not remaining and can_answer_now:
                break

        entry["last_iteration"] = iteration


        previous_iteration_comments = [it.get("comment", "") for it in entry.get("iterations", []) if it.get("comment")]
        # --- LOGICAL FLOW FOR ITERATION COMMENT ---
        logical_flow = []
        logical_flow.append(f"Iteration {iteration}: Decided link visit order: {iteration_record['link_visit_order']}")
        logical_flow.append(f"Reasoning: {iteration_record['link_visit_reasoning']}")
        for url in iteration_record['link_visit_order']:
            doc = ensure_document(entry, url, passages_by_url)
            rel = len(doc.get("relevant_passages", []))
            logical_flow.append(f"After visiting {url}, found {rel} relevant passages.")
        can_answer, judge_comment, minimal_passages = call_iteration_judge(
            session,
            service_url,
            model,
            question,
            answer,
            entry.get("documents", []),
            iteration,
            judge_char_limit,
            previous_iteration_comments,
            max_output_tokens,
        )

        iteration_record["can_answer"] = bool(can_answer)
        iteration_record["comment"] = "\n".join(logical_flow) + "\n" + judge_comment
        entry.setdefault("iterations", []).append(iteration_record)

        if can_answer:
            entry["complete"] = True
            # Store minimal passages as a separate field
            entry["minimal_passages"] = minimal_passages if minimal_passages else []
            # --- LOGICAL FLOW FOR FINAL COMMENT WITH SHORT NAMES ---
            final_flow = []
            final_flow.append(f"Final logical flow for question {run_order}:")
            for it in entry["iterations"]:
                final_flow.append(f"Iteration {it['iteration']}: {it['link_visit_reasoning'][:200]}")
            final_flow.append(f"\nTotal relevant passages found: {entry['total_relevant_passages']}")
            final_flow.append(f"\nStep-by-step reasoning:")
            for url in entry["links"]:
                doc = ensure_document(entry, url, passages_by_url)
                rels = doc.get("relevant_passages", [])
                if rels:
                    short_name = _short_url_name(url)
                    indices = ", ".join(str(p['index']) for p in rels)
                    final_flow.append(f"  - {short_name}: {len(rels)} passages (indices: {indices})")
            final_flow.append(f"\nJudge analysis:")
            entry["final_comment"] = "\n".join(final_flow) + "\n" + judge_comment
            update_entry_totals(entry)
            state.save()
            print(
                f"Completed question {run_order}/{total_questions} (dataset idx {dataset_index}) after iteration {iteration}.",
                flush=True,
            )
            return

        update_entry_totals(entry)
        state.save()
        print(
            f"Iteration {iteration} finished for question {run_order}/{total_questions}; judge comment: {judge_comment[:150]}...",
            flush=True,
        )

    if not entry.get("final_comment") and entry.get("iterations"):
        entry["final_comment"] = entry["iterations"][-1].get("comment", "")
    update_entry_totals(entry)
    state.save()
    print(
        f"Max iterations reached for question {run_order}/{total_questions} (dataset idx {dataset_index}).",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteratively inspect passages for answer clues.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset TSV file")
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES, help="Path to passages JSON file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path for debug JSON output")
    parser.add_argument("--judge-url", default=DEFAULT_JUDGE_URL, help="LLM endpoint URL")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="LLM model identifier")
    parser.add_argument("--judge-max-output-tokens", type=int, default=1024, help="Maximum tokens for judge responses")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional limit on number of questions")
    parser.add_argument("--skip-questions", type=int, default=0, help="Number of questions to skip from the start")
    parser.add_argument("--resume", action="store_true", help="Reprocess questions even if already complete")
    parser.add_argument("--include-context", action="store_true", default=True, help="Include found passages when judging new ones (default: True)")
    parser.add_argument("--no-context", action="store_false", dest="include_context", help="Disable context inclusion")
    parser.add_argument("--context-limit", type=int, default=10, help="How many passages to include as context")
    parser.add_argument("--context-char-limit", type=int, default=DEFAULT_CONTEXT_LIMIT, help="Character limit per context passage")
    parser.add_argument("--judge-char-limit", type=int, default=DEFAULT_CONTEXT_LIMIT, help="Character limit per passage excerpt when asking judge")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="Maximum iterations per question")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passages_by_url = _load_passages(args.passages)
    dataset_rows = _load_dataset(args.dataset)
    state = RunState(args.output)
    session = requests.Session()

    total_dataset_questions = len(dataset_rows)
    total_questions = (
        total_dataset_questions
        if args.max_questions is None
        else min(args.max_questions, total_dataset_questions)
    )

    state.metadata.setdefault("schema_version", 3)
    state.metadata["dataset"] = str(args.dataset)
    state.metadata["passage_file"] = str(args.passages)
    state.metadata["judge_url"] = args.judge_url
    state.metadata["judge_model"] = args.judge_model
    state.metadata["include_context"] = args.include_context
    state.metadata["context_limit"] = args.context_limit
    state.metadata["context_char_limit"] = args.context_char_limit
    state.metadata["judge_char_limit"] = args.judge_char_limit
    state.metadata["judge_max_output_tokens"] = args.judge_max_output_tokens
    state.metadata["max_iterations"] = args.max_iterations
    state.metadata["total_questions"] = total_questions
    state.save()

    processed = 0
    skipped = 0
    for dataset_index, question, answer, links in dataset_rows:
        if skipped < args.skip_questions:
            skipped += 1
            continue
        if args.max_questions is not None and processed >= args.max_questions:
            break
        if not question:
            continue

        existing = state.ensure_question(dataset_index, processed + 1, question, answer, links)
        # Skip completed questions unless --resume is specified
        if existing.get("complete") and not args.resume:
            print(
                f"Skipping completed question {existing.get('run_order')} (dataset idx {dataset_index})",
                flush=True,
            )
            processed += 1
            continue

        process_question(
            state,
            dataset_index,
            processed + 1,
            question,
            answer,
            links,
            passages_by_url,
            session,
            args.judge_url,
            args.judge_model,
            total_questions=total_questions,
            include_context=args.include_context,
            context_limit=args.context_limit,
            context_char_limit=args.context_char_limit,
            max_iterations=args.max_iterations,
            judge_char_limit=args.judge_char_limit,
            max_output_tokens=args.judge_max_output_tokens,
        )
        processed += 1

    state.save()


if __name__ == "__main__":
    main()
