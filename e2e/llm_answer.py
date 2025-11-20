"""Shared helpers for generating LLM answers from retrieved evidence."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from prompt import SINGLE_SHOT_SYSTEM_PROMPT, SINGLE_SHOT_USER_PROMPT
from utils import is_token_limit_error

DEFAULT_DOC_DIR = "processed_html"


def uniform_clip_texts(texts: List[str], total_limit: Optional[int]) -> List[str]:
    if total_limit is None or total_limit <= 0 or not texts:
        return list(texts)
    count = len(texts)
    if count == 0:
        return []
    per_doc, remainder = divmod(total_limit, count)
    if per_doc <= 0 and remainder == 0:
        return [""] * count
    clipped: List[str] = []
    for idx, text in enumerate(texts):
        extra = 1 if idx < remainder else 0
        limit = per_doc + extra
        clipped.append(text[:max(limit, 0)])
    return clipped


def _get_metadata(doc: Any) -> Dict[str, Any]:
    if hasattr(doc, "metadata"):
        return doc.metadata or {}
    if isinstance(doc, dict):
        return doc
    return {}


@lru_cache(maxsize=256)
def _read_text(path: str) -> str:
    path_obj = Path(path)
    if not path_obj.exists():
        return ""
    return path_obj.read_text(encoding="utf-8", errors="ignore")


def load_document_text(
    metadata: Dict[str, Any],
    base_dir: Optional[str] = None,
    default_base_dir: str = DEFAULT_DOC_DIR,
    max_chars: Optional[int] = None,
) -> tuple[str, Optional[str]]:
    base_filename = metadata.get("base_filename")
    if not base_filename:
        return "", None

    target_dir = base_dir or default_base_dir
    base_path = Path(target_dir)
    candidates = (
        base_path / f"{base_filename}.txt",
        base_path / f"{base_filename}.html",
        base_path / f"{base_filename}.htm",
    )

    for candidate in candidates:
        candidate_path = str(candidate)
        content = _read_text(candidate_path)
        if content:
            if max_chars and max_chars > 0:
                content = content[:max_chars]
            return content, candidate_path
    return "", None


def _build_entries_from_records(
    records: Sequence[tuple[Optional[str], str, Optional[str]]],
    context_char_limit: Optional[int],
) -> List[Dict[str, Any]]:
    raw_contents = [content for (_, content, _) in records]
    clipped_contents = uniform_clip_texts(raw_contents, context_char_limit)
    entries: List[Dict[str, Any]] = []
    for idx, (url, raw_content, source_path) in enumerate(records):
        clipped_content = clipped_contents[idx] if idx < len(clipped_contents) else raw_content
        entry: Dict[str, Any] = {
            "url": url,
            "content": clipped_content,
            "raw_content": raw_content,
        }
        if source_path:
            entry["source_path"] = source_path
        entries.append(entry)
    return entries


def convert_results_to_entries(
    results: Iterable[Any],
    limit: Optional[int] = None,
    *,
    full_doc: bool = False,
    base_dir: Optional[str] = None,
    default_base_dir: str = DEFAULT_DOC_DIR,
    context_char_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    seen_ids: set[str] = set()
    records: List[tuple[Optional[str], str, Optional[str]]] = []
    count = 0

    for doc in results:
        metadata = _get_metadata(doc)
        if isinstance(doc, dict) and isinstance(doc.get("metadata"), dict):
            metadata = doc["metadata"]

        explicit_url = doc.get("url") if isinstance(doc, dict) else None
        url = (
            explicit_url
            or metadata.get("original_url")
            or metadata.get("source")
        )
        base_filename = (
            metadata.get("base_filename")
            or (doc.get("base_filename") if isinstance(doc, dict) else None)
        )
        doc_id = url or base_filename
        if doc_id and doc_id in seen_ids:
            continue

        if full_doc:
            content, source_path = load_document_text(
                metadata,
                base_dir=base_dir,
                default_base_dir=default_base_dir,
                max_chars=context_char_limit,
            )
            if not content:
                fallback_candidates: List[Optional[str]] = []
                if isinstance(doc, dict):
                    fallback_candidates.extend([
                        doc.get("content"),
                        doc.get("raw_content"),
                    ])
                fallback_candidates.extend([
                    getattr(doc, "page_content", None),
                    metadata.get("content"),
                ])
                content = next((text for text in fallback_candidates if text), "")
        else:
            if isinstance(doc, dict):
                content = doc.get("content") or doc.get("raw_content") or ""
            else:
                content = getattr(doc, "page_content", metadata.get("content", "")) or ""
            source_path = None

        records.append((url, content, source_path))
        if doc_id:
            seen_ids.add(doc_id)
        count += 1
        if limit and limit > 0 and count >= limit:
            break

    return _build_entries_from_records(records, context_char_limit)


def extract_unique_urls(results: Iterable[Any]) -> List[str]:
    urls: List[str] = []
    seen = set()
    for doc in results:
        metadata = _get_metadata(doc)
        url = metadata.get("original_url") or metadata.get("source")
        if url and url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def _build_evidence_block(doc_entries: List[Dict[str, Any]], char_limit: int) -> str:
    raw_texts = [doc.get("raw_content") or doc.get("content", "") or "" for doc in doc_entries]
    clipped = uniform_clip_texts(raw_texts, char_limit)
    parts: List[str] = []
    for idx, doc in enumerate(doc_entries, 1):
        snippet = clipped[idx - 1] if idx - 1 < len(clipped) else raw_texts[idx - 1]
        doc["content"] = snippet
        source = doc.get("url") or "Unknown source"
        parts.append(f"[{idx}] Source: {source}\n{snippet.strip()}")
    return "\n\n".join(parts) if parts else "No supporting documents were retrieved."


def generate_answer_from_entries(
    question: str,
    doc_entries: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
    base_char_limit: Optional[int] = None,
) -> str:
    """Run the single-shot QA prompt against the provided evidence entries."""
    if not doc_entries or not llm_config:
        return ""

    if base_char_limit is None or base_char_limit <= 0:
        base_char_limit = sum(len(doc.get("raw_content", doc.get("content", "")) or "") for doc in doc_entries)

    output_token_limit = llm_config.get("output_token_limit")
    service_url = llm_config.get("service_url")
    model_name = llm_config.get("model_name")
    request_timeout = llm_config.get("request_timeout", 600)

    attempt_limit = base_char_limit
    min_limit = max(256, attempt_limit // 4) if attempt_limit else 256
    retry_factor = 0.6
    max_attempts = 4

    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        evidence_block = _build_evidence_block(doc_entries, attempt_limit)
        user_prompt = SINGLE_SHOT_USER_PROMPT.format(question=question, evidence=evidence_block)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SINGLE_SHOT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": output_token_limit,
        }

        try:
            response = requests.post(service_url, json=payload, timeout=request_timeout)
            print(response)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as http_err:
            last_error = http_err
            if is_token_limit_error(http_err.response, str(http_err)) and attempt < max_attempts - 1:
                new_limit = int(attempt_limit * retry_factor)
                attempt_limit = max(min_limit, new_limit)
                continue
            raise
        except (requests.exceptions.RequestException, json.JSONDecodeError) as req_err:
            last_error = req_err
            raise

    if last_error:
        raise last_error
    return ""


__all__ = [
    "generate_answer_from_entries",
    "uniform_clip_texts",
    "convert_results_to_entries",
    "load_document_text",
    "extract_unique_urls",
]
