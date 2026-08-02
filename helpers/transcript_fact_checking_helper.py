"""DeepSeek transcript fact-checking utilities for summary bullets."""

from __future__ import annotations

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import requests
from requests import RequestException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers.downloading_transcript_helper import sanitize_filename_part
from helpers.output_helper import write_json
from helpers.transcript_summarization_helper import (
    DEFAULT_BACKOFF_SECONDS,
    build_encoding,
    build_llm,
    compute_cost,
    count_tokens,
    extract_usage_counts,
    invoke_with_retries,
)


DEEPSEEK_RESPONSES_URL = "https://api.deepseek.com/v1/responses"
FACT_CHECK_MODEL = "deepseek-v4-flash"
FACT_CHECK_FALLBACK_MODEL = "deepseek-v4-pro"
FACT_CHECK_TEMPERATURE = 0.0
FACT_CHECK_MAX_TOKENS = 20000
FACT_CHECK_MAX_RETRIES = 1
FACT_CHECK_BACKOFF_SECONDS = DEFAULT_BACKOFF_SECONDS
FACT_CHECK_BATCH_SIZE = 3
FACT_CHECK_REQUEST_TIMEOUT_SECONDS = 120
FACT_CHECK_BATCH_MAX_WORKERS = 4
SEARCH_API_KEY_ENV = "SEARCH_API_KEY"

FACT_CHECK_OBVIOUS_OPINION_HINTS = (
    "i think",
    "i believe",
    "i feel",
    "i guess",
    "likely",
    "probably",
    "possibly",
    "maybe",
    "might",
    "could",
    "appears",
    "seems",
    "suggests",
    "in my view",
    "in my opinion",
    "should",
    "would",
    "impossible",
    "absurd",
    "irrational",
)


class FactCheckBulletDraft(BaseModel):
    """Draft structure expected from the fact-check model."""

    bullet_id: int
    validation_status: Literal["true", "false", "unverified"]
    note: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FactCheckDraft(BaseModel):
    """Parsed fact-check draft returned by the model."""

    fact_checks: list[FactCheckBulletDraft] = Field(default_factory=list)


class FactCheckBulletResult(BaseModel):
    """Single bullet fact-check result written to disk."""

    bullet_id: int
    text: str
    is_opinion: bool
    validation_status: str
    note: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FactCheckRecord(BaseModel):
    """Final fact-check record written to pipeline outputs."""

    video_id: str
    channel_name: str
    title: str
    fact_checks: list[FactCheckBulletResult] = Field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    success: bool = True
    error: str | None = None


def build_fact_check_prompt() -> str:
    """
    Build the instruction block for fact checking summary bullets.

    Arguments:
        None

    Returns:
        str: Prompt instructions for the fact-check model.

    Example:
        >>> "confidence" in build_fact_check_prompt()
        True
    """
    current_date = datetime.now().isoformat(timespec="seconds")
    prompt = (
        "You are a strict fact checker for transcript summary bullets.\n\n"
        f"Current date: {current_date}\n"
        "Warning: your knowledge cutoff may be stale for current events. Use web search results when available.\n\n"
        "RULES:\n"
        "1. Check only bullets where is_opinion is false.\n"
        "2. Prefer live web evidence when search context or web search results are available.\n"
        "3. Return JSON only.\n"
        "4. Use these statuses only: true, false, unverified.\n"
        "5. Keep the note short and factual.\n"
        "6. Include source URLs in sources when evidence is available.\n"
        "7. Confidence must be between 0.0 and 1.0.\n"
        "8. Do not rewrite the bullet text.\n\n"
        "JSON SHAPE:\n"
        '{"fact_checks": [{"bullet_id": 1, "validation_status": "true", "note": "...", "sources": ["https://..."], "confidence": 0.87}]}\n'
    )

    return prompt


def build_search_query(record: dict[str, Any], search_query: str | None = None) -> str:
    """
    Build a search query for the fact-check step.

    Arguments:
        record (dict[str, Any]): Summary transcript record.
        search_query (str | None): Optional explicit query.

    Returns:
        str: Normalized search query.

    Example:
        >>> build_search_query({"title": "Demo", "bullets": [{"text": "Claim"}]})
        'Demo Claim'
    """
    if search_query:
        query_text = str(search_query)
    else:
        title = str(record.get("title") or "").strip()
        bullets = record.get("bullets") or []
        bullet_texts = [str(bullet.get("text") or "").strip() for bullet in bullets if str(bullet.get("text") or "").strip()]
        query_text = " ".join(part for part in [title, *bullet_texts[:2]] if part)

    normalized_query = re.sub(r"\s+", " ", query_text).strip()
    if len(normalized_query) > 300:
        normalized_query = normalized_query[:300].rstrip()

    return normalized_query


def _chunk_list(values: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into fixed-size chunks."""
    if chunk_size <= 0:
        return [values]

    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _looks_like_obvious_opinion(text: str) -> bool:
    """Heuristically detect bullets that should be skipped as opinion."""
    normalized_text = str(text or "").strip().lower()
    if not normalized_text:
        return False

    return any(hint in normalized_text for hint in FACT_CHECK_OBVIOUS_OPINION_HINTS)


def _normalize_fact_check_bullets(bullets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize bullet opinion flags using metadata plus a lightweight heuristic."""
    normalized_bullets: list[dict[str, Any]] = []

    for bullet in bullets:
        bullet_copy = dict(bullet)
        text = str(bullet_copy.get("text") or "")
        bullet_copy["is_opinion"] = bool(bullet_copy.get("is_opinion", False)) or _looks_like_obvious_opinion(text)
        normalized_bullets.append(bullet_copy)

    return normalized_bullets


def _build_unverified_batch_draft(bullets: list[dict[str, Any]], note: str) -> FactCheckDraft:
    """Build a fallback draft for a batch that failed or timed out."""
    fact_checks = [
        FactCheckBulletDraft(
            bullet_id=int(bullet.get("bullet_id") or 0),
            validation_status="unverified",
            note=note,
            sources=[],
            confidence=0.0,
        )
        for bullet in bullets
    ]

    return FactCheckDraft(fact_checks=fact_checks)


def build_fact_check_messages(
    record: dict[str, Any],
    search_results: list[dict[str, Any]] | None = None,
    search_query: str | None = None,
) -> list[SystemMessage | HumanMessage]:
    """
    Build the chat messages for one fact-check request.

    Arguments:
        record (dict[str, Any]): Summary transcript record.
        search_results (list[dict[str, Any]] | None): Optional web search context.
        search_query (str | None): Optional explicit search query.

    Returns:
        list[SystemMessage | HumanMessage]: Messages to send to the model.

    Example:
        >>> messages = build_fact_check_messages({"title": "Demo", "bullets": []})
        >>> len(messages) >= 2
        True
    """
    bullets = record.get("bullets") or []
    checkable_bullets = [bullet for bullet in bullets if not bullet.get("is_opinion", False)]
    system_text = build_fact_check_prompt()
    human_payload: dict[str, Any] = {
        "video_id": record.get("video_id", ""),
        "channel_name": record.get("channel_name", ""),
        "title": record.get("title", ""),
        "bullets": checkable_bullets,
    }
    if search_query:
        human_payload["search_query"] = search_query
    if search_results is not None:
        human_payload["search_results"] = search_results

    human_text = json.dumps(human_payload, indent=2, ensure_ascii=False)
    messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]

    return messages


def _normalize_string_list(values: Any) -> list[str]:
    """Normalize values into a unique list of non-empty strings."""
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    normalized_values: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            normalized_values.append(text)

    return normalized_values


def _normalize_confidence(value: Any) -> float:
    """Normalize a confidence value to the range [0.0, 1.0]."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    return confidence


def _coerce_fact_check_bullet_draft(item: Any) -> FactCheckBulletDraft:
    """Coerce an arbitrary JSON item into a bullet draft."""
    item = item if isinstance(item, dict) else {}
    validation_status = str(item.get("validation_status") or "unverified").strip().lower()
    if validation_status not in {"true", "false", "unverified"}:
        validation_status = "unverified"

    return FactCheckBulletDraft(
        bullet_id=int(item.get("bullet_id") or 0),
        validation_status=validation_status,
        note=str(item.get("note") or ""),
        sources=_normalize_string_list(item.get("sources")),
        confidence=_normalize_confidence(item.get("confidence")),
    )


def parse_fact_check_draft(response_text: str) -> FactCheckDraft:
    """
    Parse the model response into a fact-check draft.

    Arguments:
        response_text (str): Raw model response text.

    Returns:
        FactCheckDraft: Parsed fact-check draft.

    Example:
        >>> isinstance(parse_fact_check_draft('{"fact_checks": []}'), FactCheckDraft)
        True
    """
    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.lower().startswith("json"):
            cleaned_text = cleaned_text[4:]

    try:
        draft = FactCheckDraft.model_validate_json(cleaned_text)
    except Exception:
        try:
            payload = json.loads(cleaned_text)
        except json.JSONDecodeError:
            payload = {}

        fact_checks = payload.get("fact_checks") if isinstance(payload, dict) else []
        bullet_drafts = [_coerce_fact_check_bullet_draft(item) for item in fact_checks or []]
        draft = FactCheckDraft(fact_checks=bullet_drafts)

    return draft


def _extract_json_like_fragment(text: str) -> str:
    """Extract a likely JSON fragment from a text blob."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)

    return text


def _extract_text_from_response_payload(payload: Any) -> str:
    """Extract response text from a DeepSeek Responses API payload."""
    if isinstance(payload, str):
        logging.debug("📝 EXTRACT: Extracted text length: %s", len(payload))
        return payload

    if not isinstance(payload, dict):
        logging.debug("📝 EXTRACT: Extracted text length: 0")
        return ""

    direct_text = payload.get("output_text") or payload.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        logging.debug("📝 EXTRACT: Extracted text length: %s", len(direct_text.strip()))
        return direct_text.strip()

    output_items = payload.get("output")
    if isinstance(output_items, list):
        logging.debug(
            "📝 EXTRACT: Output structure: %s",
            [item.get("type") if isinstance(item, dict) else type(item).__name__ for item in output_items],
        )
    if isinstance(output_items, list):
        for item in reversed(output_items):
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or "").strip().lower()
            role = str(item.get("role") or "").strip().lower()
            if item_type != "message" or role != "assistant":
                continue

            content = item.get("content")
            if isinstance(content, list):
                first_content_item = content[0] if content else None
                if isinstance(first_content_item, dict):
                    nested_text = first_content_item.get("text")
                    if isinstance(nested_text, str) and nested_text.strip():
                        extracted_text = nested_text.strip()
                        logging.debug("📝 EXTRACT: Extracted text length: %s", len(extracted_text))
                        return extracted_text

    text_parts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                text_parts.append(value.strip())
            return

        if isinstance(value, list):
            for item in value:
                visit(item)
            return

        if isinstance(value, dict):
            for key in ("text", "output_text"):
                nested_text = value.get(key)
                if isinstance(nested_text, str) and nested_text.strip():
                    text_parts.append(nested_text.strip())
            for key in ("content", "output", "messages", "choices"):
                visit(value.get(key))

    visit(payload.get("output"))
    visit(payload.get("choices"))
    visit(payload.get("content"))

    if text_parts:
        extracted_text = "\n".join(text_parts)
        logging.debug("📝 EXTRACT: Extracted text length: %s", len(extracted_text))
        return extracted_text

    logging.debug("📝 EXTRACT: Extracted text length: 0")
    return ""


def _extract_url_from_citation_candidate(candidate: Any) -> str:
    """Extract a URL from a citation-like payload."""
    if isinstance(candidate, str):
        candidate = candidate.strip()
        if candidate.startswith("http"):
            return candidate

        url_match = re.search(r"https?://[^\s\"'<>]+", candidate)
        if url_match:
            return url_match.group(0).rstrip(".,;)]}")

        return ""

    if not isinstance(candidate, dict):
        return ""

    for key in (
        "url",
        "source_url",
        "href",
        "link",
        "document_url",
        "page_url",
        "sourceUrl",
        "canonical_url",
        "article_url",
        "web_url",
        "uri",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("source", "document", "page", "article", "result"):
        nested_value = candidate.get(key)
        url = _extract_url_from_citation_candidate(nested_value)
        if url:
            return url

    for key in ("citation", "url_citation"):
        nested_value = candidate.get(key)
        url = _extract_url_from_citation_candidate(nested_value)
        if url:
            return url

    return ""


def _extract_citations_from_payload(payload: Any) -> list[str]:
    """Extract citation URLs from a DeepSeek Responses API payload."""
    citation_urls: list[str] = []
    seen: set[str] = set()
    url_pattern = re.compile(r"https?://[^\s\"'<>]+")

    def add_url(url: str) -> None:
        if url and url not in seen:
            seen.add(url)
            citation_urls.append(url)

    def add_urls_from_text(text: Any) -> None:
        if not isinstance(text, str) or not text.strip():
            return

        for match in url_pattern.findall(text):
            add_url(match.rstrip(".,;)]}"))

    def visit_annotations(annotations: Any) -> None:
        if isinstance(annotations, list):
            for annotation in annotations:
                visit(annotation)
        else:
            visit(annotations)

    def visit(value: Any) -> None:
        if value is None:
            return

        if isinstance(value, list):
            for item in value:
                visit(item)
            return

        if isinstance(value, dict):
            url = _extract_url_from_citation_candidate(value)
            if url:
                add_url(url)

            visit_annotations(value.get("annotations"))

            for text_key in (
                "text",
                "output_text",
                "url",
                "source_url",
                "href",
                "link",
                "document_url",
                "page_url",
                "sourceUrl",
                "canonical_url",
                "article_url",
                "web_url",
                "uri",
            ):
                text_value = value.get(text_key)
                if isinstance(text_value, str):
                    url = _extract_url_from_citation_candidate(text_value)
                    if url:
                        add_url(url)

            for key in ("citations", "output", "content", "items", "data"):
                visit(value.get(key))
            return

        if isinstance(value, str):
            add_urls_from_text(value)

    if isinstance(payload, dict):
        output_items = payload.get("output")

        visit_annotations(payload.get("annotations"))
        visit(payload.get("annotations"))
        visit(payload.get("citations"))
        visit(output_items)
        visit(payload.get("response"))

        if not citation_urls and isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue

                item_type = str(item.get("type") or "").strip().lower()
                if item_type != "web_search_call":
                    continue

                logging.debug(
                    "🔎 WEB SEARCH CALL STRUCTURE: %s",
                    json.dumps(item, indent=2, ensure_ascii=False, default=str),
                )

                visit(item.get("results"))
                visit(item.get("output"))
                visit(item.get("content"))
                visit(item.get("data"))
                visit(item.get("response"))
                visit(item.get("action"))

                if citation_urls:
                    break

        if not citation_urls:
            final_message_text = _extract_text_from_response_payload(payload)
            add_urls_from_text(final_message_text)

        if not citation_urls:
            def visit_all_strings(value: Any) -> None:
                if value is None:
                    return

                if isinstance(value, str):
                    add_urls_from_text(value)
                    return

                if isinstance(value, list):
                    for item in value:
                        visit_all_strings(item)
                    return

                if isinstance(value, dict):
                    for nested_value in value.values():
                        visit_all_strings(nested_value)

            visit_all_strings(payload)

    logging.debug("📚 CITATIONS: Found %s citations", len(citation_urls))
    return citation_urls


def _extract_usage_from_payload(
    payload: dict[str, Any],
    fallback_input_tokens: int,
    response_text: str,
    encoding: Any,
) -> tuple[int, int, int]:
    """Extract usage counts from a Responses API payload."""
    usage = payload.get("usage") or payload.get("usage_metadata") or {}
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or fallback_input_tokens
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")

    if output_tokens is None:
        output_tokens = count_tokens(encoding, response_text)

    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = int(input_tokens) + int(output_tokens)

    return int(input_tokens), int(output_tokens), int(total_tokens)


def _dedupe_sources(primary_sources: list[str], fallback_sources: list[str] | None = None) -> list[str]:
    """Merge source lists while preserving order and uniqueness."""
    merged_sources: list[str] = []
    seen: set[str] = set()

    for source in list(primary_sources) + list(fallback_sources or []):
        cleaned_source = str(source).strip()
        if cleaned_source and cleaned_source not in seen:
            seen.add(cleaned_source)
            merged_sources.append(cleaned_source)

    return merged_sources


def _apply_default_sources_to_draft(draft: FactCheckDraft, default_sources: list[str]) -> FactCheckDraft:
    """Apply global citation sources to draft bullets that do not already have sources."""
    if not default_sources:
        return draft

    merged_bullets: list[FactCheckBulletDraft] = []
    for item in draft.fact_checks:
        merged_sources = item.sources or []
        if not merged_sources:
            merged_sources = list(default_sources)
        else:
            merged_sources = _dedupe_sources(merged_sources, default_sources)

        merged_bullets.append(
            FactCheckBulletDraft(
                bullet_id=item.bullet_id,
                validation_status=item.validation_status,
                note=item.note,
                sources=merged_sources,
                confidence=_normalize_confidence(item.confidence),
            )
        )

    return FactCheckDraft(fact_checks=merged_bullets)


def _post_responses_api(
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Send a request to the DeepSeek Responses API and return the JSON payload."""
    logging.debug("📡 API CALL: Sending request to %s", DEEPSEEK_RESPONSES_URL)
    response = requests.post(
        DEEPSEEK_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    logging.debug("📡 API CALL: Status code: %s", response.status_code)

    if response.status_code >= 400:
        response_text = response.text.strip()
        response_json: Any = None
        try:
            response_json = response.json()
        except ValueError:
            response_json = None

        logging.error("DeepSeek Responses API error status=%s", response.status_code)
        logging.error("DeepSeek Responses API error body (first 4000 chars): %s", response_text[:4000])
        if response_json is not None:
            logging.error(
                "DeepSeek Responses API error JSON (first 4000 chars): %s",
                json.dumps(response_json, indent=2, ensure_ascii=False, default=str)[:4000],
            )

        error_message = ""
        if isinstance(response_json, dict):
            error_message = str(
                response_json.get("error", {}).get("message")
                or response_json.get("message")
                or response_json.get("detail")
                or ""
            ).strip()

        if not error_message:
            error_message = response_text or f"HTTP {response.status_code}"

        raise RequestException(f"DeepSeek Responses API error {response.status_code}: {error_message}")

    response.raise_for_status()
    response_payload = response.json()

    if not isinstance(response_payload, dict):
        raise ValueError("Responses API returned an unexpected payload shape.")

    logging.debug("📡 API CALL: Response keys: %s", list(response_payload.keys()))
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "📡 API CALL: Response preview (first 2000 chars): %s",
            json.dumps(response_payload, indent=2, ensure_ascii=False, default=str)[:2000],
        )

    return response_payload


def _invoke_responses_api_with_retries(
    api_key: str,
    payload: dict[str, Any],
    max_retries: int = FACT_CHECK_MAX_RETRIES,
    backoff_seconds: int = FACT_CHECK_BACKOFF_SECONDS,
    timeout_seconds: int = FACT_CHECK_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Invoke the Responses API with retries."""
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response_payload = _post_responses_api(api_key, payload, timeout_seconds=timeout_seconds)
            return response_payload
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            if attempt == max_retries:
                break

            delay_seconds = backoff_seconds * (2**attempt)
            logging.warning(
                "Responses API request failed on attempt %s/%s; retrying in %s seconds: %s",
                attempt + 1,
                max_retries + 1,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("Responses API invocation failed without an exception.")


def _run_fact_check_batch_with_web_search(
    api_key: str,
    batch_record: dict[str, Any],
    search_query: str | None,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    backoff_seconds: int,
) -> tuple[FactCheckDraft, list[str], int, int, int]:
    """Run a single web-search batch and return a draft, citations, and usage stats."""
    resolved_search_query = build_search_query(batch_record, search_query)
    logging.debug(
        "🔍 BATCH START: Processing %s bullets for video %s",
        len(batch_record.get("bullets") or []),
        batch_record.get("video_id", "unknown"),
    )

    request_record = {
        "video_id": batch_record.get("video_id", ""),
        "channel_name": batch_record.get("channel_name", ""),
        "title": batch_record.get("title", ""),
        "bullets": batch_record.get("bullets") or [],
    }
    if resolved_search_query:
        request_record["search_query"] = resolved_search_query

    human_payload_json = json.dumps(request_record, indent=2, ensure_ascii=False)
    logging.debug("🔍 BATCH: Request payload size: %s chars", len(human_payload_json))
    request_payload = {
        "model": model_name,
        "input": [
            {"role": "system", "content": build_fact_check_prompt()},
            {"role": "user", "content": human_payload_json},
        ],
        "tools": [{"type": "web_search"}],
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    prompt_text = "\n\n".join([build_fact_check_prompt(), human_payload_json])
    encoding = build_encoding(model_name)
    fallback_input_tokens = count_tokens(encoding, prompt_text)

    try:
        start_time = time.perf_counter()
        response_payload = _invoke_responses_api_with_retries(
            api_key,
            request_payload,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            timeout_seconds=FACT_CHECK_REQUEST_TIMEOUT_SECONDS,
        )
        elapsed_time = time.perf_counter() - start_time
        response_text = _extract_text_from_response_payload(response_payload)
        response_text = _extract_json_like_fragment(response_text)
        citations = _extract_citations_from_payload(response_payload)
        logging.debug("🔍 BATCH: Response received in %.2f seconds", elapsed_time)
        logging.debug("🔍 BATCH: Response text length: %s chars", len(response_text))
        logging.debug("🔍 BATCH: Citations found: %s", len(citations))
        draft = parse_fact_check_draft(response_text)
        draft = _apply_default_sources_to_draft(draft, citations)
        input_tokens, output_tokens, total_tokens = _extract_usage_from_payload(
            response_payload,
            fallback_input_tokens,
            response_text,
            encoding,
        )
        return draft, citations, input_tokens, output_tokens, total_tokens
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Web-search batch failed for %s; returning unverified batch results: %s",
            batch_record.get("video_id", "unknown"),
            exc,
        )
        bullets = batch_record.get("bullets") or []
        draft = _build_unverified_batch_draft(bullets, f"Fact-check request timed out or failed: {exc}")
        return draft, [], 0, 0, 0


def _run_fact_check_batch_with_web_search_simplified(
    api_key: str,
    batch_record: dict[str, Any],
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    backoff_seconds: int,
) -> tuple[FactCheckDraft, list[str], int, int, int]:
    """Retry a batch with a simplified payload that omits the search query."""
    simplified_record = dict(batch_record)
    simplified_record.pop("search_query", None)
    logging.debug(
        "🔁 BATCH RETRY: Re-running simplified batch for video %s",
        simplified_record.get("video_id", "unknown"),
    )
    return _run_fact_check_batch_with_web_search(
        api_key,
        simplified_record,
        None,
        model_name,
        temperature,
        max_tokens,
        max_retries,
        backoff_seconds,
    )


def build_fact_check_filename(record: dict[str, Any]) -> str:
    """
    Build the filename for one fact-check result.

    Arguments:
        record (dict[str, Any]): Summary transcript record.

    Returns:
        str: Filename prefixed with summary_fc_.

    Example:
        >>> build_fact_check_filename({"video_id": "abc", "channel_name": "Demo"})
        'summary_fc_Demo_abc.json'
    """
    channel_name = sanitize_filename_part(str(record.get("channel_name") or "unknown"))
    video_id = str(record.get("video_id") or "unknown")
    filename = f"summary_fc_{channel_name}_{video_id}.json"

    return filename


def merge_fact_check_results(
    record: dict[str, Any],
    draft: FactCheckDraft,
    default_sources: list[str] | None = None,
) -> dict[str, Any]:
    """
    Merge model output into a summary transcript record.

    Arguments:
        record (dict[str, Any]): Summary transcript record.
        draft (FactCheckDraft): Parsed fact-check draft.
        default_sources (list[str] | None): Optional citations to apply when bullet sources are missing.

    Returns:
        dict[str, Any]: Fact-check record with bullet verdicts attached.

    Example:
        >>> merged = merge_fact_check_results({"video_id": "1", "channel_name": "Demo", "title": "T", "summary": "", "bullets": []}, FactCheckDraft())
        >>> merged["fact_checks"]
        []
    """
    draft_by_id = {item.bullet_id: item for item in draft.fact_checks}
    fallback_sources = default_sources or []
    merged_bullets: list[FactCheckBulletResult] = []

    for bullet in record.get("bullets") or []:
        bullet_id = int(bullet.get("bullet_id") or 0)
        is_opinion = bool(bullet.get("is_opinion", False))

        if is_opinion:
            validation_status = "skipped"
            note = "Skipped because opinion"
            sources = []
            confidence = 0.0
        else:
            draft_item = draft_by_id.get(bullet_id)
            if draft_item is None:
                validation_status = "unverified"
                note = "No fact-check result returned"
                sources = list(fallback_sources)
                confidence = 0.0
            else:
                validation_status = draft_item.validation_status
                note = draft_item.note
                sources = _dedupe_sources(draft_item.sources, fallback_sources)
                confidence = _normalize_confidence(draft_item.confidence)

        merged_bullets.append(
            FactCheckBulletResult(
                bullet_id=bullet_id,
                text=str(bullet.get("text") or ""),
                is_opinion=is_opinion,
                validation_status=validation_status,
                note=note,
                sources=sources,
                confidence=confidence,
            )
        )

    fact_check_record = FactCheckRecord(
        video_id=str(record.get("video_id") or ""),
        channel_name=str(record.get("channel_name") or ""),
        title=str(record.get("title") or ""),
        fact_checks=merged_bullets,
    )

    if hasattr(fact_check_record, "model_dump"):
        fact_check_payload = fact_check_record.model_dump()
    else:
        fact_check_payload = fact_check_record.dict()

    return fact_check_payload


def fact_check_transcript_record_with_web_search(
    record: dict[str, Any],
    api_key: str,
    search_query: str | None = None,
    search_provider: str | None = None,
    model_name: str = FACT_CHECK_MODEL,
    temperature: float = FACT_CHECK_TEMPERATURE,
    max_tokens: int = FACT_CHECK_MAX_TOKENS,
    max_retries: int = FACT_CHECK_MAX_RETRIES,
    backoff_seconds: int = FACT_CHECK_BACKOFF_SECONDS,
) -> tuple[FactCheckDraft, list[str], int, int, int]:
    """
    Fact-check one summary transcript record using the DeepSeek Responses API with web search enabled.

    Arguments:
        record (dict[str, Any]): Summary transcript record.
        api_key (str): DeepSeek API key.
        search_query (str | None): Optional search query override.
        search_provider (str | None): Reserved compatibility argument for caller-provided search provider.
        model_name (str): DeepSeek web-search capable model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.

    Returns:
        tuple[FactCheckDraft, list[str], int, int, int]: Parsed draft, citation URLs, input tokens, output tokens, total tokens.

    Example:
        >>> isinstance(fact_check_transcript_record_with_web_search({"title": "T", "bullets": []}, "key")[0], FactCheckDraft)
        True
    """
    _ = search_provider
    normalized_bullets = _normalize_fact_check_bullets(list(record.get("bullets") or []))
    checkable_bullets = [bullet for bullet in normalized_bullets if not bullet.get("is_opinion", False)]
    batch_size = 3 if len(checkable_bullets) <= 6 else 2
    bullet_batches = _chunk_list(checkable_bullets, batch_size)
    logging.debug(
        "📊 VIDEO: %s has %s checkable bullets",
        record.get("video_id", "unknown"),
        len(checkable_bullets),
    )
    logging.debug(
        "📊 VIDEO: Using batch size %s and splitting into %s batches",
        batch_size,
        len(bullet_batches),
    )
    merged_fact_checks: list[FactCheckBulletDraft] = []
    merged_citations: list[str] = []
    seen_citations: set[str] = set()
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    if not bullet_batches:
        logging.debug(
            "✅ VIDEO COMPLETE: %s | Fact-checked: %s/%s | Sources: %s | Tokens: %s",
            record.get("video_id", "unknown"),
            0,
            0,
            0,
            0,
        )
        return FactCheckDraft(fact_checks=[]), [], 0, 0, 0

    max_workers = min(FACT_CHECK_BATCH_MAX_WORKERS, len(bullet_batches))
    logging.debug("📊 VIDEO: Using %s workers", max_workers)
    batch_results: list[tuple[int, FactCheckDraft, list[str], int, int, int]] = []
    batch_retry_candidates: list[tuple[int, dict[str, Any]]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch_index = {
            executor.submit(
                _run_fact_check_batch_with_web_search,
                api_key,
                {
                    "video_id": record.get("video_id", ""),
                    "channel_name": record.get("channel_name", ""),
                    "title": record.get("title", ""),
                    "bullets": bullet_batch,
                },
                search_query,
                model_name,
                temperature,
                max_tokens,
                max_retries,
                backoff_seconds,
            ): index
            for index, bullet_batch in enumerate(bullet_batches, start=1)
        }

        for future in as_completed(future_to_batch_index):
            batch_index = future_to_batch_index[future]
            draft, citations, input_tokens, output_tokens, batch_total_tokens = future.result()
            batch_record = {
                "video_id": record.get("video_id", ""),
                "channel_name": record.get("channel_name", ""),
                "title": record.get("title", ""),
                "bullets": bullet_batches[batch_index - 1],
            }
            if not draft.fact_checks and citations:
                batch_retry_candidates.append((batch_index, batch_record))
            batch_results.append((batch_index, draft, citations, input_tokens, output_tokens, batch_total_tokens))

    if batch_retry_candidates:
        logging.debug("🔁 VIDEO: Retrying %s batch(es) with simplified payloads", len(batch_retry_candidates))
        retry_results: dict[int, tuple[FactCheckDraft, list[str], int, int, int]] = {}
        with ThreadPoolExecutor(max_workers=min(FACT_CHECK_BATCH_MAX_WORKERS, len(batch_retry_candidates))) as executor:
            future_to_retry_index = {
                executor.submit(
                    _run_fact_check_batch_with_web_search_simplified,
                    api_key,
                    batch_record,
                    model_name,
                    temperature,
                    max_tokens,
                    max_retries,
                    backoff_seconds,
                ): batch_index
                for batch_index, batch_record in batch_retry_candidates
            }

            for future in as_completed(future_to_retry_index):
                batch_index = future_to_retry_index[future]
                retry_results[batch_index] = future.result()

        updated_batch_results: list[tuple[int, FactCheckDraft, list[str], int, int, int]] = []
        for index, existing_draft, existing_citations, existing_input_tokens, existing_output_tokens, existing_total_tokens in batch_results:
            retry_result = retry_results.get(index)
            if retry_result is None:
                updated_batch_results.append(
                    (
                        index,
                        existing_draft,
                        existing_citations,
                        existing_input_tokens,
                        existing_output_tokens,
                        existing_total_tokens,
                    )
                )
                continue

            retry_draft, retry_citations, retry_input_tokens, retry_output_tokens, retry_total_tokens = retry_result
            if retry_draft.fact_checks:
                logging.debug("🔁 VIDEO: Batch %s produced fact_checks on simplified retry", index)
                updated_batch_results.append(
                    (
                        index,
                        retry_draft,
                        retry_citations,
                        retry_input_tokens,
                        retry_output_tokens,
                        retry_total_tokens,
                    )
                )
            else:
                fallback_sources = retry_citations or existing_citations
                if fallback_sources:
                    logging.debug(
                        "🔁 VIDEO: Batch %s returned citations but no fact_checks after retry; falling back to unverified results",
                        index,
                    )
                    unverified_draft = _build_unverified_batch_draft(
                        bullet_batches[index - 1],
                        "Fact-check request returned citations but no JSON fact-checks after retry",
                    )
                    updated_batch_results.append(
                        (
                            index,
                            unverified_draft,
                            fallback_sources,
                            retry_input_tokens or existing_input_tokens,
                            retry_output_tokens or existing_output_tokens,
                            retry_total_tokens or existing_total_tokens,
                        )
                    )
                    continue

                updated_batch_results.append(
                    (
                        index,
                        existing_draft,
                        existing_citations,
                        existing_input_tokens,
                        existing_output_tokens,
                        existing_total_tokens,
                    )
                )

        batch_results = updated_batch_results

    for batch_index, draft, citations, input_tokens, output_tokens, batch_total_tokens in sorted(batch_results, key=lambda item: item[0]):
        merged_fact_checks.extend(draft.fact_checks)
        for citation in citations:
            if citation not in seen_citations:
                seen_citations.add(citation)
                merged_citations.append(citation)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += batch_total_tokens
        logging.debug(
            "📊 VIDEO: Batch %s/%s completed (input=%s, output=%s, total=%s)",
            batch_index,
            len(bullet_batches),
            input_tokens,
            output_tokens,
            batch_total_tokens,
        )

    merged_draft = FactCheckDraft(fact_checks=merged_fact_checks)
    logging.debug(
        "✅ VIDEO COMPLETE: %s | Fact-checked: %s/%s | Sources: %s | Tokens: %s",
        record.get("video_id", "unknown"),
        len(merged_fact_checks),
        len(checkable_bullets),
        len(merged_citations),
        total_tokens,
    )
    return merged_draft, merged_citations, total_input_tokens, total_output_tokens, total_tokens


def fact_check_transcript_record(
    record: dict[str, Any],
    api_key: str,
    search_api_key: str | None = None,
    search_query: str | None = None,
    search_provider: str | None = None,
    model_name: str = FACT_CHECK_FALLBACK_MODEL,
    temperature: float = FACT_CHECK_TEMPERATURE,
    max_tokens: int = FACT_CHECK_MAX_TOKENS,
    max_retries: int = FACT_CHECK_MAX_RETRIES,
    backoff_seconds: int = FACT_CHECK_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """
    Fact-check one summary transcript record.

    Arguments:
        record (dict[str, Any]): Summary transcript record.
        api_key (str): DeepSeek API key for the closed-book chat path.
        search_api_key (str | None): Optional DeepSeek API key used to enable Responses API web search.
        search_query (str | None): Optional explicit search query.
        search_provider (str | None): Reserved compatibility argument for caller-provided search provider.
        model_name (str): Fact-check chat model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.

    Returns:
        dict[str, Any]: Fact-check record with verdicts and usage stats.

    Example:
        >>> fact_check_transcript_record({"video_id": "1", "channel_name": "Demo", "title": "T", "summary": "", "bullets": []}, "key")["success"]
        True
    """
    summary_record = dict(record)
    summary_record["bullets"] = _normalize_fact_check_bullets(list(summary_record.get("bullets") or []))
    bullets = summary_record.get("bullets") or []
    checkable_bullets = [bullet for bullet in bullets if not bullet.get("is_opinion", False)]

    if not checkable_bullets:
        fact_check_record = merge_fact_check_results(summary_record, FactCheckDraft())
        fact_check_record["tokens_used"] = 0
        fact_check_record["cost"] = 0.0
        fact_check_record["success"] = True
        fact_check_record["error"] = None

        return fact_check_record

    if search_api_key:
        try:
            draft, citations, input_tokens, output_tokens, total_tokens = fact_check_transcript_record_with_web_search(
                summary_record,
                search_api_key,
                search_query=search_query,
                search_provider=search_provider,
                model_name=FACT_CHECK_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            fact_check_record = merge_fact_check_results(summary_record, draft, default_sources=citations)
            fact_check_record["tokens_used"] = total_tokens
            fact_check_record["cost"] = compute_cost(input_tokens, output_tokens)
            fact_check_record["success"] = True
            fact_check_record["error"] = None

            return fact_check_record
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Web-search fact checking failed for %s; falling back to closed-book mode: %s",
                summary_record.get("video_id", "unknown"),
                exc,
            )

    fact_check_payload = {
        "video_id": summary_record.get("video_id", ""),
        "channel_name": summary_record.get("channel_name", ""),
        "title": summary_record.get("title", ""),
        "bullets": checkable_bullets,
    }
    messages = build_fact_check_messages(
        fact_check_payload,
        search_query=build_search_query(summary_record, search_query),
    )
    prompt_text = "\n\n".join(message.content for message in messages)
    encoding = build_encoding(model_name)
    fallback_input_tokens = count_tokens(encoding, prompt_text)
    llm = build_llm(api_key, model_name, temperature, max_tokens)

    try:
        response = invoke_with_retries(
            llm,
            messages,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        response_text = getattr(response, "content", "") or ""
        draft = parse_fact_check_draft(response_text)
        input_tokens, output_tokens, total_tokens = extract_usage_counts(
            response,
            fallback_input_tokens,
            response_text,
            encoding,
        )
        fact_check_record = merge_fact_check_results(summary_record, draft)
        fact_check_record["tokens_used"] = total_tokens
        fact_check_record["cost"] = compute_cost(input_tokens, output_tokens)
        fact_check_record["success"] = True
        fact_check_record["error"] = None
    except Exception as exc:  # noqa: BLE001
        logging.error("Fact checking failed for %s: %s", summary_record.get("video_id", "unknown"), exc)
        fact_check_record = merge_fact_check_results(summary_record, FactCheckDraft())
        fact_check_record["tokens_used"] = 0
        fact_check_record["cost"] = 0.0
        fact_check_record["success"] = False
        fact_check_record["error"] = str(exc)

    return fact_check_record


def fact_check_transcript_records(
    records: list[dict[str, Any]] | dict[str, Any],
    api_key: str,
    search_api_key: str | None = None,
    search_provider: str | None = None,
    model_name: str = FACT_CHECK_FALLBACK_MODEL,
    temperature: float = FACT_CHECK_TEMPERATURE,
    max_tokens: int = FACT_CHECK_MAX_TOKENS,
    max_retries: int = FACT_CHECK_MAX_RETRIES,
    backoff_seconds: int = FACT_CHECK_BACKOFF_SECONDS,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """
    Fact-check multiple summary transcript records.

    Arguments:
        records (list[dict[str, Any]]): Summary transcript records.
        api_key (str): DeepSeek API key for closed-book mode.
        search_api_key (str | None): Optional DeepSeek API key used to enable Responses API web search.
        search_provider (str | None): Reserved compatibility argument for caller-provided search provider.
        model_name (str): Fact-check chat model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.
        max_workers (int | None): Optional number of worker threads for parallel video processing.

    Returns:
        list[dict[str, Any]]: Fact-check records.

    Example:
        >>> fact_check_transcript_records([], "key")
        []
    """
    if isinstance(records, dict):
        normalized_records = [records]
    else:
        normalized_records = list(records)

    fact_check_records: list[dict[str, Any]] = []

    if max_workers is not None and max_workers > 1 and len(normalized_records) > 1:
        ordered_results: list[dict[str, Any] | None] = [None] * len(normalized_records)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    fact_check_transcript_record,
                    record,
                    api_key=api_key,
                    search_api_key=search_api_key,
                    search_provider=search_provider,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    backoff_seconds=backoff_seconds,
                ): index
                for index, record in enumerate(normalized_records)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                ordered_results[index] = future.result()

        fact_check_records = [result for result in ordered_results if result is not None]
    else:
        for record in normalized_records:
            fact_check_record = fact_check_transcript_record(
                record,
                api_key=api_key,
                search_api_key=search_api_key,
                search_provider=search_provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            fact_check_records.append(fact_check_record)

    return fact_check_records


def build_fact_check_stats(records: list[dict[str, Any]], fact_check_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build summary stats for the fact-check stage.

    Arguments:
        records (list[dict[str, Any]]): Input summary transcript records.
        fact_check_records (list[dict[str, Any]]): Fact-check results.

    Returns:
        dict[str, Any]: Aggregate fact-check stats.

    Example:
        >>> stats = build_fact_check_stats([], [])
        >>> stats["record_count"]
        0
    """
    checked_bullet_count = 0
    opinion_bullet_count = 0
    true_count = 0
    false_count = 0
    unverified_count = 0
    skipped_count = 0
    claims_verified_via_search_count = 0
    source_citation_count = 0
    confidence_total = 0.0
    confidence_count = 0
    tokens_used = 0
    cost = 0.0

    for fact_check_record in fact_check_records:
        tokens_used += int(fact_check_record.get("tokens_used") or 0)
        cost += float(fact_check_record.get("cost") or 0.0)

        for bullet in fact_check_record.get("fact_checks") or []:
            if bullet.get("is_opinion"):
                opinion_bullet_count += 1

            status = str(bullet.get("validation_status") or "")
            sources = bullet.get("sources") or []
            confidence = _normalize_confidence(bullet.get("confidence"))

            if status == "true":
                checked_bullet_count += 1
                true_count += 1
            elif status == "false":
                checked_bullet_count += 1
                false_count += 1
            elif status == "unverified":
                checked_bullet_count += 1
                unverified_count += 1
            elif status == "skipped":
                skipped_count += 1

            if status != "skipped":
                confidence_total += confidence
                confidence_count += 1

            if sources:
                claims_verified_via_search_count += 1
                source_citation_count += len(sources)

    average_confidence_score = confidence_total / confidence_count if confidence_count else 0.0

    stats = {
        "stage": "fact_checking",
        "record_count": len(records),
        "fact_checked_record_count": len(fact_check_records),
        "checked_bullet_count": checked_bullet_count,
        "opinion_bullet_count": opinion_bullet_count,
        "true_count": true_count,
        "false_count": false_count,
        "unverified_count": unverified_count,
        "skipped_count": skipped_count,
        "claims_verified_via_search_count": claims_verified_via_search_count,
        "source_citation_count": source_citation_count,
        "average_confidence_score": average_confidence_score,
        "tokens_used": tokens_used,
        "cost": cost,
    }

    return stats


def write_fact_check_records(records: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """
    Write fact-check records to individual files.

    Arguments:
        records (list[dict[str, Any]]): Fact-check records.
        output_dir (str | Path): Directory that will receive the files.

    Returns:
        list[Path]: Written file paths.
    """
    output_path = Path(output_dir)
    written_paths: list[Path] = []

    for record in records:
        filename = build_fact_check_filename(record)
        file_path = output_path / filename
        write_json(file_path, record)
        written_paths.append(file_path)

    return written_paths