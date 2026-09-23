"""DeepSeek summary-of-summaries helper for extracting unique claims across videos."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers.output_helper import write_json
from helpers.transcript_summarization_helper import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    build_encoding,
    build_llm,
    compute_cost,
    count_tokens,
    extract_usage_counts,
    invoke_with_retries,
)


class UniqueClaim(BaseModel):
    """A single unique claim extracted from video summaries."""

    claim_id: int
    text: str
    source_videos: list[str] = Field(default_factory=list)
    source_bullets: list[int] = Field(default_factory=list)


class ClaimsDraft(BaseModel):
    """Draft structure expected from the model."""

    claims: list[UniqueClaim] = Field(default_factory=list)


def build_claims_prompt() -> str:
    """
    Build the instruction block for extracting unique claims from summaries.

    Returns:
        str: Prompt instructions for the model.

    Example:
        >>> "unique" in build_claims_prompt()
        True
    """
    current_date = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    prompt = (
        "You are a claim extraction expert.\n\n"
        f"Current date: {current_date}\n\n"
        "TASK: Extract ~20 unique, factual claims from the video summaries below.\n\n"
        "RULES:\n"
        "1. EXTRACT only factual claims (not opinions, not rhetorical questions).\n"
        "2. DEDUPLICATE - if the same claim appears in multiple videos, keep it once.\n"
        "3. AGGREGATE - merge similar claims into one clear statement.\n"
        "4. AIM for approximately 20 unique claims across all videos.\n"
        "5. Each claim must be a complete, standalone sentence.\n"
        "6. Track which videos and bullet IDs support each claim.\n\n"
        "OUTPUT FORMAT (JSON only):\n"
        "{\n"
        '  "claims": [\n'
        '    {"claim_id": 1, "text": "The US has no clear military objective in Iran.", "source_videos": ["video1", "video2"], "source_bullets": [1, 3]},\n'
        '    {"claim_id": 2, "text": "China\'s trade surplus reached $500B in 2025.", "source_videos": ["video3"], "source_bullets": [2]}\n'
        "  ]\n"
        "}\n\n"
        "Return ONLY valid JSON. No markdown, no explanation."
    )

    return prompt


def load_summary_files(summary_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load all _summary.json files from a directory.

    Arguments:
        summary_dir (str | Path): Directory containing summary JSON files.

    Returns:
        list[dict[str, Any]]: List of summary records.

    Example:
        >>> summaries = load_summary_files("output_agenttube/2026-08-04/transcript_summary")
        >>> len(summaries) > 0
        True
    """
    summary_path = Path(summary_dir)
    summary_records: list[dict[str, Any]] = []

    if not summary_path.exists():
        logging.warning("Summary directory not found: %s", summary_path)
        return summary_records

    for file_path in summary_path.glob("*_summary.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                record = json.load(file)
                summary_records.append(record)
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Failed to load %s: %s", file_path.name, exc)

    return summary_records


def build_claims_messages(
    summary_records: list[dict[str, Any]],
) -> tuple[list[SystemMessage | HumanMessage], str]:
    """
    Build messages for claims extraction.

    Arguments:
        summary_records (list[dict[str, Any]]): Summary records.

    Returns:
        tuple[list[SystemMessage | HumanMessage], str]: Messages and prompt text.
    """
    # Build a condensed representation of all summaries
    summaries_text = []
    for record in summary_records:
        video_title = record.get("title", "Unknown")
        channel = record.get("channel_name", "Unknown")
        bullets = record.get("bullets", [])
        bullet_texts = [f"  {b.get('bullet_id')}. {b.get('text')}" for b in bullets]
        summaries_text.append(f"VIDEO: {channel} - {video_title}\n" + "\n".join(bullet_texts))

    combined_text = "\n\n".join(summaries_text)

    system_text = build_claims_prompt()
    human_text = f"Extract unique claims from these summaries:\n\n{combined_text}"

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ]

    prompt_text = system_text + "\n\n" + human_text

    return messages, prompt_text


def parse_claims_draft(response_text: str) -> ClaimsDraft:
    """
    Parse the model response into claims.

    Arguments:
        response_text (str): Raw model response.

    Returns:
        ClaimsDraft: Parsed claims draft.

    Example:
        >>> draft = parse_claims_draft('{"claims": [{"claim_id": 1, "text": "Test", "source_videos": ["a"], "source_bullets": [1]}]}')
        >>> len(draft.claims)
        1
    """
    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.lower().startswith("json"):
            cleaned_text = cleaned_text[4:]

    try:
        draft = ClaimsDraft.model_validate_json(cleaned_text)
    except Exception as exc:
        logging.warning("Failed to parse claims draft: %s", exc)
        try:
            payload = json.loads(cleaned_text)
            claims_data = payload.get("claims", [])
            draft = ClaimsDraft(claims=claims_data)
        except json.JSONDecodeError:
            draft = ClaimsDraft(claims=[])

    return draft


def extract_unique_claims(
    summary_records: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> tuple[ClaimsDraft, int, float]:
    """
    Extract unique claims from summary records.

    Arguments:
        summary_records (list[dict[str, Any]]): Summary records.
        api_key (str): DeepSeek API key.
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay.

    Returns:
        tuple[ClaimsDraft, int, float]: Claims draft, tokens used, cost.
    """
    if not summary_records:
        return ClaimsDraft(claims=[]), 0, 0.0

    messages, prompt_text = build_claims_messages(summary_records)
    encoding = build_encoding(model)
    fallback_input_tokens = count_tokens(encoding, prompt_text)

    llm_error: Exception | None = None
    final_response = None
    model_name = model

    for current_model in (model, fallback_model):
        llm = build_llm(
            api_key=api_key,
            model_name=current_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            final_response = invoke_with_retries(
                llm,
                messages,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            model_name = current_model
            llm_error = None
            break
        except Exception as exc:
            llm_error = exc
            if current_model == model:
                logging.warning("Primary model %s failed; falling back to %s.", model, fallback_model)
                continue
            break

    if final_response is None:
        logging.error("Claims extraction failed: %s", llm_error)
        return ClaimsDraft(claims=[]), 0, 0.0

    response_text = getattr(final_response, "content", "") or ""
    draft = parse_claims_draft(response_text)

    input_tokens, output_tokens, total_tokens = extract_usage_counts(
        final_response,
        fallback_input_tokens,
        response_text,
        encoding,
    )
    cost = compute_cost(input_tokens, output_tokens)

    logging.info(
        "Claims extraction cost: $%.6f (input=%s, output=%s, total=%s, model=%s, claims=%s)",
        cost,
        input_tokens,
        output_tokens,
        total_tokens,
        model_name,
        len(draft.claims),
    )

    return draft, total_tokens, cost


def write_claims(
    claims_draft: ClaimsDraft,
    output_dir: str | Path,
    tokens_used: int = 0,
    cost: float = 0.0,
) -> Path:
    """
    Write claims to JSON file.

    Arguments:
        claims_draft (ClaimsDraft): Claims draft.
        output_dir (str | Path): Output directory.
        tokens_used (int): Token usage.
        cost (float): Cost in USD.

    Returns:
        Path: Path of written file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "claims": [claim.model_dump() if hasattr(claim, "model_dump") else claim.dict() for claim in claims_draft.claims],
        "metadata": {
            "claim_count": len(claims_draft.claims),
            "tokens_used": tokens_used,
            "cost": cost,
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        },
    }

    file_path = output_path / "claims.json"
    write_json(file_path, payload)

    return file_path


def load_claims(claims_file: str | Path) -> dict[str, Any]:
    """
    Load claims from JSON file.

    Arguments:
        claims_file (str | Path): Path to claims.json.

    Returns:
        dict[str, Any]: Claims payload.

    Example:
        >>> claims = load_claims("output_agenttube/2026-08-04/claims/claims.json")
        >>> "claims" in claims
        True
    """
    claims_path = Path(claims_file)

    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    with open(claims_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return payload


def build_claims_stats(summary_record_count: int, claims_draft: ClaimsDraft) -> dict[str, Any]:
    """
    Build stats for the summary-of-summaries stage.

    Arguments:
        summary_record_count (int): Number of summary records processed.
        claims_draft (ClaimsDraft): Extracted claims.

    Returns:
        dict[str, Any]: Stage stats.

    Example:
        >>> stats = build_claims_stats(5, ClaimsDraft(claims=[]))
        >>> stats["claim_count"]
        0
    """
    stats = {
        "stage": "summary_of_summaries",
        "summary_record_count": summary_record_count,
        "claim_count": len(claims_draft.claims),
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }

    return stats