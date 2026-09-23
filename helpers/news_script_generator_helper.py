"""DeepSeek news script generator for fact-checked claims."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from helpers.output_helper import write_json, write_text
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


def build_script_prompt() -> str:
    """
    Build the instruction block for news script generation.

    Returns:
        str: Prompt instructions for the model.

    Example:
        >>> "NPR" in build_script_prompt()
        True
    """
    current_date = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    prompt = (
        "You are a professional news scriptwriter for NPR/CNN style broadcasts.\n\n"
        f"Current date: {current_date}\n\n"
        "TASK: Write a flowing 15-minute news script based on fact-checked claims.\n\n"
        "STYLE REQUIREMENTS:\n"
        "1. NPR/CNN journalism tone - authoritative, clear, narrative.\n"
        "2. NO bullet points - write in flowing paragraphs.\n"
        "3. Structure: News intro → Context → Each claim with verdict → Conclusion.\n"
        "4. For each claim, clearly state: 'Claim: ... Verdict: True/False/Unverified'.\n"
        "5. Cite sources naturally: 'According to a report from...'\n"
        "6. Target length: ~4500 words (15 minutes of spoken audio).\n"
        "7. Include a host introduction and closing.\n\n"
        "OUTPUT FORMAT:\n"
        "Plain text script with clear section breaks using [NEWS INTRO], [CONTEXT], [CLAIM 1], etc.\n"
        "No markdown, no JSON, just the script.\n\n"
        "Return ONLY the script text."
    )

    return prompt


def build_script_messages(
    claims_data: dict[str, Any],
) -> tuple[list[SystemMessage | HumanMessage], str]:
    """
    Build messages for script generation.

    Arguments:
        claims_data (dict[str, Any]): Claims with fact-check verdicts.

    Returns:
        tuple[list[SystemMessage | HumanMessage], str]: Messages and prompt text.
    """
    claims = claims_data.get("claims", [])
    claims_text = []

    for claim in claims:
        claim_text = claim.get("text", "")
        verdict = claim.get("validation_status", "unverified")
        note = claim.get("note", "")
        sources = claim.get("sources", [])
        confidence = claim.get("confidence", 0.0)

        claims_text.append(
            f"Claim: {claim_text}\n"
            f"Verdict: {verdict}\n"
            f"Note: {note}\n"
            f"Sources: {', '.join(sources) if sources else 'No sources available'}\n"
            f"Confidence: {confidence:.0%}\n"
        )

    combined_claims = "\n---\n".join(claims_text)

    system_text = build_script_prompt()
    human_text = f"Write a news script based on these fact-checked claims:\n\n{combined_claims}"

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ]

    prompt_text = system_text + "\n\n" + human_text

    return messages, prompt_text


def generate_news_script(
    claims_data: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 8000,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> tuple[str, int, float]:
    """
    Generate a news script from fact-checked claims.

    Arguments:
        claims_data (dict[str, Any]): Claims with fact-check verdicts.
        api_key (str): DeepSeek API key.
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature (higher for creativity).
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay.

    Returns:
        tuple[str, int, float]: Script text, tokens used, cost.
    """
    if not claims_data.get("claims"):
        return "No claims available to generate script.", 0, 0.0

    messages, prompt_text = build_script_messages(claims_data)
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
        logging.error("Script generation failed: %s", llm_error)
        return "Script generation failed.", 0, 0.0

    script_text = getattr(final_response, "content", "") or ""

    input_tokens, output_tokens, total_tokens = extract_usage_counts(
        final_response,
        fallback_input_tokens,
        script_text,
        encoding,
    )
    cost = compute_cost(input_tokens, output_tokens)

    logging.info(
        "Script generation cost: $%.6f (input=%s, output=%s, total=%s, model=%s, words=%s)",
        cost,
        input_tokens,
        output_tokens,
        total_tokens,
        model_name,
        len(script_text.split()),
    )

    return script_text, total_tokens, cost


def write_news_script(
    script_text: str,
    output_dir: str | Path,
    claims_data: dict[str, Any],
    tokens_used: int = 0,
    cost: float = 0.0,
) -> tuple[Path, Path]:
    """
    Write news script to text and JSON files.

    Arguments:
        script_text (str): Generated script.
        output_dir (str | Path): Output directory.
        claims_data (dict[str, Any]): Original claims data.
        tokens_used (int): Token usage.
        cost (float): Cost in USD.

    Returns:
        tuple[Path, Path]: Paths of text and JSON files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write text file
    text_path = output_path / "news_script.txt"
    write_text(text_path, script_text)

    # Write JSON with metadata
    json_payload = {
        "script": script_text,
        "metadata": {
            "word_count": len(script_text.split()),
            "character_count": len(script_text),
            "tokens_used": tokens_used,
            "cost": cost,
            "claim_count": len(claims_data.get("claims", [])),
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        },
        "claims": claims_data.get("claims", []),
    }

    json_path = output_path / "news_script.json"
    write_json(json_path, json_payload)

    return text_path, json_path


def load_fact_checked_claims(claims_file: str | Path) -> dict[str, Any]:
    """
    Load fact-checked claims from JSON file.

    Arguments:
        claims_file (str | Path): Path to claims_fact_checked.json.

    Returns:
        dict[str, Any]: Claims payload with verdicts.

    Example:
        >>> claims = load_fact_checked_claims("output_agenttube/2026-08-04/fact_checked_claims/claims_fact_checked.json")
        >>> "claims" in claims
        True
    """
    claims_path = Path(claims_file)

    if not claims_path.exists():
        raise FileNotFoundError(f"Fact-checked claims file not found: {claims_path}")

    with open(claims_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return payload


def build_script_stats(claims_count: int, word_count: int, tokens_used: int, cost: float) -> dict[str, Any]:
    """
    Build stats for the script generation stage.

    Arguments:
        claims_count (int): Number of claims in the script.
        word_count (int): Word count of the script.
        tokens_used (int): Token usage.
        cost (float): Cost in USD.

    Returns:
        dict[str, Any]: Stage stats.

    Example:
        >>> stats = build_script_stats(20, 4500, 5000, 0.02)
        >>> stats["stage"]
        'news_script_generation'
    """
    stats = {
        "stage": "news_script_generation",
        "claim_count": claims_count,
        "word_count": word_count,
        "tokens_used": tokens_used,
        "cost": cost,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
    }

    return stats