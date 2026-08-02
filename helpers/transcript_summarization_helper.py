"""DeepSeek transcript summarization utilities for YouTube videos."""

from __future__ import annotations

import logging
import re
import os
import time
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from helpers.downloading_transcript_helper import build_transcript_filename, get_video_id
from helpers.output_helper import write_json


load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
FALLBACK_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2000
DEFAULT_MAX_TRANSCRIPT_TOKENS = 15_000
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1
DEFAULT_COST_PER_INPUT_TOKEN = 0.14 / 1_000_000
DEFAULT_COST_PER_OUTPUT_TOKEN = 0.28 / 1_000_000


class SummarizationDraft(BaseModel):
    """Draft structure expected from the model."""

    summary: str = ""
    key_takeaways: list[str] = Field(default_factory=list)
    main_topic: str = ""
    secondary_topics: list[str] = Field(default_factory=list)
    claims_to_fact_check: list[str] = Field(default_factory=list)
    sentiment: str = "neutral"


class TranscriptSummaryResult(BaseModel):
    """Final structured summary returned by the helper."""

    summary: str = ""
    key_takeaways: list[str] = Field(default_factory=list)
    main_topic: str = ""
    secondary_topics: list[str] = Field(default_factory=list)
    claims_to_fact_check: list[str] = Field(default_factory=list)
    sentiment: str = "neutral"
    tokens_used: int = 0
    cost: float = 0.0
    success: bool = True
    error: str | None = None


class SummaryTranscriptBullet(BaseModel):
    """Single bullet in the final summary transcript output."""

    bullet_id: int
    text: str
    is_opinion: bool | None = None


class SummaryTranscriptRecord(BaseModel):
    """Final summary transcript record written to pipeline outputs."""

    video_id: str
    channel_name: str
    title: str
    summary: str
    bullets: list[SummaryTranscriptBullet] = Field(default_factory=list)


OPINION_HINTS = (
    "i think",
    "i believe",
    "i recommend",
    "appears",
    "seems",
    "may",
    "might",
    "could",
    "possibly",
    "probably",
    "likely",
    "arguably",
    "suggests",
    "would",
    "should",
    "impossible",
    "absurd",
    "irrational",
    "clinically insane",
    "false",
    "unverified",
)


def build_encoding(model_name: str) -> tiktoken.Encoding:
    """
    Build the tokenizer.

    Arguments:
        model_name (str): Model name used to resolve token encoding.

    Returns:
        tiktoken.Encoding: Tokenizer for the selected model.

    Example:
        >>> encoding = build_encoding("deepseek-v4-flash")
        >>> encoding is not None
        True
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return encoding


def build_prompt() -> ChatPromptTemplate:
    """
    Build the chat prompt template.

    Arguments:
        None

    Returns:
        ChatPromptTemplate: Prompt template for transcript summarization.

    Example:
        >>> prompt = build_prompt()
        >>> prompt is not None
        True
    """
    system_text = (
        "You are a precise summarizer. Extract only the MOST IMPORTANT points.\n\n"
        "RULES:\n"
        "1. **SELECT** — not everything. Only include points that are central, new, or consequential.\n"
        "2. **AGGREGATE** — group related points into single bullets.\n"
        "3. **PRIORITIZE** — aim for 10-15 bullets.\n"
        "4. **SPEAKER ATTRIBUTION** — EVERY bullet MUST include the speaker's name. Use this format:\n"
        '   "[Speaker Name] + [what they said]"\n'
        '   Example: "[Alister Crooke] The US lacks a clear military objective in Iran."\n\n'
        "OUTPUT FORMAT:\n"
        "Simple numbered list. Plain text. No JSON."
    )
    human_text = (
        "Summarize this video. Include speaker attribution in every bullet. Group related ideas.\n"
        "{transcript}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", human_text),
        ]
    )

    return prompt


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    """Normalize optional metadata."""
    metadata = metadata or {}
    normalized_metadata = {
        "video_title": str(metadata.get("video_title") or "Unknown title"),
        "channel_name": str(metadata.get("channel_name") or "Unknown channel"),
        "video_url": str(metadata.get("video_url") or "Unknown URL"),
        "publish_date": str(metadata.get("publish_date") or "Unknown date"),
    }

    return normalized_metadata


def count_tokens(encoding: tiktoken.Encoding, text: str) -> int:
    """Count tokens."""
    if not text:
        token_count = 0
    else:
        token_count = len(encoding.encode(text))

    return token_count


def truncate_transcript_if_needed(
    encoding: tiktoken.Encoding,
    transcript: str,
    max_transcript_tokens: int,
) -> str:
    """Truncate long transcripts."""
    transcript_tokens = count_tokens(encoding, transcript)

    if transcript_tokens <= max_transcript_tokens:
        truncated_transcript = transcript
        return truncated_transcript

    logging.warning(
        "Transcript exceeds %s tokens (%s tokens); truncating for summarization.",
        max_transcript_tokens,
        transcript_tokens,
    )

    encoded_transcript = encoding.encode(transcript)
    truncated_tokens = encoded_transcript[:max_transcript_tokens]
    truncated_transcript = encoding.decode(truncated_tokens)

    return truncated_transcript


def build_messages(
    transcript: str,
    metadata: dict[str, Any] | None = None,
    model_name: str = DEFAULT_MODEL,
    max_transcript_tokens: int = DEFAULT_MAX_TRANSCRIPT_TOKENS,
) -> tuple[list[SystemMessage | HumanMessage], int, tiktoken.Encoding, str]:
    """Create prompt messages and estimate input tokens."""
    encoding = build_encoding(model_name)
    prompt = build_prompt()
    normalized_metadata = normalize_metadata(metadata)
    truncated_transcript = truncate_transcript_if_needed(encoding, transcript, max_transcript_tokens)

    rendered_messages = prompt.format_messages(
        video_title=normalized_metadata["video_title"],
        channel_name=normalized_metadata["channel_name"],
        publish_date=normalized_metadata["publish_date"],
        video_url=normalized_metadata["video_url"],
        transcript=truncated_transcript,
    )
    prompt_text = "\n\n".join(message.content for message in rendered_messages)
    input_tokens = count_tokens(encoding, prompt_text)

    return rendered_messages, input_tokens, encoding, truncated_transcript


def build_llm(
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> ChatOpenAI:
    """
    Build the DeepSeek chat client.

    Arguments:
        api_key (str): DeepSeek API key.
        model_name (str): DeepSeek model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.

    Returns:
        ChatOpenAI: Configured chat client.

    Example:
        >>> client = build_llm("key", "deepseek-chat", 0.3, 1000)
        >>> client is not None
        True
    """
    llm = ChatOpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return llm


def is_model_unavailable_error(model_name: str, exc: Exception) -> bool:
    """
    Check for an unavailable model.

    Arguments:
        model_name (str): Model name being used.
        exc (Exception): Exception raised by the model call.

    Returns:
        bool: True when the error looks like a missing model.

    Example:
        >>> is_model_unavailable_error("deepseek-v4-flash", Exception("404"))
        False
    """
    error_text = str(exc).lower()
    model_unavailable = (
        model_name.lower() in error_text
        and (
            "not found" in error_text
            or "model" in error_text
            or "404" in error_text
            or "unavailable" in error_text
        )
    )

    return model_unavailable


def invoke_with_retries(
    llm: ChatOpenAI,
    messages: list[SystemMessage | HumanMessage],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> Any:
    """
    Invoke the model with retries.

    Arguments:
        llm (ChatOpenAI): Configured chat client.
        messages (list[SystemMessage | HumanMessage]): Prompt messages.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.

    Returns:
        Any: Model response object.

    Example:
        >>> # This function requires a live client, so the example is illustrative.
        >>> True
        True
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            return response
        except Exception as exc:  # noqa: BLE001
            last_exception = exc

            if attempt == max_retries:
                break

            delay_seconds = backoff_seconds * (2**attempt)
            logging.warning(
                "Summarization request failed on attempt %s/%s; retrying in %s seconds: %s",
                attempt + 1,
                max_retries + 1,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("Summarization invocation failed without an exception.")


def parse_draft(response_text: str) -> SummarizationDraft:
    """
    Parse the numbered-list model response.

    Arguments:
        response_text (str): Raw model response text.

    Returns:
        SummarizationDraft: Parsed structured summary draft.

    Example:
        >>> isinstance(parse_draft("1. [Speaker] Point"), SummarizationDraft)
        True
    """
    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.lower().startswith("text"):
            cleaned_text = cleaned_text[4:]

    bullet_lines = []
    for line in cleaned_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue

        bullet_match = re.match(r"^\d+\s*[.)-]\s*(.+)$", stripped_line)
        if bullet_match:
            bullet_text = bullet_match.group(1).strip()
        else:
            bullet_text = stripped_line.lstrip("-•").strip()

        if bullet_text:
            bullet_lines.append(bullet_text)

    if not bullet_lines and cleaned_text:
        bullet_lines = [cleaned_text]

    summary_text = "\n".join(f"{index}. {bullet}" for index, bullet in enumerate(bullet_lines, 1))
    first_bullet = bullet_lines[0] if bullet_lines else ""

    draft = SummarizationDraft(
        summary=summary_text,
        key_takeaways=bullet_lines,
        main_topic=first_bullet,
        secondary_topics=[],
        claims_to_fact_check=[],
        sentiment="neutral",
    )

    return draft


def extract_usage_counts(response: Any, fallback_input_tokens: int, response_text: str, encoding: tiktoken.Encoding) -> tuple[int, int, int]:
    """
    Extract token usage.

    Arguments:
        response (Any): Model response object.
        fallback_input_tokens (int): Estimated input token count.
        response_text (str): Raw response text.
        encoding (tiktoken.Encoding): Tokenizer for fallback counting.

    Returns:
        tuple[int, int, int]: Input, output, and total token counts.

    Example:
        >>> True
        True
    """
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(usage_metadata.get("input_tokens") or fallback_input_tokens)
    output_tokens = usage_metadata.get("output_tokens")

    if output_tokens is None:
        output_tokens = count_tokens(encoding, response_text)

    output_tokens = int(output_tokens)
    total_tokens = usage_metadata.get("total_tokens")

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    total_tokens = int(total_tokens)

    return input_tokens, output_tokens, total_tokens


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Compute the estimated cost.

    Arguments:
        input_tokens (int): Input token count.
        output_tokens (int): Output token count.

    Returns:
        float: Estimated DeepSeek cost.

    Example:
        >>> compute_cost(1, 1) > 0
        True
    """
    cost = (
        input_tokens * DEFAULT_COST_PER_INPUT_TOKEN
        + output_tokens * DEFAULT_COST_PER_OUTPUT_TOKEN
    )

    return cost


def build_failure_result(error_message: str) -> TranscriptSummaryResult:
    """
    Build a failure result.

    Arguments:
        error_message (str): Error message to store in the result.

    Returns:
        TranscriptSummaryResult: Structured failure payload.

    Example:
        >>> result = build_failure_result("boom")
        >>> result.success
        False
    """
    failure_result = TranscriptSummaryResult(
        summary="",
        key_takeaways=[],
        main_topic="",
        secondary_topics=[],
        claims_to_fact_check=[],
        sentiment="neutral",
        tokens_used=0,
        cost=0.0,
        success=False,
        error=error_message,
    )

    return failure_result


def build_transcript_path(record: dict[str, Any], transcripts_dir: str) -> Path:
    """
    Build the transcript file path for a record.

    Arguments:
        record (dict[str, Any]): Transcript detection record.
        transcripts_dir (str): Directory containing transcript files.

    Returns:
        Path: Expected transcript file path.

    Example:
        >>> path = build_transcript_path({"channel_name": "Demo", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}, "transcripts")
        >>> str(path).endswith("Demo_dQw4w9WgXcQ.txt")
        True
    """
    video_id = get_video_id(record.get("url", ""))
    filename = build_transcript_filename(record.get("channel_name"), video_id)
    transcript_path = Path(transcripts_dir) / filename

    return transcript_path


def load_transcript_text(transcript_path: Path) -> str | None:
    """
    Load transcript text from disk.

    Arguments:
        transcript_path (Path): Transcript file path.

    Returns:
        str | None: Transcript text if the file exists, otherwise None.

    Example:
        >>> load_transcript_text(Path("/tmp/missing.txt")) is None
        True
    """
    if not transcript_path.exists():
        logging.warning("Transcript file not found: %s", transcript_path)
        transcript_text = None

        return transcript_text

    with open(transcript_path, "r", encoding="utf-8") as file:
        transcript_text = file.read()

    return transcript_text


def build_summary_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """
    Build metadata for transcript summarization.

    Arguments:
        record (dict[str, Any]): Transcript detection record.

    Returns:
        dict[str, Any]: Metadata dictionary for the summarizer.

    Example:
        >>> metadata = build_summary_metadata({"title": "Test", "channel_name": "Demo"})
        >>> metadata["video_title"]
        'Test'
    """
    summary_metadata = {
        "video_title": record.get("title", "Unknown title"),
        "channel_name": record.get("channel_name", "Unknown channel"),
        "video_url": record.get("url", "Unknown URL"),
        "publish_date": record.get("published_at", "Unknown date"),
    }

    return summary_metadata


def build_summary_transcript_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Build the final, simplified summary transcript payload for one record.

    Arguments:
        record (dict[str, Any]): Transcript detection record merged with summary_result.

    Returns:
        dict[str, Any]: Simplified summary transcript record.

    Example:
        >>> build_summary_transcript_record({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "channel_name": "Demo", "title": "Test", "summary_result": {"summary": "hello", "key_takeaways": ["a"]}})["video_id"]
        'dQw4w9WgXcQ'
    """
    summary_result = record.get("summary_result") or {}
    bullets = summary_result.get("key_takeaways") or []

    def infer_is_opinion(text: str) -> bool:
        normalized_text = text.lower()
        return any(hint in normalized_text for hint in OPINION_HINTS)

    summary_transcript_record = SummaryTranscriptRecord(
        video_id=get_video_id(record.get("url", "")),
        channel_name=str(record.get("channel_name") or ""),
        title=str(record.get("title") or ""),
        summary=str(summary_result.get("summary") or ""),
        bullets=[
            SummaryTranscriptBullet(
                bullet_id=index + 1,
                text=str(bullet),
                is_opinion=infer_is_opinion(str(bullet)),
            )
            for index, bullet in enumerate(bullets)
        ],
    )

    if hasattr(summary_transcript_record, "model_dump"):
        summary_transcript_payload = summary_transcript_record.model_dump()
    else:
        summary_transcript_payload = summary_transcript_record.dict()

    return summary_transcript_payload


def build_summary_transcript_filename(record: dict[str, Any]) -> str:
    """
    Build the filename for a summary transcript record.

    Arguments:
        record (dict[str, Any]): Summary transcript record.

    Returns:
        str: Filename ending in _summary.json.
    """
    video_id = str(record.get("video_id") or get_video_id(record.get("url", "")))
    channel_name = str(record.get("channel_name") or "unknown")
    transcript_filename = build_transcript_filename(channel_name, video_id)
    summary_filename = transcript_filename.replace(".txt", "_summary.json")

    return summary_filename


def write_summary_transcript_records(records: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """
    Write summary transcript records to individual files.

    Arguments:
        records (list[dict[str, Any]]): Summary transcript records.
        output_dir (str | Path): Directory that will receive the files.

    Returns:
        list[Path]: Written file paths.
    """
    output_path = Path(output_dir)
    written_paths: list[Path] = []

    for record in records:
        filename = build_summary_transcript_filename(record)
        file_path = output_path / filename
        write_json(file_path, record)
        written_paths.append(file_path)

    return written_paths


def build_summary_transcript_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build simplified summary transcript payloads for multiple records.

    Arguments:
        records (list[dict[str, Any]]): Records merged with summary results.

    Returns:
        list[dict[str, Any]]: Simplified summary transcript records.
    """
    summary_transcript_records = [build_summary_transcript_record(record) for record in records]

    return summary_transcript_records


def summarize_transcript_record(
    record: dict[str, Any],
    transcripts_dir: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_transcript_tokens: int = DEFAULT_MAX_TRANSCRIPT_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """
    Summarize one transcript record.

    Arguments:
        record (dict[str, Any]): Transcript detection record.
        transcripts_dir (str): Directory containing transcript files.
        api_key (str | None): DeepSeek API key.
        model (str): Primary DeepSeek model name.
        fallback_model (str): Backup DeepSeek model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_transcript_tokens (int): Maximum transcript token budget.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.

    Returns:
        dict[str, Any]: Record merged with its summary result.

    Example:
        >>> summarize_transcript_record({"transcript_available": False}, "transcripts")
        {'transcript_available': False, 'summary_result': {'summary': '', 'key_takeaways': [], 'main_topic': '', 'secondary_topics': [], 'claims_to_fact_check': [], 'sentiment': 'neutral', 'tokens_used': 0, 'cost': 0.0, 'success': False, 'error': 'Transcript is not available.'}}
    """
    summary_record = dict(record)

    if not record.get("transcript_available", False):
        summary_result = build_failure_result("Transcript is not available.")
    else:
        transcript_path = build_transcript_path(record, transcripts_dir)
        transcript_text = load_transcript_text(transcript_path)
        summary_metadata = build_summary_metadata(record)
        summary_result = summarize_transcript(
            transcript_text,
            metadata=summary_metadata,
            api_key=api_key,
            model=model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_transcript_tokens=max_transcript_tokens,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )

    if hasattr(summary_result, "model_dump"):
        summary_payload = summary_result.model_dump()
    else:
        summary_payload = summary_result.dict()

    summary_record["summary_result"] = summary_payload

    return summary_record


def summarize_transcript_records(
    records: list[dict[str, Any]],
    transcripts_dir: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_transcript_tokens: int = DEFAULT_MAX_TRANSCRIPT_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
    summary_limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Summarize multiple transcript records.

    Arguments:
        records (list[dict[str, Any]]): Transcript detection records.
        transcripts_dir (str): Directory containing transcript files.
        api_key (str | None): DeepSeek API key.
        model (str): Primary DeepSeek model name.
        fallback_model (str): Backup DeepSeek model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_transcript_tokens (int): Maximum transcript token budget.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.
        summary_limit (int | None): Optional limit on the number of records to summarize.

    Returns:
        list[dict[str, Any]]: Records merged with summary results.

    Example:
        >>> summarize_transcript_records([], "transcripts")
        []
    """
    summarized_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, 1):
        if summary_limit is not None and index > summary_limit:
            logging.info("Summary limit reached at %s record(s).", summary_limit)
            break

        summarized_record = summarize_transcript_record(
            record,
            transcripts_dir,
            api_key=api_key,
            model=model,
            fallback_model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_transcript_tokens=max_transcript_tokens,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        summarized_records.append(summarized_record)

    result_records = summarized_records

    return result_records


def summarize_transcript(
    transcript: str | None,
    metadata: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_transcript_tokens: int = DEFAULT_MAX_TRANSCRIPT_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> TranscriptSummaryResult:
    """
    Summarize a transcript.

    Arguments:
        transcript (str | None): Transcript text to summarize.
        metadata (dict[str, Any] | None): Optional metadata dictionary.
        api_key (str | None): DeepSeek API key.
        model (str): Primary DeepSeek model name.
        fallback_model (str): Backup DeepSeek model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_transcript_tokens (int): Maximum transcript token budget.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.

    Returns:
        TranscriptSummaryResult: Structured summary or structured failure.

    Example:
        >>> result = summarize_transcript("hello world", api_key="test")
        >>> result is not None
        True
    """
    resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

    if not resolved_api_key:
        missing_key_result = build_failure_result(
            "Missing DeepSeek API key. Set DEEPSEEK_API_KEY before running the summarizer."
        )

        return missing_key_result

    if transcript is None or not transcript.strip():
        empty_result = build_failure_result("Transcript is empty or missing.")

        return empty_result

    messages, estimated_input_tokens, encoding, _ = build_messages(
        transcript,
        metadata,
        model_name=model,
        max_transcript_tokens=max_transcript_tokens,
    )
    llm_error: Exception | None = None
    final_response = None
    model_name = model

    for current_model in (model, fallback_model):
        llm = build_llm(
            api_key=resolved_api_key,
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
        except Exception as exc:  # noqa: BLE001
            llm_error = exc
            if current_model == model and is_model_unavailable_error(model, exc):
                logging.warning(
                    "Primary model %s appears unavailable; falling back to %s.",
                    model,
                    fallback_model,
                )
                continue

            break

    if final_response is None:
        error_message = f"Summarization failed: {llm_error}"
        failure_result = build_failure_result(error_message)

        return failure_result

    response_text = getattr(final_response, "content", "") or ""

    try:
        draft = parse_draft(response_text)
    except Exception as exc:  # noqa: BLE001
        error_message = f"Failed to parse summarization output from {model_name}: {exc}"
        logging.error(error_message)
        parse_failure_result = build_failure_result(error_message)

        return parse_failure_result

    input_tokens, output_tokens, total_tokens = extract_usage_counts(
        final_response,
        estimated_input_tokens,
        response_text,
        encoding,
    )
    cost = compute_cost(input_tokens, output_tokens)

    logging.info(
        "DeepSeek summary cost: $%.6f (input=%s, output=%s, total=%s, model=%s)",
        cost,
        input_tokens,
        output_tokens,
        total_tokens,
        model_name,
    )

    success_result = TranscriptSummaryResult(
        summary=draft.summary,
        key_takeaways=draft.key_takeaways,
        main_topic=draft.main_topic,
        secondary_topics=draft.secondary_topics,
        claims_to_fact_check=draft.claims_to_fact_check,
        sentiment=draft.sentiment,
        tokens_used=total_tokens,
        cost=cost,
        success=True,
        error=None,
    )

    return success_result


def _build_mock_transcript() -> str:
    """
    Create a smoke-test transcript.

    Arguments:
        None

    Returns:
        str: Mock transcript text.

    Example:
        >>> transcript = _build_mock_transcript()
        >>> isinstance(transcript, str)
        True
    """
    mock_transcript = (
        "Welcome back. Today we discuss the state of international conflicts, "
        "how media narratives shape public opinion, and why careful fact-checking "
        "matters. The speaker argues that short clips often miss context, cites "
        "multiple reports on diplomatic tensions, and ends by urging viewers to "
        "verify claims before sharing them."
    )

    return mock_transcript


def main() -> None:
    """
    Run a smoke test.

    Arguments:
        None

    Returns:
        None

    Example:
        $ python helpers/transcript_summarization_helper.py
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mock_metadata = {
        "video_title": "Mock Transcript Test",
        "channel_name": "AgentTube Demo",
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "publish_date": "2026-07-31",
    }
    mock_transcript = _build_mock_transcript()
    result = summarize_transcript(mock_transcript, metadata=mock_metadata)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()