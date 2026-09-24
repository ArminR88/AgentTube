"""DeepSeek topic extraction helper for synthesizing across video summaries."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
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


class Perspective(BaseModel):
    """One speaker perspective on a topic."""

    speaker: str = "Unknown"
    channel: str = ""
    video_id: str = ""
    text: str = ""


class Theme(BaseModel):
    """A recurring cross-perspective theme."""

    name: str
    description: str = ""
    supporting_speakers: list[str] = Field(default_factory=list)


class Topic(BaseModel):
    """A synthesized topic with perspectives and themes."""

    topic_id: int
    name: str
    description: str = ""
    perspectives: list[Perspective] = Field(default_factory=list)
    themes: list[Theme] = Field(default_factory=list)
    consensus: str = "mixed"


class TopicsDraft(BaseModel):
    """Topic extraction output draft."""

    topics: list[Topic] = Field(default_factory=list)


def build_topics_prompt() -> str:
    """
    Build the instruction block for topic extraction.

    Arguments:
        None

    Returns:
        str: Prompt instructions for extracting topics, perspectives, and themes.

    Example:
        >>> prompt = build_topics_prompt()
        >>> "VALID JSON" in prompt
        True
    """
    prompt = (
        "You are a synthesis analyst. Read multiple transcript summaries and extract a "
        "structured Topics -> Perspectives -> Themes digest.\n\n"
        "REQUIREMENTS:\n"
        "1. Extract 8-12 distinct TOPICS.\n"
        "2. For each topic, include 5-7 PERSPECTIVES.\n"
        "3. Attribute each perspective to the PERSON speaking, not the channel.\n"
        "4. For each topic, include 2-4 THEMES.\n"
        "5. Each theme must include supporting_speakers as a list of speaker names.\n"
        "6. consensus must be one of: high, medium, low, mixed.\n"
        "7. Skip generic or redundant topics.\n\n"
        "IMPORTANT - TOPIC BALANCE: No single topic may contain more than 3 perspectives. If a topic would need 4 or more, split it into subtopics with distinct names. Example: instead of one 'Ukraine and Europe' topic with 30 perspectives, produce separate topics such as 'Ukraine's Negotiation Prospects', 'European Leaders' Denial', 'Suppression of Dissent in the UK', 'Russia's Post-2014 Identity', 'The Multipolar Transition'. Aim for 10-15 topics total, each with 2-3 perspectives.\n\n"
        "OUTPUT:\n"
        "Return VALID JSON ONLY with this exact shape:\n"
        "{\"topics\": [{\"topic_id\": 1, \"name\": \"...\", \"description\": \"...\",\n"
        "            \"consensus\": \"mixed\",\n"
        "            \"perspectives\": [{\"speaker\": \"...\", \"channel\": \"...\",\n"
        "                              \"video_id\": \"...\", \"text\": \"...\"}],\n"
        "            \"themes\": [{\"name\": \"...\", \"description\": \"...\",\n"
        "                        \"supporting_speakers\": [\"...\"]}]}]}\n\n"
        "Return ONLY valid JSON. No markdown, no explanation."
    )

    return prompt


def load_summary_files(summary_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load all summary JSON files from a directory.

    Arguments:
        summary_dir (str | Path): Directory containing *_summary.json files.

    Returns:
        list[dict[str, Any]]: Loaded summary record payloads.

    Example:
        >>> records = load_summary_files("output_agenttube/2026-08-04/transcript_summary")
        >>> isinstance(records, list)
        True
    """
    summary_path = Path(summary_dir)
    records: list[dict[str, Any]] = []

    if not summary_path.exists():
        logging.warning("Summary directory not found: %s", summary_path)
        return records

    for file_path in sorted(summary_path.glob("*_summary.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                records.append(payload)
            else:
                logging.warning("Skipping malformed summary file (not object): %s", file_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed loading summary file %s: %s", file_path, exc)

    return records


def build_topics_messages(
    summary_records: list[dict[str, Any]],
) -> tuple[list[SystemMessage | HumanMessage], str]:
    """
    Build model messages for topic extraction.

    Arguments:
        summary_records (list[dict[str, Any]]): Summary transcript records.

    Returns:
        tuple[list[SystemMessage | HumanMessage], str]: Rendered messages and joined prompt text.

    Example:
        >>> messages, prompt_text = build_topics_messages([])
        >>> len(messages) == 2
        True
    """
    record_blocks: list[str] = []

    for record in summary_records:
        channel = str(record.get("channel_name") or "Unknown channel")
        title = str(record.get("title") or "Unknown title")
        video_id = str(record.get("video_id") or "")
        bullets = record.get("bullets") or []

        lines = [f"VIDEO: {channel} - {title} (video_id={video_id})"]
        if bullets:
            for bullet in bullets:
                if isinstance(bullet, dict):
                    speaker = str(bullet.get("speaker") or "Unknown").strip() or "Unknown"
                    text = str(bullet.get("text") or "").strip()
                else:
                    speaker = "Unknown"
                    text = str(bullet).strip()
                if text:
                    lines.append(f"  [{speaker}] {text}")
        else:
            summary_text = str(record.get("summary") or "").strip()
            if summary_text:
                lines.append(f"  [Unknown] {summary_text}")

        record_blocks.append("\n".join(lines))

    corpus_text = "\n\n".join(record_blocks) if record_blocks else "No summary records provided."

    system_text = build_topics_prompt()
    human_text = (
        "Extract topics from the following summary corpus.\n"
        "Use only information present in the source text.\n\n"
        f"{corpus_text}"
    )

    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ]
    prompt_text = system_text + "\n\n" + human_text

    return messages, prompt_text


def parse_topics_draft(response_text: str) -> TopicsDraft:
    """
    Parse the model response into a validated topic draft.

    Arguments:
        response_text (str): Raw model output text.

    Returns:
        TopicsDraft: Parsed topics draft.

    Example:
        >>> draft = parse_topics_draft('{"topics": []}')
        >>> isinstance(draft, TopicsDraft)
        True
    """
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```json\s*", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"^```", "", cleaned_text)
        cleaned_text = re.sub(r"```$", "", cleaned_text).strip()

    payload: Any
    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned_text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

    if not isinstance(payload, dict):
        payload = {}

    raw_topics = payload.get("topics") or []
    topics: list[Topic] = []

    for raw_topic in raw_topics:
        if not isinstance(raw_topic, dict):
            continue

        name = str(raw_topic.get("name") or "").strip()
        if not name:
            continue

        raw_perspectives = raw_topic.get("perspectives") or []
        perspectives: list[Perspective] = []
        for raw_perspective in raw_perspectives:
            if not isinstance(raw_perspective, dict):
                continue
            perspective_text = str(raw_perspective.get("text") or "").strip()
            if not perspective_text:
                continue
            perspective = Perspective(
                speaker=str(raw_perspective.get("speaker") or "Unknown").strip() or "Unknown",
                channel=str(raw_perspective.get("channel") or "").strip(),
                video_id=str(raw_perspective.get("video_id") or "").strip(),
                text=perspective_text,
            )
            perspectives.append(perspective)

        raw_themes = raw_topic.get("themes") or []
        themes: list[Theme] = []
        for raw_theme in raw_themes:
            if not isinstance(raw_theme, dict):
                continue
            theme_name = str(raw_theme.get("name") or "").strip()
            if not theme_name:
                continue
            raw_supporters = raw_theme.get("supporting_speakers") or []
            supporting_speakers = [
                str(speaker).strip()
                for speaker in raw_supporters
                if str(speaker).strip()
            ]
            theme = Theme(
                name=theme_name,
                description=str(raw_theme.get("description") or "").strip(),
                supporting_speakers=supporting_speakers,
            )
            themes.append(theme)

        consensus = str(raw_topic.get("consensus") or "mixed").strip().lower()
        if consensus not in {"high", "medium", "low", "mixed"}:
            consensus = "mixed"

        topic = Topic(
            topic_id=int(raw_topic.get("topic_id") or (len(topics) + 1)),
            name=name,
            description=str(raw_topic.get("description") or "").strip(),
            perspectives=perspectives,
            themes=themes,
            consensus=consensus,
        )
        topics.append(topic)

    return TopicsDraft(topics=topics)


def extract_topics(
    summary_records: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 6000,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> tuple[TopicsDraft, int, float]:
    """
    Extract structured topics from summary records.

    Arguments:
        summary_records (list[dict[str, Any]]): Loaded summary records.
        api_key (str): DeepSeek API key.
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.

    Returns:
        tuple[TopicsDraft, int, float]: Parsed topics, tokens used, and cost.

    Example:
        >>> draft, tokens, cost = extract_topics([], api_key="demo")
        >>> draft.topics
        []
    """
    if not summary_records:
        return TopicsDraft(topics=[]), 0, 0.0

    messages, prompt_text = build_topics_messages(summary_records)
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
        except Exception as exc:  # noqa: BLE001
            llm_error = exc
            if current_model == model:
                logging.warning("Primary model %s failed; falling back to %s.", model, fallback_model)
                continue
            break

    if final_response is None:
        logging.error("Topic extraction failed: %s", llm_error)
        return TopicsDraft(topics=[]), 0, 0.0

    response_text = getattr(final_response, "content", "") or ""
    draft = parse_topics_draft(response_text)

    input_tokens, output_tokens, total_tokens = extract_usage_counts(
        final_response,
        fallback_input_tokens,
        response_text,
        encoding,
    )
    cost = compute_cost(input_tokens, output_tokens)

    topic_count = len(draft.topics)
    perspective_count = sum(len(topic.perspectives) for topic in draft.topics)
    theme_count = sum(len(topic.themes) for topic in draft.topics)

    logging.info(
        "Topic extraction cost: $%.6f (input=%s, output=%s, total=%s, model=%s, topics=%s, perspectives=%s, themes=%s)",
        cost,
        input_tokens,
        output_tokens,
        total_tokens,
        model_name,
        topic_count,
        perspective_count,
        theme_count,
    )

    return draft, total_tokens, cost


def write_topics(
    topics_draft: TopicsDraft,
    output_dir: str | Path,
    tokens_used: int = 0,
    cost: float = 0.0,
) -> Path:
    """
    Write extracted topics to topics.json.

    Arguments:
        topics_draft (TopicsDraft): Parsed topics draft.
        output_dir (str | Path): Destination directory.
        tokens_used (int): Token usage for extraction.
        cost (float): Estimated extraction cost.

    Returns:
        Path: Path to written topics.json.

    Example:
        >>> path = write_topics(TopicsDraft(topics=[]), "tmp/topics")
        >>> path.name
        'topics.json'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    topic_count = len(topics_draft.topics)
    perspective_count = sum(len(topic.perspectives) for topic in topics_draft.topics)

    payload = {
        "topics": [
            topic.model_dump() if hasattr(topic, "model_dump") else topic.dict()
            for topic in topics_draft.topics
        ],
        "metadata": {
            "topic_count": topic_count,
            "perspective_count": perspective_count,
            "tokens_used": tokens_used,
            "cost": cost,
            "generated_at": datetime.now().isoformat(),
        },
    }

    topics_path = output_path / "topics.json"
    write_json(topics_path, payload)

    return topics_path


def load_topics(topics_file: str | Path) -> dict[str, Any]:
    """
    Load topics JSON payload from disk.

    Arguments:
        topics_file (str | Path): Path to topics.json.

    Returns:
        dict[str, Any]: Loaded topics payload.

    Example:
        >>> payload = {"topics": []}
        >>> isinstance(payload, dict)
        True
    """
    topics_path = Path(topics_file)

    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}")

    with open(topics_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return payload


def build_topics_stats(summary_record_count: int, topics_draft: TopicsDraft) -> dict[str, Any]:
    """
    Build stage stats for topic extraction.

    Arguments:
        summary_record_count (int): Number of input summary records.
        topics_draft (TopicsDraft): Extracted topics draft.

    Returns:
        dict[str, Any]: Topic extraction stage statistics.

    Example:
        >>> stats = build_topics_stats(2, TopicsDraft(topics=[]))
        >>> stats["stage"]
        'topic_extraction'
    """
    topic_count = len(topics_draft.topics)
    perspective_count = sum(len(topic.perspectives) for topic in topics_draft.topics)
    theme_count = sum(len(topic.themes) for topic in topics_draft.topics)

    stats = {
        "stage": "topic_extraction",
        "summary_record_count": summary_record_count,
        "topic_count": topic_count,
        "perspective_count": perspective_count,
        "theme_count": theme_count,
        "generated_at": datetime.now().isoformat(),
    }

    return stats
