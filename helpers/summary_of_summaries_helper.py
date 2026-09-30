"""DeepSeek topic extraction helper for synthesizing across video summaries."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from helpers.shared_helper import load_json_object
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

    for file_path in sorted(summary_path.glob("*_summary.json")):
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        records.append(payload)

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
        channel_value = record.get("channel_name")
        if channel_value is None:
            channel_value = "Unknown channel"
        channel = str(channel_value)

        title_value = record.get("title")
        if title_value is None:
            title_value = "Unknown title"
        title = str(title_value)

        video_id_value = record.get("video_id")
        if video_id_value is None:
            video_id_value = ""
        video_id = str(video_id_value)

        bullets = record.get("bullets")
        if bullets is None:
            bullets = []

        lines = [f"VIDEO: {channel} - {title} (video_id={video_id})"]
        has_bullets = bool(bullets)
        if has_bullets == True:
            for bullet in bullets:
                if isinstance(bullet, dict):
                    speaker_value = bullet.get("speaker")
                    if speaker_value is None:
                        speaker_value = "Unknown"
                    speaker = str(speaker_value).strip()
                    if speaker == "":
                        speaker = "Unknown"

                    text_value = bullet.get("text")
                    if text_value is None:
                        text_value = ""
                    text = str(text_value).strip()
                else:
                    speaker = "Unknown"
                    text = str(bullet).strip()

                has_text = bool(text)
                if has_text == True:
                    lines.append(f"  [{speaker}] {text}")
        else:
            summary_value = record.get("summary")
            if summary_value is None:
                summary_value = ""
            summary_text = str(summary_value).strip()

            has_summary_text = bool(summary_text)
            if has_summary_text == True:
                lines.append(f"  [Unknown] {summary_text}")

        record_blocks.append("\n".join(lines))

    has_record_blocks = bool(record_blocks)
    if has_record_blocks == True:
        corpus_text = "\n\n".join(record_blocks)
    else:
        corpus_text = "No summary records provided."

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
        cleaned_text = cleaned_text.strip("`")
        has_json_prefix = cleaned_text.lower().startswith("json")
        if has_json_prefix == True:
            cleaned_text = cleaned_text[4:]

        cleaned_text = cleaned_text.strip()

    payload: Any
    payload = json.loads(cleaned_text)

    if not isinstance(payload, dict):
        payload = {}

    raw_topics = payload.get("topics")
    if raw_topics is None:
        raw_topics = []

    topics: list[Topic] = []

    for raw_topic in raw_topics:
        if not isinstance(raw_topic, dict):
            continue

        name_value = raw_topic.get("name")
        if name_value is None:
            name_value = ""
        name = str(name_value).strip()

        has_no_name = not name
        if has_no_name == True:
            continue

        raw_perspectives = raw_topic.get("perspectives")
        if raw_perspectives is None:
            raw_perspectives = []

        perspectives: list[Perspective] = []
        for raw_perspective in raw_perspectives:
            if not isinstance(raw_perspective, dict):
                continue

            perspective_text_value = raw_perspective.get("text")
            if perspective_text_value is None:
                perspective_text_value = ""
            perspective_text = str(perspective_text_value).strip()

            has_no_perspective_text = not perspective_text
            if has_no_perspective_text == True:
                continue

            speaker_value = raw_perspective.get("speaker")
            if speaker_value is None:
                speaker_value = "Unknown"
            speaker = str(speaker_value).strip()
            if speaker == "":
                speaker = "Unknown"

            channel_value = raw_perspective.get("channel")
            if channel_value is None:
                channel_value = ""
            channel = str(channel_value).strip()

            video_id_value = raw_perspective.get("video_id")
            if video_id_value is None:
                video_id_value = ""
            video_id = str(video_id_value).strip()

            perspective = Perspective(
                speaker=speaker,
                channel=channel,
                video_id=video_id,
                text=perspective_text,
            )
            perspectives.append(perspective)

        raw_themes = raw_topic.get("themes")
        if raw_themes is None:
            raw_themes = []

        themes: list[Theme] = []
        for raw_theme in raw_themes:
            if not isinstance(raw_theme, dict):
                continue

            theme_name_value = raw_theme.get("name")
            if theme_name_value is None:
                theme_name_value = ""
            theme_name = str(theme_name_value).strip()

            has_no_theme_name = not theme_name
            if has_no_theme_name == True:
                continue

            raw_supporters = raw_theme.get("supporting_speakers")
            if raw_supporters is None:
                raw_supporters = []

            supporting_speakers = []
            for speaker in raw_supporters:
                cleaned_speaker = str(speaker).strip()
                has_cleaned_speaker = bool(cleaned_speaker)
                if has_cleaned_speaker == True:
                    supporting_speakers.append(cleaned_speaker)

            description_value = raw_theme.get("description")
            if description_value is None:
                description_value = ""
            description = str(description_value).strip()

            theme = Theme(
                name=theme_name,
                description=description,
                supporting_speakers=supporting_speakers,
            )
            themes.append(theme)

        consensus_value = raw_topic.get("consensus")
        if consensus_value is None:
            consensus_value = "mixed"
        consensus = str(consensus_value).strip().lower()

        if consensus not in {"high", "medium", "low", "mixed"}:
            consensus = "mixed"

        topic_id_value = raw_topic.get("topic_id")
        if topic_id_value is None:
            topic_id_value = len(topics) + 1
        topic_id = int(topic_id_value)

        topic_description_value = raw_topic.get("description")
        if topic_description_value is None:
            topic_description_value = ""
        topic_description = str(topic_description_value).strip()

        topic = Topic(
            topic_id=topic_id,
            name=name,
            description=topic_description,
            perspectives=perspectives,
            themes=themes,
            consensus=consensus,
        )
        topics.append(topic)

    topics_draft = TopicsDraft(topics=topics)

    return topics_draft


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
    has_no_summary_records = not summary_records
    if has_no_summary_records == True:
        empty_topics = TopicsDraft(topics=[])
        empty_result = (empty_topics, 0, 0.0)

        return empty_result

    messages, prompt_text = build_topics_messages(summary_records)
    encoding = build_encoding(model)
    fallback_input_tokens = count_tokens(encoding, prompt_text)

    selected_model = model
    model_name = model

    # Keep signature compatibility while using a single explicit model path.
    if fallback_model == "":
        selected_model = model

    llm = build_llm(
        api_key=api_key,
        model_name=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    final_response = invoke_with_retries(
        llm,
        messages,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    model_name = selected_model

    response_text = getattr(final_response, "content", "")
    if response_text is None:
        response_text = ""

    draft = parse_topics_draft(response_text)

    input_tokens, output_tokens, total_tokens = extract_usage_counts(
        final_response,
        fallback_input_tokens,
        response_text,
        encoding,
    )
    cost = compute_cost(input_tokens, output_tokens)

    topic_count = len(draft.topics)
    perspective_count = 0
    theme_count = 0
    for topic in draft.topics:
        perspective_count += len(topic.perspectives)
        theme_count += len(topic.themes)

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

    result = (draft, total_tokens, cost)

    return result


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
    perspective_count = 0
    for topic in topics_draft.topics:
        perspective_count += len(topic.perspectives)

    topic_payloads = []
    for topic in topics_draft.topics:
        has_model_dump = hasattr(topic, "model_dump")
        if has_model_dump == True:
            topic_payload = topic.model_dump()
        else:
            topic_payload = topic.dict()

        topic_payloads.append(topic_payload)

    payload = {
        "topics": topic_payloads,
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
    perspective_count = 0
    theme_count = 0
    for topic in topics_draft.topics:
        perspective_count += len(topic.perspectives)
        theme_count += len(topic.themes)

    stats = {
        "stage": "topic_extraction",
        "summary_record_count": summary_record_count,
        "topic_count": topic_count,
        "perspective_count": perspective_count,
        "theme_count": theme_count,
        "generated_at": datetime.now().isoformat(),
    }

    return stats
