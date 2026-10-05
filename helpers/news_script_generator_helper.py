"""DeepSeek perspective digest generator for topic-based summaries."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from helpers.output_helper import get_run_date, write_json, write_text
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


SPOKEN_CLOSING = "These were today's highlights."

ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 21: "twenty-first",
    22: "twenty-second", 23: "twenty-third", 24: "twenty-fourth",
    25: "twenty-fifth", 26: "twenty-sixth", 27: "twenty-seventh",
    28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth",
    31: "thirty-first",
}


def build_spoken_opening() -> str:
    """
    Build the spoken opening sentence with today's date spelled out.

    Arguments:
        None

    Returns:
        str: Opening sentence, e.g.
            "There are a few highlights based on the sources we follow today, the fourth of October, 2026."

    Example:
        >>> opening = build_spoken_opening()
        >>> opening.startswith("There are a few highlights")
        True
    """
    run_date = get_run_date()
    parsed = datetime.strptime(run_date, "%Y-%m-%d")
    day = parsed.day
    month_name = parsed.strftime("%B")
    year = parsed.year
    ordinal = ORDINAL_WORDS[day]

    opening = (
        "There are a few highlights based on the sources we follow today, "
        f"the {ordinal} of {month_name}, {year}."
    )
    return opening


def build_news_script_prompt() -> str:
    """
    Build prompt instructions for perspective digest generation.

    Arguments:
        None

    Returns:
        str: Complete prompt with today's date already embedded.

    Example:
        >>> "highlights" in build_news_script_prompt()
        True
    """
    spoken_opening = build_spoken_opening()

    prompt = (
        "You are writing a spoken news segment for daily audio. Write for the ear,\n"
        "as if narrating a radio story. Aim for about 2500 words.\n\n"
        "HOW TO WRITE:\n\n"
        "1. Start with this exact line, on its own:\n"
        f"   {spoken_opening}\n"
        "   Do not modify it. Do not add anything before it. Do not add\n"
        "   anything after it on the same line.\n\n"
        "2. Move through the topics in flowing prose. No headers, no labels, no\n"
        "   brackets, no markdown, no bullet points.\n\n"
        "3. Never end a sentence with \"according to X.\" Instead, use these forms:\n"
        "   - \"X argues that...\"\n"
        "   - \"X says...\"\n"
        "   - \"X points out...\"\n"
        "   - \"X warned that...\"\n"
        "   Or put the speaker name mid-sentence: \"The speech, Sachs said, was\n"
        "   ghastly.\"\n\n"
        "4. When several speakers make the SAME argument, present it as one claim\n"
        "   and name them together. Example: \"Sachs, Giraldi, and Haiphong all\n"
        "   describe the speech as unprecedented.\"\n\n"
        "5. When speakers make DIFFERENT arguments on the same topic, give each\n"
        "   one their own sentence. Do not drop any speaker.\n\n"
        "6. Cover every topic in the input. Do not skip topics. Do not merge\n"
        "   separate topics.\n\n"
        "7. NEVER state the same fact, prediction, or claim twice. If a speaker\n"
        "   repeats something another speaker already said, do not repeat it -\n"
        "   either merge it into the earlier mention with a name added, or drop\n"
        "   the duplicate. Before finishing, review the draft and remove any\n"
        "   sentence that repeats a point already made.\n\n"
        "8. Use \"today\" not \"this week.\"\n\n"
        "9. Before the closing sentence, summarize the through-line in one or two\n"
        "   sentences. Do not label it \"conclusion.\"\n\n"
        "10. Plain text only. No JSON, no markdown, no asterisks.\n\n"
        "11. Close with this exact sentence: \"These were today's highlights.\"\n\n"
        "Worked example:\n\n"
        f"{spoken_opening}\n\n"
        "One story dominated: Donald Trump's speech to the United Nations General\n"
        "Assembly on Iran. Jeffrey Sachs called the language ghastly. Phil Giraldi\n"
        "said it violated the UN Charter. Danny Haiphong read it as a declaration\n"
        "of total war.\n\n"
        "They diverge on what it means. Sachs describes a constitutional crisis.\n"
        "Giraldi describes a violation of international law. Haiphong describes a\n"
        "strategic blunder.\n\n"
        "Iran's president, Masoud Pezeshkian, responded from Tehran: \"We have only\n"
        "defended ourselves. We are not terrorists.\"\n\n"
        "Another highlight: American military capacity. Haiphong claims heavy\n"
        "losses. Giraldi warns of a false flag. Sachs frames the war as illegal.\n\n"
        "Across all of it, one thread: the US is weakened and isolated.\n\n"
        "These were today's highlights."
    )

    return prompt


def build_news_script_messages(topics_data: dict[str, Any]) -> tuple[list, str]:
    """
    Build model messages from topics data.

    Arguments:
        topics_data (dict[str, Any]): Topics payload containing perspectives and themes.

    Returns:
        tuple[list, str]: Messages list and rendered prompt text.

    Example:
        >>> messages, text = build_news_script_messages({"topics": []})
        >>> len(messages) == 2
        True
    """
    topics = topics_data.get("topics") or []
    blocks: list[str] = []

    for topic in topics:
        if not isinstance(topic, dict):
            continue

        topic_name = str(topic.get("name") or "Untitled topic")
        topic_description = str(topic.get("description") or "").strip()
        consensus = str(topic.get("consensus") or "mixed")

        lines = [f"TOPIC: {topic_name}", f"Consensus: {consensus}"]
        if topic_description:
            lines.append(f"Description: {topic_description}")

        perspectives = topic.get("perspectives") or []
        if perspectives:
            lines.append("Perspectives:")
            for perspective in perspectives:
                if not isinstance(perspective, dict):
                    continue
                speaker = str(perspective.get("speaker") or "Unknown").strip() or "Unknown"
                channel = str(perspective.get("channel") or "").strip()
                text = str(perspective.get("text") or "").strip()
                if not text:
                    continue
                if channel:
                    lines.append(f"  - {speaker} [{channel}]: {text}")
                else:
                    lines.append(f"  - {speaker}: {text}")

        themes = topic.get("themes") or []
        if themes:
            lines.append("Themes:")
            for theme in themes:
                if not isinstance(theme, dict):
                    continue
                name = str(theme.get("name") or "").strip()
                if not name:
                    continue
                description = str(theme.get("description") or "").strip()
                supporting_speakers = theme.get("supporting_speakers") or []
                speaker_text = ", ".join(str(item).strip() for item in supporting_speakers if str(item).strip())
                if speaker_text:
                    lines.append(f"  - {name} (supporting speakers: {speaker_text})")
                else:
                    lines.append(f"  - {name}")
                if description:
                    lines.append(f"    {description}")

        blocks.append("\n".join(lines))

    rendered_topics = "\n\n---\n\n".join(blocks) if blocks else "No topics available."

    system_text = build_news_script_prompt()
    human_text = (
        "Write today's spoken digest. Start with the exact opening sentence. "
        "Follow the style rules. ~1200-1500 words. Return only the script text.\n"
        f"Date: {datetime.now().date().isoformat()}\n\n"
        f"{rendered_topics}"
    )

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ]
    prompt_text = system_text + "\n\n" + human_text

    return messages, prompt_text


def generate_news_script(
    topics_data: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 12000,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: int = DEFAULT_BACKOFF_SECONDS,
) -> tuple[str, int, float]:
    """
    Generate a perspective digest from topics data.

    Arguments:
        topics_data (dict[str, Any]): Topics payload with perspectives and themes.
        api_key (str): DeepSeek API key.
        model (str): Primary model name.
        fallback_model (str): Backup model name.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum output tokens.
        max_retries (int): Maximum retry count.
        backoff_seconds (int): Base retry delay in seconds.

    Returns:
        tuple[str, int, float]: Digest text, tokens used, and estimated cost.

    Example:
        >>> text, tokens, cost = generate_news_script({"topics": []}, api_key="demo")
        >>> text.startswith("No topics")
        True
    """
    has_no_topics = not topics_data.get("topics")
    if has_no_topics == True:
        empty_text = "No topics available to generate digest."
        empty_result = (empty_text, 0, 0.0)

        return empty_result

    messages, prompt_text = build_news_script_messages(topics_data)
    encoding = build_encoding(model)
    fallback_input_tokens = count_tokens(encoding, prompt_text)

    selected_model = model
    model_name = model

    # Keep signature compatibility while using one explicit model call path.
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

    digest_text = getattr(final_response, "content", "")
    if digest_text is None:
        digest_text = ""

    spoken_opening = build_spoken_opening()
    has_expected_opening = digest_text.startswith(spoken_opening)
    if has_expected_opening == False:
        digest_text = spoken_opening + "\n\n" + digest_text

    has_expected_closing = digest_text.endswith(SPOKEN_CLOSING)
    if has_expected_closing == False:
        digest_text = digest_text + "\n\n" + SPOKEN_CLOSING

    input_tokens, output_tokens, total_tokens = extract_usage_counts(
        final_response,
        fallback_input_tokens,
        digest_text,
        encoding,
    )
    cost = compute_cost(input_tokens, output_tokens)

    logging.info(
        "Perspective digest cost: $%.6f (input=%s, output=%s, total=%s, model=%s, words=%s)",
        cost,
        input_tokens,
        output_tokens,
        total_tokens,
        model_name,
        len(digest_text.split()),
    )

    result = (digest_text, total_tokens, cost)

    return result


def write_news_script(
    digest_text: str,
    output_dir: str | Path,
    topics_data: dict[str, Any],
    tokens_used: int = 0,
    cost: float = 0.0,
) -> tuple[Path, Path]:
    """
    Write perspective digest outputs to text and JSON files.

    Arguments:
        digest_text (str): Generated digest text.
        output_dir (str | Path): Destination directory.
        topics_data (dict[str, Any]): Topics payload used for generation.
        tokens_used (int): Token usage count.
        cost (float): Estimated generation cost.

    Returns:
        tuple[Path, Path]: Text file path and JSON file path.

    Example:
        >>> text_path, json_path = write_news_script("x", "tmp/news_script", {"topics": []})
        >>> text_path.name
        'news_script.txt'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    text_path = output_path / "news_script.txt"
    write_text(text_path, digest_text)

    topics = topics_data.get("topics")
    if topics is None:
        topics = []

    topic_count = len(topics)

    perspective_count = 0
    for topic in topics:
        if isinstance(topic, dict) == False:
            continue

        perspectives = topic.get("perspectives")
        if perspectives is None:
            perspectives = []

        perspective_count += len(perspectives)

    word_count = len(digest_text.split())
    character_count = len(digest_text)
    generated_at = datetime.now().isoformat()

    json_payload = {
        "digest": digest_text,
        "metadata": {
            "format": "spoken_daily_digest",
            "word_count": word_count,
            "character_count": character_count,
            "tokens_used": tokens_used,
            "cost": cost,
            "topic_count": topic_count,
            "perspective_count": perspective_count,
            "generated_at": generated_at,
        },
        "topics": topics,
    }

    json_path = output_path / "news_script.json"
    write_json(json_path, json_payload)

    result = (text_path, json_path)

    return result


def build_news_script_stats(
    topic_count: int,
    perspective_count: int,
    word_count: int,
    tokens_used: int,
    cost: float,
) -> dict[str, Any]:
    """
    Build stage stats for perspective digest generation.

    Arguments:
        topic_count (int): Number of topics provided.
        perspective_count (int): Number of perspectives provided.
        word_count (int): Word count of the generated digest.
        tokens_used (int): Token usage count.
        cost (float): Estimated generation cost.

    Returns:
        dict[str, Any]: Stage statistics payload.

    Example:
        >>> stats = build_news_script_stats(8, 40, 4500, 5000, 0.02)
        >>> stats["stage"]
        'news_script_generation'
    """
    stats = {
        "stage": "news_script_generation",
        "topic_count": topic_count,
        "perspective_count": perspective_count,
        "word_count": word_count,
        "tokens_used": tokens_used,
        "cost": cost,
        "generated_at": datetime.now().isoformat(),
    }

    return stats
