import csv
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import (
    build_monitoring_directory,
    build_output_directories,
    build_topics_output_directories,
)
from helpers.downloading_transcript_helper import get_video_id
from helpers.detecting_videos_helper import format_elapsed_seconds


def write_csv(path, columns, rows) -> None:
    """
    Write rows to a CSV file with the given column order.

    Arguments:
        path (Path): Destination file path.
        columns (list[str]): Column order.
        rows (list[dict]): Rows to write.

    Returns:
        None
    """
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def print_monitoring_banner(rows: list[dict]) -> None:
    """
    Print the stage 7 monitoring banner.

    Arguments:
        rows (list[dict]): Per-video rows with all flags and metrics.

    Returns:
        None

    Example:
        >>> print_monitoring_banner([])
        ---------------------------------------------------------------------------------------------------------
        #################################### Stage 7: Monitoring ####################################
        ---------------------------------------------------------------------------------------------------------
        Channel                        | Detected | Of Interest | With Transcript | Downloaded | Summarized |   Tokens |     Cost
        -------------------------------------------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------------------
        Total                          |        0 |           0 |               0 |          0 |          0 |        0 | $0.0000
        -------------------------------------------------------------------------------------------------------------------------
    """
    from helpers.output_helper import build_topics_output_directories
    from helpers.detecting_videos_helper import CHANNEL_NAMES
    import json

    channel_counts = {}
    for channel_name in CHANNEL_NAMES.values():
        channel_counts[channel_name] = {
            "detected": 0,
            "of_interest": 0,
            "with_transcript": 0,
            "downloaded": 0,
            "summarized": 0,
            "tokens": 0,
            "cost": 0.0,
        }

    for row in rows:
        channel_name = row["channel_name"]
        if channel_name not in channel_counts:
            channel_counts[channel_name] = {
                "detected": 0,
                "of_interest": 0,
                "with_transcript": 0,
                "downloaded": 0,
                "summarized": 0,
                "tokens": 0,
                "cost": 0.0,
            }

        channel_counts[channel_name]["detected"] += 1
        if row.get("flag_length_accepted") == True:
            channel_counts[channel_name]["of_interest"] += 1
        if row.get("flag_transcript_available") == True:
            channel_counts[channel_name]["with_transcript"] += 1
        if row.get("flag_transcript_downloaded") == True:
            channel_counts[channel_name]["downloaded"] += 1
        if row.get("flag_summary_generated") == True:
            channel_counts[channel_name]["summarized"] += 1
            channel_counts[channel_name]["tokens"] += int(row.get("tokens_used", 0))
            channel_counts[channel_name]["cost"] += float(row.get("cost", 0.0))

    print("-" * 105)
    print("#################################### Stage 7: Monitoring ####################################")
    print("-" * 105)
    print(f"{'Channel':<30} | {'Detected':>8} | {'Of Interest':>11} | {'With Transcript':>15} | {'Downloaded':>10} | {'Summarized':>10} | {'Tokens':>8} | {'Cost':>8}")
    print("-" * 121)

    total_detected = 0
    total_of_interest = 0
    total_with_transcript = 0
    total_downloaded = 0
    total_summarized = 0
    total_tokens = 0
    total_cost = 0.0

    for channel_name, counts in channel_counts.items():
        detected = counts["detected"]
        of_interest = counts["of_interest"]
        with_transcript = counts["with_transcript"]
        downloaded = counts["downloaded"]
        summarized = counts["summarized"]
        tokens = counts["tokens"]
        cost = counts["cost"]

        total_detected += detected
        total_of_interest += of_interest
        total_with_transcript += with_transcript
        total_downloaded += downloaded
        total_summarized += summarized
        total_tokens += tokens
        total_cost += cost

        print(f"{channel_name:<30} | {detected:>8} | {of_interest:>11} | {with_transcript:>15} | {downloaded:>10} | {summarized:>10} | {tokens:>8,} | ${cost:>7.4f}")

    print("-" * 121)
    print(f"{'Total':<30} | {total_detected:>8} | {total_of_interest:>11} | {total_with_transcript:>15} | {total_downloaded:>10} | {total_summarized:>10} | {total_tokens:>8,} | ${total_cost:>7.4f}")
    print("-" * 121)
    print()

    topics_dirs = build_topics_output_directories()
    summary_of_summaries_path = topics_dirs["summary_of_summaries"] / "summary_of_summaries.json"
    news_script_path = topics_dirs["news_script"] / "news_script.json"

    stage5_topics = 0
    stage5_perspectives = 0
    stage5_themes = 0
    stage5_tokens = 0
    stage5_cost = 0.0

    if summary_of_summaries_path.exists():
        with open(summary_of_summaries_path, "r", encoding="utf-8") as file:
            summary_of_summaries_data = json.load(file)
        metadata = summary_of_summaries_data.get("metadata", {})
        stage5_topics = int(metadata.get("topic_count", 0))
        stage5_perspectives = int(metadata.get("perspective_count", 0))
        stage5_tokens = int(metadata.get("tokens_used", 0))
        stage5_cost = float(metadata.get("cost", 0.0))
        for topic in summary_of_summaries_data.get("topics", []):
            stage5_themes += len(topic.get("themes", []))

    stage6_word_count = 0
    stage6_tokens = 0
    stage6_cost = 0.0

    if news_script_path.exists():
        with open(news_script_path, "r", encoding="utf-8") as file:
            news_script_data = json.load(file)
        metadata = news_script_data.get("metadata", {})
        stage6_word_count = int(metadata.get("word_count", 0))
        stage6_tokens = int(metadata.get("tokens_used", 0))
        stage6_cost = float(metadata.get("cost", 0.0))

    print(f"{'Stage':<30} | {'Topics':>6} | {'Perspectives':>12} | {'Themes':>6} | {'Word count':>10} | {'Tokens':>6} | {'Cost':>6}")
    print("-" * 92)
    print(f"{'Summary of Summaries':<30} | {stage5_topics:>6} | {stage5_perspectives:>12} | {stage5_themes:>6} | {'—':>10} | {stage5_tokens:>6,} | ${stage5_cost:.4f}")
    print(f"{'News Script Generation':<30} | {'—':>6} | {'—':>12} | {'—':>6} | {stage6_word_count:>10,} | {stage6_tokens:>6,} | ${stage6_cost:.4f}")
    print("-" * 92)
    print()

    total_pipeline_cost = total_cost + stage5_cost + stage6_cost
    print(f"TOTAL PIPELINE COST:  ${total_pipeline_cost:.4f}")
    print("-" * 92)


def run_stage() -> None:
    """
    Run the monitoring stage using default pipeline locations.

    Arguments:
        None

    Returns:
        None

    Example:
        >>> run_stage()
        >>> True
        True
    """
    logging.info("[Stage 7] Monitoring started")
    start_time = time.time()

    output_dirs = build_output_directories()
    topics_dirs = build_topics_output_directories()
    monitoring_dir = build_monitoring_directory()

    detection_path = output_dirs["pipeline_summary"] / "01_detection.json"
    transcript_detection_path = output_dirs["pipeline_summary"] / "02_transcript_detection.json"
    summary_of_summaries_path = topics_dirs["summary_of_summaries"] / "summary_of_summaries.json"

    with open(detection_path, "r", encoding="utf-8") as file:
        detection_records = json.load(file)

    with open(transcript_detection_path, "r", encoding="utf-8") as file:
        transcript_detection_records = json.load(file)

    with open(summary_of_summaries_path, "r", encoding="utf-8") as file:
        summary_of_summaries_data = json.load(file)

    available_ids = set()
    for record in transcript_detection_records:
        if record.get("transcript_available") == True:
            available_ids.add(get_video_id(record["url"]))

    used_in_topics_ids = set()
    for topic in summary_of_summaries_data.get("topics", []):
        for perspective in topic.get("perspectives", []):
            video_id_value = perspective.get("video_id")
            if video_id_value:
                used_in_topics_ids.add(video_id_value)

    rows = []
    for record in detection_records:
        channel_name = record.get("channel_name", "")
        video_id = get_video_id(record["url"])
        title = record.get("title", "")
        length = record.get("duration", "00:00:00")
        published_at_utc = record.get("published_at", "")
        published_at_dt = datetime.fromisoformat(published_at_utc.replace("Z", "+00:00"))
        published_at_local = published_at_dt.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M")

        parts = length.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
        flag_length_accepted = total_seconds >= 18 * 60

        flag_transcript_available = video_id in available_ids

        transcript_file = output_dirs["transcripts"] / f"{channel_name}_{video_id}.txt"
        flag_transcript_downloaded = transcript_file.exists()

        summary_file = output_dirs["transcript_summary"] / f"{channel_name}_{video_id}_summary.json"
        flag_summary_generated = False
        tokens_used = 0
        cost = 0.0
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_record = json.load(f)
            if len(summary_record.get("bullets", [])) > 0:
                flag_summary_generated = True
                tokens_used = int(summary_record.get("tokens_used", 0))
                cost = float(summary_record.get("cost", 0.0))

        flag_used_in_topics = video_id in used_in_topics_ids

        rows.append({
            "channel_name": channel_name,
            "video_id": video_id,
            "title": title,
            "length": length,
            "published_at_local": published_at_local,
            "flag_length_accepted": flag_length_accepted,
            "flag_transcript_available": flag_transcript_available,
            "flag_transcript_downloaded": flag_transcript_downloaded,
            "flag_summary_generated": flag_summary_generated,
            "flag_used_in_topics": flag_used_in_topics,
            "tokens_used": tokens_used,
            "cost": cost,
        })

    write_csv(
        monitoring_dir / "01_detection.csv",
        ["channel_name", "video_id", "title", "length", "published_at_local"],
        rows,
    )
    print("Saved: 01_detection.csv")
    write_csv(
        monitoring_dir / "02_transcript_detection.csv",
        ["video_id", "flag_length_accepted", "flag_transcript_available"],
        rows,
    )
    print("Saved: 02_transcript_detection.csv")
    write_csv(
        monitoring_dir / "03_download.csv",
        ["video_id", "flag_transcript_downloaded"],
        rows,
    )
    print("Saved: 03_download.csv")
    write_csv(
        monitoring_dir / "04_summarization.csv",
        ["video_id", "flag_summary_generated"],
        rows,
    )
    print("Saved: 04_summarization.csv")
    write_csv(
        monitoring_dir / "05_summary_of_summaries.csv",
        ["video_id", "flag_used_in_topics"],
        rows,
    )
    print("Saved: 05_summary_of_summaries.csv")
    write_csv(
        monitoring_dir / "monitoring_all.csv",
        [
            "channel_name",
            "video_id",
            "title",
            "length",
            "published_at_local",
            "flag_length_accepted",
            "flag_transcript_available",
            "flag_transcript_downloaded",
            "flag_summary_generated",
            "flag_used_in_topics",
        ],
        rows,
    )
    print("Saved: monitoring_all.csv")

    print_monitoring_banner(rows)

    elapsed_seconds = time.time() - start_time
    logging.info(
        "[Stage 7] Monitoring finished (%s)",
        format_elapsed_seconds(elapsed_seconds),
    )
