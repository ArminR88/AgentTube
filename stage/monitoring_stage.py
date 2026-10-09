import csv
import json
from datetime import datetime
from pathlib import Path
import sys
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
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_record = json.load(f)
            if len(summary_record.get("bullets", [])) > 0:
                flag_summary_generated = True

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
        })

    write_csv(
        monitoring_dir / "01_detection.csv",
        ["channel_name", "video_id", "title", "length", "published_at_local"],
        rows,
    )
    write_csv(
        monitoring_dir / "02_transcript_detection.csv",
        ["video_id", "flag_length_accepted", "flag_transcript_available"],
        rows,
    )
    write_csv(
        monitoring_dir / "03_download.csv",
        ["video_id", "flag_transcript_downloaded"],
        rows,
    )
    write_csv(
        monitoring_dir / "04_summarization.csv",
        ["video_id", "flag_summary_generated"],
        rows,
    )
    write_csv(
        monitoring_dir / "05_summary_of_summaries.csv",
        ["video_id", "flag_used_in_topics"],
        rows,
    )
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
