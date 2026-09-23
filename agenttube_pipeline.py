"""Thin AgentTube pipeline entrypoint."""

import sys
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()

from stage.pipeline_stage import run_stage as run_pipeline_stage


def main() -> None:
    """
    Run the AgentTube pipeline CLI.

    Arguments:
        None

    Returns:
        None

    Example:
        >>> True
        True
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the AgentTube pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--json", action="store_true", help="Print the detection schema as JSON")
    parser.add_argument("--no-detection", action="store_true", help="Skip the detection stage")
    parser.add_argument("--no-transcript-detection", action="store_true", help="Skip the transcript detection stage")
    parser.add_argument("--no-download", action="store_true", help="Skip transcript download after transcript detection")
    parser.add_argument("--no-summary", action="store_true", help="Skip transcript summarization after transcript download")
    parser.add_argument("--no-topics", action="store_true", help="Skip topic extraction")
    parser.add_argument("--no-script", action="store_true", help="Skip perspective digest generation")
    parser.add_argument("--summary-limit", type=int, help="Limit the number of summarized records")
    args = parser.parse_args()

    try:
        payload = run_pipeline_stage(
            verbose=args.verbose,
            run_detection=not args.no_detection,
            run_transcript_detection=not args.no_transcript_detection,
            run_download=not args.no_download,
            run_summary=not args.no_summary,
            run_topics=not args.no_topics,
            run_script=not args.no_script,
            summary_limit=args.summary_limit,
        )
    except SystemExit:
        sys.exit(2)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    transcript_records = cast(list[dict[str, Any]], payload.get("transcript_records") or [])
    for index, record in enumerate(transcript_records, 1):
        transcript_state = "yes" if record.get("transcript_available") else "no"
        print(
            f"{index}. {record['channel_name']} | {record['title']} | duration: {record['duration']} | transcript: {transcript_state}"
        )

    summary_transcript_records = cast(list[dict[str, Any]], payload.get("summary_transcript_records") or [])
    if summary_transcript_records:
        print(f"Summaries: {len(summary_transcript_records)} record(s) processed")

    topics_result = cast(dict[str, Any], payload.get("topics_result") or {})
    if topics_result:
        print(f"Topics extracted: {topics_result.get('topic_count', 0)}")
        print(f"Perspectives: {topics_result.get('perspective_count', 0)}")

    digest_result = cast(dict[str, Any], payload.get("digest_result") or {})
    if digest_result:
        print(f"Perspective digest generated: {digest_result.get('word_count', 0)} words")
        print(f"Digest saved to: {digest_result.get('text_output_path', '')}")


if __name__ == "__main__":
    main()
