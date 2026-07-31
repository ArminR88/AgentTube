"""Thin AgentTube pipeline entrypoint."""

import sys

from stage.pipeline_stage import run_stage as run_pipeline_stage


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the AgentTube pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--json", action="store_true", help="Print the detection schema as JSON")
    parser.add_argument("--no-detection", action="store_true", help="Skip the detection stage")
    parser.add_argument("--no-transcript-detection", action="store_true", help="Skip the transcript detection stage")
    parser.add_argument("--no-download", action="store_true", help="Skip transcript download after transcript detection")
    args = parser.parse_args()
    try:
        payload = run_pipeline_stage(
            verbose=args.verbose,
            run_detection=not args.no_detection,
            run_transcript_detection=not args.no_transcript_detection,
            run_download=not args.no_download,
        )
    except SystemExit as exc:
        sys.exit(2)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    for index, record in enumerate(payload["transcript_records"], 1):
        transcript_state = "yes" if record.get("transcript_available") else "no"
        print(
            f"{index}. {record['channel_name']} | {record['title']} | duration: {record['duration']} | transcript: {transcript_state}"
        )


if __name__ == "__main__":
    main()