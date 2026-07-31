"""Detect YouTube videos uploaded from a channel in the last 24 hours.

This is the standalone wrapper for the detection stage.
It exists so the detector can still be run directly.
"""

import json
import logging
import sys

from helpers.detecting_videos_helper import (
    build_detection_records,
    detect_recent_videos,
    get_api_key,
    setup_logging,
)


def parse_arguments() -> object:
    """
    Parse command-line arguments for the detector.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Detect recent YouTube videos")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parsed_arguments = parser.parse_args()

    return parsed_arguments


def print_results(results: dict[str, dict[str, object]]) -> None:
    """
    Print detection results in a human-readable form.

    Arguments:
        results (dict[str, dict[str, object]]): Detection results grouped by channel.

    Returns:
        None
    """
    print(f"\n{'='*60}")
    print("📡 YOUTUBE VIDEO DETECTION")
    print(f"{'='*60}")

    total_videos = 0
    channels_with_videos = 0

    for _, data in results.items():
        print(f"\n📺 {data['name']}:")

        if data["error"]:
            print(f"  ❌ Error: {data['error']}")
            continue

        videos = data["videos"]
        if not videos:
            print("  ℹ️  No videos in the last 24 hours")
            continue

        channels_with_videos += 1
        total_videos += len(videos)

        for video in videos:
            print(f"  🎬 {video['title']}")
            print(f"     🔗 {video['url']}")
            print(f"     📅 {video['published_at']}")

            description = video.get("description", "")
            if description:
                if len(description) > 100:
                    description = description[:100] + "..."
                print(f"     📝 {description}")

            print()

    print(f"{'='*60}")
    print("📊 SUMMARY:")
    print(f"  Total channels: {len(results)}")
    print(f"  Channels with new videos: {channels_with_videos}")
    print(f"  Total videos found: {total_videos}")
    print(f"{'='*60}\n")


def main() -> None:
    """
    Run the detector stage.

    Returns:
        None
    """
    args = parse_arguments()
    setup_logging(args.verbose)

    api_key = get_api_key()
    if not api_key:
        logging.error("YOUTUBE_API_KEY is not set.")
        sys.exit(2)

    detection_results = detect_recent_videos(api_key)

    if args.json:
        print(json.dumps(build_detection_records(detection_results), indent=2, default=str))
        return

    print_results(detection_results)

    if any(data["error"] is not None for data in detection_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()