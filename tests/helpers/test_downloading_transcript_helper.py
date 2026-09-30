from helpers.downloading_transcript_helper import (
    build_transcript_filename,
    get_video_id,
    sanitize_filename_part,
)


def test_sanitize_filename_part_judge_name():
    assert sanitize_filename_part("Judge Napolitano!") == "Judge_Napolitano"


def test_sanitize_filename_part_hello_world():
    assert sanitize_filename_part("hello world") == "hello_world"


def test_sanitize_filename_part_empty_string():
    assert sanitize_filename_part("") == "unknown"


def test_sanitize_filename_part_underscores_only():
    assert sanitize_filename_part("___") == "unknown"


def test_get_video_id_standard_url():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert get_video_id(url) == "dQw4w9WgXcQ"


def test_get_video_id_short_host_url():
    url = "https://youtube.com/watch?v=abc123XYZ_-"
    assert get_video_id(url) == "abc123XYZ_-"


def test_get_video_id_no_match():
    assert get_video_id("no video id here") == "unknown"


def test_build_transcript_filename_with_channel_name():
    assert build_transcript_filename("Judge Napolitano", "dQw4w9WgXcQ") == "Judge_Napolitano_dQw4w9WgXcQ.txt"


def test_build_transcript_filename_without_channel_name():
    assert build_transcript_filename(None, "dQw4w9WgXcQ") == "unknown_dQw4w9WgXcQ.txt"
