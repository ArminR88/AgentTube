from helpers.news_script_generator_helper import build_news_script_stats


def test_build_digest_stats_stage_name():
    stats = build_news_script_stats(1, 2, 3, 4, 0.1)
    assert stats["stage"] == "news_script_generation"


def test_build_digest_stats_counts():
    stats = build_news_script_stats(8, 40, 4500, 5000, 0.02)
    assert stats["word_count"] == 4500
    assert stats["topic_count"] == 8
    assert stats["perspective_count"] == 40
