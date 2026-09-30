from helpers.detecting_videos_helper import format_duration, get_channel_name, title_matches_filter


def test_format_duration_pt1h2m3s():
    assert format_duration("PT1H2M3S") == "01:02:03"


def test_format_duration_pt0s():
    assert format_duration("PT0S") == "00:00:00"


def test_format_duration_pt30m():
    assert format_duration("PT30M") == "00:30:00"


def test_format_duration_pt1d():
    assert format_duration("PT1D") == "00:00:00"


def test_format_duration_invalid():
    assert format_duration("invalid") == "00:00:00"


def test_get_channel_name_known_id():
    assert get_channel_name("UCOxLhz6B_elvLflntSEfnzA") == "Danny_Haiphong"


def test_get_channel_name_unknown_id():
    unknown_id = "unknown-channel-id"
    assert get_channel_name(unknown_id) == unknown_id


def test_title_matches_filter_simple_match():
    assert title_matches_filter("Parsi on Iran", ["parsi"]) == True


def test_title_matches_filter_case_insensitive_match():
    assert title_matches_filter("JOHNSON on policy", ["johnson"]) == True


def test_title_matches_filter_no_match():
    assert title_matches_filter("Neutral title", ["parsi"]) == False


def test_title_matches_filter_any_keyword_match():
    assert title_matches_filter("Talk with Diesen", ["johnson", "diesen", "pape"]) == True
