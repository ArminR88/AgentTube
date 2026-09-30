import pytest

from helpers.summary_of_summaries_helper import parse_topics_draft


def test_parse_topics_draft_valid_json_with_topics():
    response_text = '{"topics":[{"topic_id":1,"name":"A","description":"d","consensus":"mixed","perspectives":[],"themes":[]}]}'
    draft = parse_topics_draft(response_text)
    assert len(draft.topics) == 1


def test_parse_topics_draft_malformed_json_without_raising():
    with pytest.raises(Exception) as exc_info:
        parse_topics_draft("{not valid json")
    assert exc_info is not None


def test_parse_topics_draft_empty_topics():
    draft = parse_topics_draft('{"topics": []}')
    assert draft.topics == []
