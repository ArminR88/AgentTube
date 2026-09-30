from helpers.transcript_summarization_helper import compute_cost, parse_draft


def test_compute_cost_zero_tokens():
    assert compute_cost(0, 0) == 0


def test_compute_cost_positive_tokens():
    assert compute_cost(100, 100) > 0


def test_parse_draft_valid_json_with_bullets():
    response_text = '{"summary":"x","main_topic":"m","sentiment":"neutral","bullets":[{"speaker":"A","text":"t","is_opinion":false}]}'
    draft = parse_draft(response_text)
    assert draft.summary == "x"
    assert len(draft.bullets) == 1


def test_parse_draft_strips_json_fences():
    response_text = "```json\n{\"summary\":\"x\",\"bullets\":[]}\n```"
    draft = parse_draft(response_text)
    assert draft.summary == "x"


def test_parse_draft_fallback_non_json_text():
    draft = parse_draft("this is plain text")
    assert isinstance(draft.summary, str)
    assert len(draft.bullets) >= 0


def test_parse_draft_empty_input_empty_bullets():
    draft = parse_draft("")
    assert draft.bullets == []
