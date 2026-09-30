from stage.detecting_videos_stage import run_stage


def test_run_stage_is_callable():
    assert callable(run_stage) == True
