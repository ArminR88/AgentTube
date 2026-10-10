"""AgentTube pipeline entrypoint."""

from dotenv import load_dotenv

load_dotenv()

from helpers.shared_helper import setup_logging
from stage.detecting_videos_stage import run_stage as run_detection_stage
from stage.transcript_detection_stage import run_stage as run_transcript_detection_stage
from stage.downloading_transcript_stage import run_stage as run_download_stage
from stage.transcript_summarization_stage import run_stage as run_summary_stage
from stage.summary_of_summaries_stage import run_stage as run_topics_stage
from stage.news_script_generator_stage import run_stage as run_script_stage
from stage.monitoring_stage import run_stage as run_monitoring_stage


def run_pipeline(
    verbose=False,
    cond_run_detection_stage=False,
    cond_run_transcript_detection_stage=True,
    cond_run_download_stage=False,
    cond_run_summary_stage=False,
    cond_run_topics_stage=False,
    cond_run_script_stage=False,
    cond_run_monitoring_stage=False,
):
    """
    Run all AgentTube stages in order.

    Arguments:
        verbose (bool): Enable debug-level logging if True. Default: False.
        cond_run_detection_stage (bool): Run video detection stage when True.
        cond_run_transcript_detection_stage (bool): Run transcript detection stage when True.
        cond_run_download_stage (bool): Run transcript download stage when True.
        cond_run_summary_stage (bool): Run transcript summarization stage when True.
        cond_run_topics_stage (bool): Run topic extraction stage when True.
        cond_run_script_stage (bool): Run script generation stage when True.
        cond_run_monitoring_stage (bool): Run monitoring stage when True.

    Returns:
        None

    Example:
        >>> run_pipeline(verbose=True)
        >>> True
        True
    """
    setup_logging(verbose)
    if cond_run_detection_stage == True:
        run_detection_stage()
    if cond_run_transcript_detection_stage == True:
        run_transcript_detection_stage()
    if cond_run_download_stage == True:
        run_download_stage()
    if cond_run_summary_stage == True:
        run_summary_stage()
    if cond_run_topics_stage == True:
        run_topics_stage()
    if cond_run_script_stage == True:
        run_script_stage()
    if cond_run_monitoring_stage == True:
        run_monitoring_stage()


if __name__ == "__main__":
    run_pipeline()
    print("Pipeline finished.")
