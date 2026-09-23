"""Stage runner for the full AgentTube pipeline."""

from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.output_helper import build_claims_output_directories, build_output_directories, write_failure_artifacts, write_json, write_text
from helpers.shared_helper import setup_logging
from helpers.transcript_summarization_helper import build_summary_transcript_records
from helpers.transcript_fact_checking_helper import build_fact_check_stats, fact_check_claims, write_fact_checked_claims
from helpers.summary_of_summaries_helper import build_claims_stats, extract_unique_claims, load_summary_files, write_claims
from helpers.news_script_generator_helper import build_script_stats, generate_news_script, load_fact_checked_claims, write_news_script
from stage.detecting_videos_stage import run_stage as run_detection_stage
from stage.downloading_transcript_stage import run_stage as run_download_stage
from stage.transcript_fact_checking_stage import run_stage as run_fact_check_stage
from stage.transcript_detection_stage import run_stage as run_transcript_detection_stage
from stage.transcript_summarization_stage import run_stage as run_summary_stage


def run_stage(
    verbose: bool = False,
    run_detection: bool = True,
    run_transcript_detection: bool = True,
    run_download: bool = True,
    run_summary: bool = True,
    run_fact_check: bool = True,
    run_claims: bool = True,
    run_script: bool = True,
    search_api_key: str | None = None,
    search_provider: str | None = None,
    summary_limit: int | None = None,
    output_root: str = "output_agenttube",
) -> dict[str, object]:
    """
    Run the full AgentTube pipeline and persist dated output artifacts.

    Arguments:
        verbose (bool): Enable debug-level logging if True. Default: False.
        run_detection (bool): Run the detection stage if True.
        run_transcript_detection (bool): Run the transcript detection stage if True.
        run_download (bool): Run transcript download if True.
        run_summary (bool): Run transcript summarization if True.
        run_fact_check (bool): Run transcript fact checking if True.
        run_claims (bool): Run summary-of-summaries claims extraction if True.
        run_script (bool): Run news script generation if True.
        search_api_key (str | None): Optional API key that enables Responses API web search.
        search_provider (str | None): Reserved compatibility argument for future search workflows.
        summary_limit (int | None): Optional cap on summarized records.
        output_root (str): Top-level output folder name.

    Returns:
        dict[str, object]: Pipeline payload containing records, outputs, and stats.

    Example:
        >>> payload = run_stage()
        >>> "output_root" in payload
        True
    """
    setup_logging(verbose)
    output_dirs = build_output_directories(output_root)
    claims_dirs = build_claims_output_directories(output_root)

    try:
        records = run_detection_stage() if run_detection else []
    except SystemExit as exc:
        write_failure_artifacts(output_dirs, str(exc))
        raise

    if run_transcript_detection and not run_detection:
        raise SystemExit("Transcript detection requires detection to run first.")

    if run_transcript_detection:
        transcript_records = run_transcript_detection_stage(records)
    else:
        transcript_records = records
    download_stats = None

    if run_download and not run_transcript_detection:
        raise SystemExit("Download requires transcript detection to run first.")

    if run_download:
        download_stats = run_download_stage(transcript_records, output_dir=str(output_dirs["transcripts"]))

    if run_summary and not run_download:
        raise SystemExit("Summarization requires transcript download to run first.")

    if run_fact_check and not run_summary:
        raise SystemExit("Fact checking requires transcript summarization to run first.")

    if run_claims and not run_summary:
        raise SystemExit("Claims extraction requires transcript summarization to run first.")

    if run_script and not run_claims:
        raise SystemExit("Script generation requires claims extraction to run first.")

    summary_records = []
    summary_transcript_records = []
    summary_stats = None
    fact_check_records = []
    fact_check_stats = None
    claims_result = None
    claims_stats = None
    fact_checked_claims_data = None
    script_result = None
    script_stats = None

    # Stage 4: Transcript Summarization
    if run_summary:
        summary_records = run_summary_stage(
            transcript_records,
            transcripts_dir=str(output_dirs["transcripts"]),
            summary_limit=summary_limit,
            summary_output_dir=str(output_dirs["transcript_summary"]),
        )
        summary_transcript_records = build_summary_transcript_records(summary_records)
        expected_summary_count = len(transcript_records)
        if summary_limit is not None:
            expected_summary_count = min(expected_summary_count, summary_limit)
        summary_stats = {
            "stage": "transcript_summarization",
            "record_count": len(summary_records),
            "summary_success_count": sum(1 for record in summary_records if record.get("summary_result", {}).get("success")),
            "summary_limit": summary_limit,
            "generated_at": datetime.now().isoformat(),
        }

    # Stage 5: Summary of Summaries (Claims Extraction)
    if run_claims:
        # Load all summary files
        summary_files = load_summary_files(str(output_dirs["transcript_summary"]))
        
        if summary_files:
            # Get API key for claims extraction
            import os
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            
            claims_draft, tokens_used, cost = extract_unique_claims(
                summary_files,
                api_key=api_key,
            )
            
            # Write claims
            claims_path = write_claims(claims_draft, str(claims_dirs["claims"]), tokens_used, cost)
            
            # Build stats
            claims_stats = build_claims_stats(len(summary_files), claims_draft)
            
            claims_result = {
                "claims": [claim.model_dump() if hasattr(claim, "model_dump") else claim.dict() for claim in claims_draft.claims],
                "claim_count": len(claims_draft.claims),
                "tokens_used": tokens_used,
                "cost": cost,
                "output_path": str(claims_path),
            }
        else:
            claims_stats = {
                "stage": "summary_of_summaries",
                "summary_record_count": 0,
                "claim_count": 0,
                "error": "No summary files found",
                "generated_at": datetime.now().isoformat(),
            }
            claims_result = {"claims": [], "claim_count": 0, "tokens_used": 0, "cost": 0.0}

    # Stage 6: Fact-Check Claims / Legacy Fact-Check
    fact_check_records = []
    fact_check_stats = None
    fact_checked_claims_data = None
    if run_fact_check:
        import os

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if run_claims and claims_result and claims_result.get("claim_count", 0) > 0:
            claims_data = {
                "claims": claims_result["claims"],
                "metadata": {
                    "claim_count": claims_result["claim_count"],
                    "tokens_used": claims_result["tokens_used"],
                    "cost": claims_result["cost"],
                },
            }
            fact_checked_claims_data = fact_check_claims(
                claims_data,
                api_key=api_key,
                search_api_key=search_api_key,
                search_provider=search_provider,
            )
            fact_checked_path = write_fact_checked_claims(
                fact_checked_claims_data,
                str(claims_dirs["fact_checked_claims"]),
            )
            fact_check_stats = {
                "stage": "claims_fact_checking",
                "claim_count": len(fact_checked_claims_data.get("claims", [])),
                "tokens_used": fact_checked_claims_data.get("metadata", {}).get("fact_check_tokens", 0),
                "cost": fact_checked_claims_data.get("metadata", {}).get("fact_check_cost", 0.0),
                "generated_at": datetime.now().isoformat(),
            }
        else:
            if run_summary and summary_transcript_records:
                fact_check_records = run_fact_check_stage(
                    summary_transcript_records,
                    output_dir=str(output_dirs["fact_checking"]),
                    search_api_key=search_api_key,
                    search_provider=search_provider,
                )
                fact_check_stats = build_fact_check_stats(summary_transcript_records, fact_check_records)
            else:
                fact_check_stats = {
                    "stage": "fact_checking",
                    "error": "No claims or summaries available",
                    "generated_at": datetime.now().isoformat(),
                }

    # Stage 7: News Script Generator
    if run_script and fact_checked_claims_data:
        import os
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        # Generate script
        script_text, tokens_used, cost = generate_news_script(
            fact_checked_claims_data,
            api_key=api_key,
            temperature=0.7,
            max_tokens=8000,
        )
        
        # Write script files
        text_path, json_path = write_news_script(
            script_text,
            str(claims_dirs["news_script"]),
            fact_checked_claims_data,
            tokens_used,
            cost,
        )
        
        # Build stats
        word_count = len(script_text.split())
        claim_count = len(fact_checked_claims_data.get("claims", []))
        script_stats = build_script_stats(claim_count, word_count, tokens_used, cost)
        
        script_result = {
            "script": script_text,
            "word_count": word_count,
            "tokens_used": tokens_used,
            "cost": cost,
            "text_output_path": str(text_path),
            "json_output_path": str(json_path),
        }
    elif run_script and not fact_checked_claims_data:
        script_stats = {
            "stage": "news_script_generation",
            "claim_count": 0,
            "word_count": 0,
            "tokens_used": 0,
            "cost": 0.0,
            "error": "No fact-checked claims available",
            "generated_at": datetime.now().isoformat(),
        }

    # Build pipeline summary
    detection_summary = {
        "stage": "detection",
        "record_count": len(records),
        "generated_at": datetime.now().isoformat(),
    }
    transcript_summary = {
        "stage": "transcript_detection",
        "input_record_count": len(records),
        "record_count": len(transcript_records),
        "transcript_available_count": sum(1 for record in transcript_records if record.get("transcript_available")),
        "generated_at": datetime.now().isoformat(),
    }
    download_summary = None
    if download_stats is not None:
        download_summary = {
            "stage": "download",
            **download_stats,
            "generated_at": datetime.now().isoformat(),
        }

    pipeline_summary = {
        "run_date": output_dirs["run_root"].name,
        "output_root": str(output_dirs["run_root"]),
        "stages": {
            "detection": detection_summary,
            "transcript_detection": transcript_summary,
            "download": download_summary,
            "transcript_summarization": summary_stats,
            "summary_of_summaries": claims_stats,
            "fact_checking": fact_check_stats,
            "news_script_generation": script_stats,
        },
    }

    # Verification
    verification_passed = True
    verification_messages = []
    if len(records) != len(transcript_records):
        verification_messages.append(
            f"Filtered final results: detection={len(records)} transcript_detection={len(transcript_records)}"
        )
    if download_stats is not None and download_stats["success"] + download_stats["failed"] != download_stats["total"]:
        verification_passed = False
        verification_messages.append("Download stats do not add up to the reported total.")
    expected_summary_count = len(transcript_records)
    if summary_limit is not None:
        expected_summary_count = min(expected_summary_count, summary_limit)
    if run_summary and len(summary_records) != expected_summary_count:
        verification_passed = False
        verification_messages.append(
            f"Summary record mismatch: expected={expected_summary_count} summarization={len(summary_records)}"
        )
    if run_claims:
        claims_count = claims_result.get("claim_count", 0) if claims_result else 0
        if claims_count < 3:
            verification_passed = False
            verification_messages.append(f"Expected at least 3 claims, got {claims_count}")
        elif claims_count < 10:
            verification_messages.append(f"Only {claims_count} claims extracted")
    if run_fact_check and run_claims and fact_checked_claims_data:
        fc_claims = fact_checked_claims_data.get("claims", [])
        verified_count = sum(1 for c in fc_claims if c.get("validation_status") in ["true", "false"])
        if verified_count == 0 and len(fc_claims) > 0:
            verification_passed = False
            verification_messages.append(f"All {len(fc_claims)} claims are unverified")
    if run_script:
        script_words = script_result.get("word_count", 0) if script_result else 0
        if script_words < 100:
            verification_passed = False
            verification_messages.append(f"Script too short: {script_words} words")
        elif script_words < 500:
            verification_messages.append(f"Script only {script_words} words")
    if run_fact_check and run_claims and claims_result and fact_checked_claims_data and len(fact_checked_claims_data.get("claims") or []) != claims_result["claim_count"]:
        verification_passed = False
        verification_messages.append(
            f"Fact-check record mismatch: expected={claims_result['claim_count']} fact_checking={len(fact_checked_claims_data.get('claims') or [])}"
        )
    if run_script and not script_result:
        verification_passed = False
        verification_messages.append("News script generation did not produce an output payload.")

    verification_summary = {
        "passed": verification_passed,
        "generated_at": datetime.now().isoformat(),
        "messages": verification_messages,
    }

    # Write all output files
    write_json(output_dirs["pipeline_summary"] / "01_detection.json", records)
    write_json(output_dirs["pipeline_summary"] / "02_transcript_detection.json", transcript_records)
    if download_summary is not None:
        write_json(output_dirs["pipeline_summary"] / "03_download_summary.json", download_summary)
    if summary_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "04_transcript_summarization.json", summary_stats)
    if claims_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "05_summary_of_summaries.json", claims_stats)
    if fact_check_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "06_fact_checking.json", fact_check_stats)
    if script_stats is not None:
        write_json(output_dirs["pipeline_summary"] / "07_news_script_generation.json", script_stats)
    write_json(output_dirs["pipeline_summary"] / "pipeline_summary.json", pipeline_summary)
    write_json(output_dirs["verification"] / "verification.json", verification_summary)
    write_text(
        output_dirs["verification"] / "verification.log",
        "Passed: " + ("yes" if verification_passed else "no") + "\n"
        + "\n".join(verification_messages or ["All checks passed."])
        + "\n",
    )

    pipeline_payload = {
        "records": records,
        "transcript_records": transcript_records,
        "summary_records": summary_records,
        "summary_transcript_records": summary_transcript_records,
        "fact_check_records": fact_checked_claims_data.get("claims") if fact_checked_claims_data else fact_check_records,
        "fact_check_stats": fact_check_stats,
        "claims_result": claims_result,
        "claims_stats": claims_stats,
        "fact_checked_claims": fact_checked_claims_data,
        "script_result": script_result,
        "script_stats": script_stats,
        "download_stats": download_stats,
        "output_root": str(output_dirs["run_root"]),
    }

    return pipeline_payload