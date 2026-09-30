# AgentTube

A six-stage YouTube pipeline that detects videos, checks transcript availability, downloads transcripts, generates summaries, extracts topics, and writes a final script.

## Prerequisites

- Python 3.11
- Podman
- DEEPSEEK_API_KEY
- YOUTUBE_API_KEY

## Local setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the repository root with required environment variables.
5. Run the pipeline:
   ```bash
   ./local_agenttube_run.sh
   ```

## Pipeline stages

1. detection: fetch recent channel videos and write flat detection records.
2. transcript_detection: detect transcript availability per video.
3. download: download transcript text files.
4. summary: summarize each transcript.
5. topics: extract cross-video topics and perspectives.
6. script: generate the final spoken digest/script.

## Output layout

Outputs are written under:

```text
output_agenttube/<date>/
  transcripts/
  transcript_summary/
  pipeline_summary/
  topics/
  news_script/
```

## Running with Podman

Build image:

```bash
podman build -t agenttube -f Containerfile .
```

Run container:

```bash
podman run --rm --env-file .env -v "$(pwd)/output_agenttube:/app/output_agenttube:Z" agenttube
```

The `:Z` suffix on the volume mount is required on SELinux systems (Fedora, RHEL) to allow the container to write to the mount.

## Running tests

```bash
pip install -r requirements-dev.txt && pytest
```

## CI

GitHub Actions runs `pytest` on every push.

## Configuration

- `YOUTUBE_API_KEY`: YouTube Data API key used for detection and video metadata fetch.
- `DEEPSEEK_API_KEY`: DeepSeek API key used by summary, topic extraction, and script generation stages.
