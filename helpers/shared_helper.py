"""Shared helpers for AgentTube stages and reusable utilities."""

import json
import logging
from pathlib import Path
from typing import Any

import requests


def dev_request_json(url: str, *, timeout: int, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Fetch JSON from a URL.

    Arguments:
        url (str): Target URL.
        timeout (int): Request timeout in seconds.
        params (dict[str, Any] | None): Optional query parameters.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        requests.RequestException: On request failure.
        ValueError: If the response body is not valid JSON.
    """
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    json_data = response.json()

    return json_data


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for batch processing.

    Arguments:
        verbose (bool): Enable debug-level logging if True. Default: False.

    Returns:
        None: Configures the logging module globally.

    Example:
        >>> setup_logging(verbose=True)
        >>> logging.debug("This will now appear in logs")
    """
    if verbose == True:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_json_object(path: str | Path, *, missing_label: str = "JSON file") -> dict[str, Any]:
    """
    Load a JSON object from disk.

    Arguments:
        path (str | Path): File path to read.
        missing_label (str): Label used in missing-file error messages.

    Returns:
        dict[str, Any]: Parsed JSON object.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If parsed JSON is not an object.
    """
    json_path = Path(path)

    if json_path.exists() == False:
        raise FileNotFoundError(f"{missing_label} not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) == False:
        raise ValueError(f"{missing_label} must contain a JSON object: {json_path}")

    return payload