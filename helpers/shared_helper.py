"""Shared helpers for AgentTube stages and reusable utilities."""

import logging
import time
from typing import Any

import requests
from requests import RequestException


class DevTooManyAttemptsError(RequestException):
    """
    Signal that a dev-only retry budget was exhausted.

    Arguments:
        None

    Returns:
        None
    """


def dev_request_json(url: str, *, timeout: int, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Fetch JSON from a URL with a dev-only retry budget for rate limits.

    Arguments:
        url (str): Target URL.
        timeout (int): Request timeout in seconds.
        params (dict[str, Any] | None): Optional query parameters.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        RequestException: On request failure.
        ValueError: If the response body is not valid JSON.
        DevTooManyAttemptsError: If repeated 429 responses exhaust retries.
    """
    max_retries = 5
    backoff_seconds = 2
    response = None

    for attempt in range(max_retries + 1):
        response = requests.get(url, params=params, timeout=timeout)

        if response.status_code != 429:
            break

        if attempt == max_retries:
            error_message = (
                f"Dev retry budget exhausted after {max_retries + 1} attempts while fetching {url}"
            )
            logging.error(error_message)
            raise DevTooManyAttemptsError(error_message)

        retry_after_text = response.headers.get("Retry-After")
        if retry_after_text and retry_after_text.isdigit():
            delay_seconds = int(retry_after_text)
        else:
            delay_seconds = backoff_seconds * (2**attempt)

        logging.warning("Rate limited while fetching %s; retrying in %s seconds", url, delay_seconds)
        time.sleep(delay_seconds)

    assert response is not None

    try:
        response.raise_for_status()
    except RequestException as exc:
        try:
            api_error = response.json().get("error", {}).get("message")
        except ValueError:
            api_error = None

        if api_error:
            raise RequestException(api_error) from exc

        raise

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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )