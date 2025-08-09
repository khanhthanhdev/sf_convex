"""
S3 client factory and transfer configuration.

Creates a configured boto3 S3 client and a TransferConfig tuned for
multipart uploads and concurrency. Reads settings from environment variables
so it can be shared across agents without additional dependencies.

Environment variables used (must match backend settings):
  - AWS_REGION (default: us-east-1)
  - S3_ENDPOINT_URL (optional, e.g., MinIO or VPC endpoint URL)
  - S3_FORCE_PATH_STYLE (true/false)
  - S3_MAX_CONCURRENCY (default: 8)
  - S3_MULTIPART_THRESHOLD_MB (default: 64)

Optional proxy support via standard envs:
  - HTTP_PROXY / HTTPS_PROXY
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoCoreConfig
from boto3.s3.transfer import TransferConfig


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "t", "yes", "y", "on"}


def get_s3_client(
    *,
    region_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    force_path_style: Optional[bool] = None,
    retries_total_max_attempts: int = 10,
) -> Any:
    """Create and return a configured boto3 S3 client.

    Args:
        region_name: AWS region; defaults to env AWS_REGION or 'us-east-1'.
        endpoint_url: Optional custom endpoint (e.g., MinIO/VPC endpoint).
        force_path_style: Use path-style addressing when True.
        retries_total_max_attempts: Total max attempts including initial request.

    Returns:
        boto3 S3 client instance.
    """

    region = region_name or os.getenv("AWS_REGION", "us-east-1")
    endpoint = endpoint_url or os.getenv("S3_ENDPOINT_URL")
    path_style = (
        force_path_style
        if force_path_style is not None
        else _get_bool_env("S3_FORCE_PATH_STYLE", False)
    )

    # Optional proxy support via standard env variables
    proxies: Optional[Dict[str, str]] = None
    http_proxy = os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("HTTPS_PROXY")
    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy

    # botocore Config with retries and S3 addressing style
    botocore_config = BotoCoreConfig(
        region_name=region,
        signature_version="v4",
        retries={
            "total_max_attempts": retries_total_max_attempts,
            "mode": "standard",
        },
        s3={
            "addressing_style": "path" if path_style else "virtual",
        },
        proxies=proxies,
    )

    session = boto3.session.Session(region_name=region)
    return session.client("s3", endpoint_url=endpoint, config=botocore_config)


def get_s3_transfer_config(
    *,
    multipart_threshold_mb: Optional[int] = None,
    max_concurrency: Optional[int] = None,
) -> TransferConfig:
    """Return a TransferConfig tuned for multipart uploads and concurrency.

    Args:
        multipart_threshold_mb: Threshold in MB to switch to multipart uploads.
        max_concurrency: Maximum parallel threads for transfers.
    """
    threshold_mb = multipart_threshold_mb or int(
        os.getenv("S3_MULTIPART_THRESHOLD_MB", "64")
    )
    concurrency = max_concurrency or int(os.getenv("S3_MAX_CONCURRENCY", "8"))

    threshold_bytes = threshold_mb * 1024 * 1024

    return TransferConfig(
        multipart_threshold=threshold_bytes,
        max_concurrency=concurrency,
        # Keep other defaults (chunk size, etc.) unless profiling suggests changes
    )


__all__ = ["get_s3_client", "get_s3_transfer_config"]


