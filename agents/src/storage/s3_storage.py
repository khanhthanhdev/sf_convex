"""
S3 storage abstraction for uploading and serving agent outputs.

Implements the interface defined in docs/s3_storage_plan.md (Step 3):
- S3Storage(bucket, base_prefix)
- upload_file
- upload_bytes
- upload_dir
- generate_presigned_url
- url_for
- write_manifest (optional)

This class intentionally reads minimal deployment settings from environment
variables to remain lightweight and reusable across agents without importing
backend settings directly.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .s3_client import get_s3_client, get_s3_transfer_config


def _norm_join(*parts: str) -> str:
    # Join with POSIX-style separators for S3 keys
    joined = "/".join([p.strip("/") for p in parts if p is not None and str(p) != ""])
    return joined


def _guess_content_type(path_or_key: str) -> Optional[str]:
    ctype, _ = mimetypes.guess_type(path_or_key)
    return ctype


# ------------------------ Step 4: Key Strategy ------------------------
def sanitize_topic_to_prefix(topic: str) -> str:
    """Sanitize topic name to kebab-case prefix used as S3 root.

    - Lowercase
    - Replace any non-alphanumeric with '-'
    - Collapse multiple '-' and trim leading/trailing '-'
    """
    s = topic.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def key_for_local_path(local_path: str, *, topic_name: str, output_dir: str) -> str:
    """Derive S3 key from a local absolute/relative path by mirroring structure.

    Rule (per plan): strip local `output_dir` from absolute path; prepend
    `topic-name/` (sanitized to kebab-case) to the remaining relative path.
    """
    lp = Path(local_path).resolve()
    od = Path(output_dir).resolve()
    try:
        rel = lp.relative_to(od)
    except ValueError as e:
        raise ValueError(f"Local path {lp} is not under output_dir {od}") from e

    topic_prefix = sanitize_topic_to_prefix(topic_name)
    return _norm_join(topic_prefix, rel.as_posix())


class S3Storage:
    """
    High-level S3 storage helper.

    Args:
        bucket: Target S3 bucket name.
        base_prefix: Optional base prefix applied to all keys (e.g., "projects/").
    """

    def __init__(self, bucket: str, base_prefix: str = "") -> None:
        if not bucket:
            raise ValueError("bucket is required")
        self.bucket = bucket
        self.base_prefix = base_prefix.strip("/")

        # Configure S3 client and transfer config
        self.client = get_s3_client()
        self.transfer_config = get_s3_transfer_config()

        # Optional behavior/env
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.public_base_url = os.getenv("S3_PUBLIC_BASE_URL")
        self.kms_key_id = os.getenv("S3_KMS_KEY_ID")
        self.presign_expiration = int(os.getenv("S3_PRESIGN_EXPIRATION", "3600"))
        self.max_concurrency = int(os.getenv("S3_MAX_CONCURRENCY", "8"))

    # ---------------------------- key helpers ----------------------------
    def _full_key(self, key: str) -> str:
        if self.base_prefix:
            return _norm_join(self.base_prefix, key)
        return key.strip("/")

    # ----------------------------- uploads ------------------------------
    def upload_file(self, local_path: str, key: str, extra_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a local file to S3 using high-performance TransferConfig.

        Returns the S3 key actually used (including base_prefix).
        """
        if not Path(local_path).is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_key = self._full_key(key)

        # Build ExtraArgs with content-type and optional SSE-KMS
        extra: Dict[str, Any] = dict(extra_args or {})
        ctype = _guess_content_type(local_path) or _guess_content_type(full_key)
        if ctype:
            extra.setdefault("ContentType", ctype)
        if self.kms_key_id:
            extra.setdefault("ServerSideEncryption", "aws:kms")
            extra.setdefault("SSEKMSKeyId", self.kms_key_id)

        self.client.upload_file(
            Filename=str(local_path),
            Bucket=self.bucket,
            Key=full_key,
            ExtraArgs=extra if extra else None,
            Config=self.transfer_config,
        )
        return full_key

    def upload_bytes(self, data: bytes, key: str, extra_args: Optional[Dict[str, Any]] = None) -> str:
        """Upload raw bytes to S3, setting content-type when possible."""
        full_key = self._full_key(key)
        extra: Dict[str, Any] = dict(extra_args or {})
        ctype = _guess_content_type(full_key)
        if ctype:
            extra.setdefault("ContentType", ctype)
        if self.kms_key_id:
            extra.setdefault("ServerSideEncryption", "aws:kms")
            extra.setdefault("SSEKMSKeyId", self.kms_key_id)

        # put_object is efficient for small/medium payloads
        self.client.put_object(
            Bucket=self.bucket,
            Key=full_key,
            Body=data,
            **({} if not extra else extra),
        )
        return full_key

    def upload_dir(self, local_root: str, key_prefix: str) -> Tuple[int, int]:
        """
        Recursively upload a directory to S3.

        Args:
            local_root: Path to local directory to upload.
            key_prefix: Prefix under which files will be stored (relative to base_prefix).
        Returns:
            (files_count, errors_count)
        """
        root_path = Path(local_root)
        if not root_path.is_dir():
            raise NotADirectoryError(f"Local directory not found: {local_root}")

        files: list[Path] = [p for p in root_path.rglob("*") if p.is_file()]
        errors = 0

        def _upload_one(p: Path) -> None:
            rel = p.relative_to(root_path).as_posix()
            k = _norm_join(key_prefix, rel)
            self.upload_file(str(p), k)

        # Use limited concurrency; boto3 handles per-transfer parallelism too.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {executor.submit(_upload_one, p): p for p in files}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    errors += 1

        return (len(files), errors)

    # ------------------------- URL generation ---------------------------
    def generate_presigned_url(self, key: str, expires: Optional[int] = None) -> str:
        """Generate a presigned GET URL for a key."""
        full_key = self._full_key(key)
        expiration = int(expires or self.presign_expiration)
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": full_key},
            ExpiresIn=expiration,
        )

    def url_for(self, key: str, *, prefer_presigned: bool = False) -> str:
        """
        Return a URL for accessing the key.
        - If S3_PUBLIC_BASE_URL is set, return "{PUBLIC_BASE_URL}/{full_key}".
        - Else, when prefer_presigned=True, return a presigned GET.
        - Else, return the standard virtual-hosted–style S3 URL.
        """
        full_key = self._full_key(key)
        if self.public_base_url:
            return _norm_join(self.public_base_url, full_key)
        if prefer_presigned:
            return self.generate_presigned_url(full_key)
        # Virtual-hosted–style URL
        return f"https://{self.bucket}.s3.{self.aws_region}.amazonaws.com/{full_key}"

    # --------------------------- manifest -------------------------------
    def write_manifest(self, topic_prefix: str, manifest_dict: Dict[str, Any]) -> str:
        """Write a manifest.json under the given topic prefix."""
        key = _norm_join(topic_prefix, "manifest.json")
        data = json.dumps(manifest_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.upload_bytes(data, key, extra_args={"ContentType": "application/json"})
