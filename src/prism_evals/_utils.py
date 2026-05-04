from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())
    return repr(value)


def redact_data_urls(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_data_url(value)
    if isinstance(value, dict):
        return {key: redact_media_value(str(key), item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data_urls(item) for item in value]
    return value


def redact_media_value(key: str, value: Any) -> Any:
    if isinstance(value, str) and key in {"b64_json", "result", "partial_image_b64"}:
        return _redact_base64_media(value)
    return redact_data_urls(value)


def raw_payload(value: Any, *, redact_raw_data_urls: bool = True) -> Any:
    payload = to_jsonable(value)
    if not redact_raw_data_urls:
        return payload
    return redact_data_urls(payload)


def _redact_data_url(value: str) -> str:
    if not value.startswith("data:"):
        return value
    header, separator, data = value.partition(",")
    if not separator:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    metadata = [
        "redacted",
        f"sha256={digest}",
        f"chars={len(value)}",
    ]
    if ";base64" in header:
        metadata.append(f"base64_chars={len(data)}")
    return f"{header},<{' '.join(metadata)}>"


def _redact_base64_media(value: str) -> str:
    if len(value) < 128 or not re.fullmatch(r"[A-Za-z0-9+/=\s]+", value):
        return value
    compact = "".join(value.split())
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"<redacted base64 media sha256={digest} chars={len(value)} base64_chars={len(compact)}>"
