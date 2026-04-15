from __future__ import annotations

from typing import Any

from evals.models import ErrorRecord


def exception_to_error(exc: BaseException) -> ErrorRecord:
    details: dict[str, Any] = {}
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        details["status_code"] = status_code
    request_id = getattr(exc, "request_id", None)
    if request_id is not None:
        details["request_id"] = request_id
    code = getattr(exc, "code", None)
    if code is not None:
        details["code"] = code
    return ErrorRecord(type=exc.__class__.__name__, message=str(exc), details=details)
