"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    @staticmethod
    def _safe_text(text: str) -> str:
        """Redact PII/secrets before they are persisted in audit artifacts."""
        from guardrails.output_guardrails import content_filter

        return content_filter(text or "")["redacted"]

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Open a correlated audit event and return its request ID."""
        request_id = request_id or uuid4().hex
        if request_id in self._open:
            raise ValueError(f"request_id is already open: {request_id}")
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": user_id or "anonymous",
            "input": self._safe_text(text),
            "started_at": utc_now_iso(),
            "started_perf": perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close an audit event with its output, decision and latency."""
        if request_id is None:
            request_id = next(
                (
                    key
                    for key, value in reversed(list(self._open.items()))
                    if value["user_id"] == (user_id or "anonymous")
                ),
                None,
            )
        request_id = request_id or uuid4().hex
        opened = self._open.pop(request_id, None)
        now_perf = perf_counter()
        started_perf = opened["started_perf"] if opened else now_perf
        event = {
            "request_id": request_id,
            "user_id": user_id or "anonymous",
            "input": opened["input"] if opened else "",
            "output": self._safe_text(text),
            "blocked": bool(blocked),
            "layer": layer,
            "started_at": opened["started_at"] if opened else utc_now_iso(),
            "completed_at": utc_now_iso(),
            "latency_ms": round(max(0.0, now_perf - started_perf) * 1000, 3),
        }
        self.logs.append(event)
        return event

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write logs to disk (JSON array)."""
        path = Path(filepath)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
