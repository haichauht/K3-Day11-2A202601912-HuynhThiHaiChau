"""Reference security boundary for the Day 11 Guards Agent.

This is deliberately framework-independent so students can inspect the policy
and reason about the difference between untrusted content and an authorised
action. It is not a solution for the TODOs in ``src/assignment``.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from urllib.parse import urlsplit
from uuid import uuid4


TRUSTED_EGRESS_ENDPOINTS = frozenset({
    ("api.vinbank.example", "/v1/transfers"),
    ("api.vinbank.example", "/v1/accounts/close"),
    ("api.vinbank.example", "/v1/accounts/change-password"),
    ("api.vinbank.example", "/v1/customers/profile"),
    ("api.vinbank.example", "/v1/customers/delete-data"),
    ("cases.vinbank.example", "/v1/cases"),
})
ACTION_ENDPOINTS = {
    "transfer_money": ("api.vinbank.example", "/v1/transfers"),
    "close_account": ("api.vinbank.example", "/v1/accounts/close"),
    "change_password": (
        "api.vinbank.example",
        "/v1/accounts/change-password",
    ),
    "update_personal_info": (
        "api.vinbank.example",
        "/v1/customers/profile",
    ),
    "delete_data": ("api.vinbank.example", "/v1/customers/delete-data"),
    "create_case": ("cases.vinbank.example", "/v1/cases"),
}
HIGH_RISK_ACTIONS = frozenset({
    "transfer_money", "close_account", "change_password",
    "delete_data", "update_personal_info",
})
ZERO_WIDTH = "\u200b\u200c\u200d\ufeff\u2060"
SECRET_PATTERNS = (
    r"\badmin123\b",
    r"sk-[a-z0-9-]{8,}",
    r"db\.vinbank\.internal(?::\d+)?",
    r"(?:password|mật\s*khẩu)\s*[:=]\s*\S+",
)
INSTRUCTION_OVERRIDE_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)?\s*instructions?",
    r"(?:system|developer)\s+(?:prompt|instruction)|system\s+override",
    r"(?:reveal|disclose|translate|encode|summarize)\b.*(?:secret|password|credential|api\s*key|internal)",
    r"bỏ\s+qua\s+(?:mọi\s+)?hướng\s+dẫn|tiết\s+lộ\s+(?:mật\s*khẩu|api|thông\s*tin\s*nội\s*bộ)",
)


@dataclass(frozen=True)
class ExternalContent:
    """Data retrieved from email/RAG/web; never an instruction authority."""

    source: str
    text: str
    trusted: bool = False


@dataclass(frozen=True)
class ActionRequest:
    """A proposed side effect awaiting deterministic policy and human approval."""

    action: str
    destination: str
    payload: str
    approval_id: str | None = None
    reviewer_id: str | None = None
    correlation_id: str | None = None
    method: str = "POST"


@dataclass(frozen=True)
class ActionDecision:
    allowed: bool
    reason: str
    requires_human: bool


@dataclass
class ApprovalRecord:
    """A one-time reviewer approval bound to one exact proposed side effect."""

    approval_id: str
    reviewer_id: str
    correlation_id: str
    action: str
    destination: str
    method: str
    payload_digest: str
    issued_at: datetime
    expires_at: datetime
    consumed_at: datetime | None = None


def payload_fingerprint(payload: str) -> str:
    """Return a non-reversible digest suitable for approval/audit binding."""
    return hashlib.sha256((payload or "").encode("utf-8")).hexdigest()


def _canonical_destination(destination: str) -> str:
    """Canonical form used when binding a destination to an approval."""
    parsed = urlsplit(destination or "")
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port not in (None, 443) else ""
    return f"{parsed.scheme.lower()}://{host}{port}{parsed.path}"


class ApprovalRegistry:
    """Issue, validate and consume short-lived exact-diff approvals."""

    def __init__(self):
        self.records: dict[str, ApprovalRecord] = {}
        self.audit_events: list[dict] = []
        self._lock = Lock()

    def issue(
        self,
        request: ActionRequest,
        *,
        reviewer_id: str,
        correlation_id: str,
        ttl_seconds: int = 300,
        now: datetime | None = None,
    ) -> ApprovalRecord:
        reviewer_id = (reviewer_id or "").strip()
        correlation_id = (correlation_id or "").strip()
        if not reviewer_id:
            raise ValueError("reviewer_id is required")
        if not correlation_id:
            raise ValueError("correlation_id is required")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")

        issued_at = now or datetime.now(timezone.utc)
        record = ApprovalRecord(
            approval_id=f"HITL-{uuid4().hex[:8].upper()}",
            reviewer_id=reviewer_id,
            correlation_id=correlation_id,
            action=(request.action or "").strip().casefold(),
            destination=_canonical_destination(request.destination),
            method=(request.method or "").upper(),
            payload_digest=payload_fingerprint(request.payload),
            issued_at=issued_at,
            expires_at=issued_at + timedelta(seconds=ttl_seconds),
        )
        with self._lock:
            self.records[record.approval_id] = record
            self.audit_events.append({
                "event": "approval_issued",
                "approval_id": record.approval_id,
                "correlation_id": record.correlation_id,
                "reviewer_id": record.reviewer_id,
                "action": record.action,
                "destination": record.destination,
                "method": record.method,
                "payload_digest": record.payload_digest,
                "issued_at": record.issued_at.isoformat(),
                "expires_at": record.expires_at.isoformat(),
            })
        return record

    def validate_and_consume(
        self,
        request: ActionRequest,
        *,
        now: datetime | None = None,
        consume: bool = True,
    ) -> tuple[bool, str]:
        """Validate exact binding/expiry and atomically consume the approval."""
        checked_at = now or datetime.now(timezone.utc)
        approval_id = (request.approval_id or "").strip()
        with self._lock:
            record = self.records.get(approval_id)
            if record is None:
                return False, "approval is unknown or was not issued by the reviewer workflow"
            if record.consumed_at is not None:
                return False, "approval has already been consumed"
            if checked_at >= record.expires_at:
                return False, "approval has expired"
            if (request.reviewer_id or "").strip() != record.reviewer_id:
                return False, "approval reviewer does not match"
            if (request.correlation_id or "").strip() != record.correlation_id:
                return False, "approval correlation ID does not match"
            if (request.action or "").strip().casefold() != record.action:
                return False, "approval action does not match"
            if _canonical_destination(request.destination) != record.destination:
                return False, "approval destination does not match"
            if (request.method or "").upper() != record.method:
                return False, "approval HTTP method does not match"
            if payload_fingerprint(request.payload) != record.payload_digest:
                return False, "approval payload/diff does not match"

            if consume:
                record.consumed_at = checked_at
                self.audit_events.append({
                    "event": "approval_consumed",
                    "approval_id": record.approval_id,
                    "correlation_id": record.correlation_id,
                    "action": record.action,
                    "destination": record.destination,
                    "payload_digest": record.payload_digest,
                    "consumed_at": checked_at.isoformat(),
                })
        return True, "recorded approval matches the exact action and payload"


def normalize_for_security(text: str) -> str:
    """Canonicalize Unicode and remove invisible separators before policy checks."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return normalized.translate(str.maketrans("", "", ZERO_WIDTH))


def contains_secret(text: str) -> bool:
    """Detect a synthetic lab secret even when punctuation/spacing is altered."""
    normalized = re.sub(r"[^a-z0-9]", "", normalize_for_security(text).casefold())
    secrets = ("admin123", "skvinbanksecret2024", "dbvinbankinternal")
    return any(secret in normalized for secret in secrets) or any(
        re.search(pattern, normalize_for_security(text), re.IGNORECASE)
        for pattern in SECRET_PATTERNS
    )


def contains_instruction_override(text: str) -> bool:
    """Identify instruction-like text after Unicode normalization."""
    normalized = normalize_for_security(text)
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in INSTRUCTION_OVERRIDE_PATTERNS)


def assess_external_content(content: ExternalContent) -> ActionDecision:
    """Treat third-party content as data and reject attempts to change policy."""
    if content.trusted:
        return ActionDecision(True, "trusted source metadata", False)
    if contains_instruction_override(content.text):
        return ActionDecision(False, "untrusted content contains an instruction override", False)
    return ActionDecision(True, "untrusted content is data only", False)


def authorize_action(
    request: ActionRequest,
    *,
    approval_registry: ApprovalRegistry | None = None,
    consume_approval: bool = True,
) -> ActionDecision:
    """Enforce exact endpoint, payload DLP and bound HITL approval at the sink."""
    try:
        destination = urlsplit(request.destination or "")
        port = destination.port
    except (TypeError, ValueError):
        return ActionDecision(False, "destination is malformed", False)

    endpoint = (destination.hostname, destination.path)
    if (
        destination.scheme.lower() != "https"
        or endpoint not in TRUSTED_EGRESS_ENDPOINTS
        or destination.username is not None
        or destination.password is not None
        or port not in (None, 443)
        or bool(destination.query)
        or bool(destination.fragment)
        or (request.method or "").upper() != "POST"
    ):
        return ActionDecision(False, "destination endpoint is not allowlisted", False)

    normalized_action = (request.action or "").strip().casefold()
    expected_endpoint = ACTION_ENDPOINTS.get(normalized_action)
    if expected_endpoint is not None and expected_endpoint != endpoint:
        return ActionDecision(False, "action is not allowed at this endpoint", False)
    if contains_secret(request.payload):
        return ActionDecision(False, "payload contains protected data", False)
    if normalized_action in HIGH_RISK_ACTIONS:
        if approval_registry is None:
            return ActionDecision(
                False,
                "high-risk action needs a workflow-issued bound approval",
                True,
            )
        approved, reason = approval_registry.validate_and_consume(
            request,
            consume=consume_approval,
        )
        if not approved:
            return ActionDecision(False, reason, True)
    return ActionDecision(True, "least-privilege policy permits this action", False)
