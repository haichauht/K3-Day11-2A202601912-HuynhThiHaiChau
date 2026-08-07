"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 11: Confidence Router
  TODO 12: Design 3 HITL decision points
"""
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from agents.security_boundary import (
    ActionRequest,
    ApprovalRegistry,
    payload_fingerprint,
)


# ============================================================
# TODO 11: Implement ConfidenceRouter
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        # TODO 11: Implement routing logic
        #
        # 1. Check if action_type is in HIGH_RISK_ACTIONS
        #    -> If yes: always escalate (action="escalate", priority="high",
        #       requires_human=True, reason="High-risk action: {action_type}")
        #
        # 2. Check confidence thresholds:
        #    - confidence >= 0.9:
        #      action="auto_send", priority="low",
        #      requires_human=False, reason="High confidence"
        #
        #    - 0.7 <= confidence < 0.9:
        #      action="queue_review", priority="normal",
        #      requires_human=True, reason="Medium confidence — needs review"
        #
        #    - confidence < 0.7:
        #      action="escalate", priority="high",
        #      requires_human=True, reason="Low confidence — escalating"

        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidence must be a number between 0 and 1")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        normalized_action = (action_type or "general").strip().casefold()
        if normalized_action in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {normalized_action}",
                priority="high",
                requires_human=True,
            )
        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )
        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )
        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


# ============================================================
# TODO 12: Design 3 HITL decision points + a review lifecycle
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "Money movement approval",
        "trigger": "Any transfer_money action, regardless of model confidence or amount.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Verified customer, source/destination accounts, amount, currency, fraud signals, user intent, and exact transaction diff.",
        "example": "The assistant proposes transferring 50,000,000 VND to a newly added beneficiary.",
        "approval_path": "Approve issues a short-lived approval ID bound to the exact transaction; reject cancels it; timeout fails closed without sending funds.",
        "audit_fields": "correlation_id, request_id, user_id, intent, source, destination, amount, current/proposed diff, reviewer_id, decision, reason, timestamp, approval_id",
    },
    {
        "id": 2,
        "name": "Account and identity changes",
        "trigger": "close_account, change_password, delete_data, or update_personal_info is proposed.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Authentication strength, recent account activity, fraud flags, current values, redacted proposed values, and downstream impact.",
        "example": "A user asks to replace the registered phone number immediately after a failed login series.",
        "approval_path": "Approve binds the reviewer to the exact field-level diff; reject preserves current data; timeout leaves the account unchanged and opens a case.",
        "audit_fields": "correlation_id, request_id, user_id, intent, authentication evidence, field-level diff, risk signals, reviewer_id, decision, reason, timestamp",
    },
    {
        "id": 3,
        "name": "Low-confidence or conflicting policy answer",
        "trigger": "Confidence is below 0.9, the Judge flags uncertainty, or retrieved sources conflict on a customer-impacting answer.",
        "hitl_model": "human-as-tiebreaker",
        "context_needed": "Customer question, proposed answer, confidence, citations with provenance, conflicting facts, guardrail/Judge verdicts, and answer diff.",
        "example": "Two policy documents disagree about whether an early loan repayment fee applies.",
        "approval_path": "Approve releases the reviewed answer; reject replaces it with an escalation message; timeout sends no unverified claim and queues follow-up.",
        "audit_fields": "correlation_id, request_id, user_id, intent, response draft, source provenance, confidence, judge verdict, reviewer_id, decision, edited diff, timestamp",
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_for_audit(value):
    """Recursively redact PII/secrets before persisting HITL audit fields."""
    if isinstance(value, dict):
        return {key: _redact_for_audit(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_for_audit(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_for_audit(item) for item in value)
    if isinstance(value, str):
        from guardrails.output_guardrails import content_filter

        return content_filter(value)["redacted"]
    return value


@dataclass
class ReviewRequest:
    """Proposal context shown to a reviewer before any side effect."""

    correlation_id: str
    intent: str
    action_type: str
    response: str
    confidence: float
    current_state: dict
    proposed_state: dict
    reviewer_context: dict
    destination: str = ""
    payload: str = ""
    method: str = "POST"
    status: str = "pending"
    created_at: str = field(default_factory=_utc_now)

    @property
    def proposed_diff(self) -> dict:
        keys = set(self.current_state) | set(self.proposed_state)
        return {
            key: {
                "before": self.current_state.get(key),
                "after": self.proposed_state.get(key),
            }
            for key in sorted(keys)
            if self.current_state.get(key) != self.proposed_state.get(key)
        }


@dataclass(frozen=True)
class ReviewDecision:
    correlation_id: str
    decision: str
    reviewer_id: str | None
    reason: str
    decided_at: str
    approval_id: str | None = None
    approval_expires_at: str | None = None


class HITLReviewManager:
    """Fail-closed approve/reject/timeout lifecycle with an audit trail."""

    def __init__(
        self,
        *,
        approval_registry: ApprovalRegistry | None = None,
        approval_ttl_seconds: int = 300,
    ):
        if approval_ttl_seconds < 1:
            raise ValueError("approval_ttl_seconds must be at least 1")
        self.requests: dict[str, ReviewRequest] = {}
        self.audit_events: list[dict] = []
        self.approval_registry = approval_registry or ApprovalRegistry()
        self.approval_ttl_seconds = approval_ttl_seconds

    def submit(
        self,
        *,
        intent: str,
        action_type: str,
        response: str,
        confidence: float,
        current_state: dict | None = None,
        proposed_state: dict | None = None,
        reviewer_context: dict | None = None,
        destination: str | None = None,
        payload: str | None = None,
        method: str = "POST",
    ) -> ReviewRequest:
        routing = ConfidenceRouter().route(response, confidence, action_type)
        if not routing.requires_human:
            raise ValueError("This proposal does not require human review")
        current = dict(current_state or {})
        proposed = dict(proposed_state or {})
        context = dict(reviewer_context or {})
        resolved_destination = destination or str(context.get("destination", ""))
        resolved_payload = payload
        if resolved_payload is None:
            resolved_payload = json.dumps(
                proposed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        request = ReviewRequest(
            correlation_id=f"REQ-{uuid4().hex[:12].upper()}",
            intent=intent,
            action_type=action_type,
            response=response,
            confidence=confidence,
            current_state=current,
            proposed_state=proposed,
            reviewer_context=context,
            destination=resolved_destination,
            payload=resolved_payload,
            method=(method or "").upper(),
        )
        self.requests[request.correlation_id] = request
        self.audit_events.append({
            "event": "submitted",
            "correlation_id": request.correlation_id,
            "intent": _redact_for_audit(intent),
            "action_type": action_type,
            "proposed_diff": _redact_for_audit(request.proposed_diff),
            "destination": request.destination,
            "method": request.method,
            "payload_digest": payload_fingerprint(request.payload),
            "timestamp": request.created_at,
        })
        return request

    def decide(
        self,
        correlation_id: str,
        *,
        decision: str,
        reviewer_id: str,
        reason: str,
    ) -> ReviewDecision:
        normalized = decision.strip().casefold()
        if normalized not in {"approve", "reject"}:
            raise ValueError("decision must be 'approve' or 'reject'")
        reviewer_id = (reviewer_id or "").strip()
        reason = (reason or "").strip()
        if not reviewer_id:
            raise ValueError("reviewer_id is required")
        if not reason:
            raise ValueError("reason is required")
        request = self._pending(correlation_id)
        new_status = "approved" if normalized == "approve" else "rejected"
        approval_id = None
        approval_expires_at = None
        if normalized == "approve":
            approval = self.approval_registry.issue(
                ActionRequest(
                    action=request.action_type,
                    destination=request.destination,
                    payload=request.payload,
                    method=request.method,
                ),
                reviewer_id=reviewer_id,
                correlation_id=request.correlation_id,
                ttl_seconds=self.approval_ttl_seconds,
            )
            approval_id = approval.approval_id
            approval_expires_at = approval.expires_at.isoformat()
        request.status = new_status
        result = ReviewDecision(
            correlation_id=correlation_id,
            decision=request.status,
            reviewer_id=reviewer_id,
            reason=reason,
            decided_at=_utc_now(),
            approval_id=approval_id,
            approval_expires_at=approval_expires_at,
        )
        self._record_decision(request, result)
        return result

    def timeout(
        self,
        correlation_id: str,
        reason: str = "review timeout",
    ) -> ReviewDecision:
        request = self._pending(correlation_id)
        request.status = "timed_out"
        result = ReviewDecision(
            correlation_id=correlation_id,
            decision="timed_out",
            reviewer_id=None,
            reason=reason,
            decided_at=_utc_now(),
        )
        self._record_decision(request, result)
        return result

    def _pending(self, correlation_id: str) -> ReviewRequest:
        if correlation_id not in self.requests:
            raise KeyError(f"Unknown correlation_id: {correlation_id}")
        request = self.requests[correlation_id]
        if request.status != "pending":
            raise ValueError(f"Review is already {request.status}")
        return request

    def _record_decision(
        self,
        request: ReviewRequest,
        result: ReviewDecision,
    ):
        self.audit_events.append({
            "event": "decision",
            "correlation_id": result.correlation_id,
            "intent": _redact_for_audit(request.intent),
            "action_type": request.action_type,
            "proposed_diff": _redact_for_audit(request.proposed_diff),
            "destination": request.destination,
            "method": request.method,
            "payload_digest": payload_fingerprint(request.payload),
            "decision": result.decision,
            "reviewer_id": result.reviewer_id,
            "reason": result.reason,
            "approval_id": result.approval_id,
            "approval_expires_at": result.approval_expires_at,
            "timestamp": result.decided_at,
        })


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
