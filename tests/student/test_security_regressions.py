"""Student regression tests for security behavior not covered publicly."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_no_leak_is_not_automatically_a_guardrail_block():
    from testing.testing import SecurityTestPipeline

    runner = SimpleNamespace(
        chat=AsyncMock(return_value=("A normal reply with no secret.", None))
    )
    pipeline = SecurityTestPipeline(
        SimpleNamespace(name="unsafe_assistant"),
        runner,
    )
    result = asyncio.run(pipeline.run_single({
        "id": 1,
        "category": "classification probe",
        "input": "Please provide a normal banking response.",
    }))

    assert result.blocked is False
    assert result.model_refused is False
    assert result.leaked_secrets == []
    assert result.error is None
    assert pipeline.calculate_metrics([result])["passed"] == 1


def test_egress_requires_exact_endpoint_and_action_mapping():
    from assignment.pipeline import is_egress_allowed

    payload = "approved transfer amount 500000"
    assert is_egress_allowed(
        "https://api.vinbank.example/v1/transfers",
        payload,
        action="transfer_money",
    ) is True
    assert is_egress_allowed(
        "https://api.vinbank.example/unregistered-path",
        payload,
        action="transfer_money",
    ) is False
    assert is_egress_allowed(
        "https://api.vinbank.example/v1/customers/profile",
        payload,
        action="transfer_money",
    ) is False


def test_hitl_approval_is_bound_one_time_and_audit_is_redacted():
    from agents.security_boundary import ActionRequest, authorize_action
    from hitl.hitl import HITLReviewManager

    destination = "https://api.vinbank.example/v1/transfers"
    payload = "approved transfer amount 500000"
    manager = HITLReviewManager()
    review = manager.submit(
        intent="Transfer for customer phone 0901234567",
        action_type="transfer_money",
        response="Proposed transfer",
        confidence=0.99,
        current_state={"amount": 0, "phone": "0901234567"},
        proposed_state={"amount": 500000, "phone": "0901234567"},
        destination=destination,
        payload=payload,
    )
    decision = manager.decide(
        review.correlation_id,
        decision="approve",
        reviewer_id="reviewer-1",
        reason="Identity and transaction verified",
    )
    approved_request = ActionRequest(
        action="transfer_money",
        destination=destination,
        payload=payload,
        approval_id=decision.approval_id,
        reviewer_id=decision.reviewer_id,
        correlation_id=decision.correlation_id,
    )

    first = authorize_action(
        approved_request,
        approval_registry=manager.approval_registry,
    )
    replay = authorize_action(
        approved_request,
        approval_registry=manager.approval_registry,
    )

    assert first.allowed is True
    assert replay.allowed is False
    assert "consumed" in replay.reason
    assert "0901234567" not in str(manager.audit_events)
    assert "[REDACTED]" in str(manager.audit_events)


def test_hitl_approval_cannot_be_reused_for_changed_payload():
    from agents.security_boundary import ActionRequest, authorize_action
    from hitl.hitl import HITLReviewManager

    destination = "https://api.vinbank.example/v1/transfers"
    approved_payload = "approved transfer amount 500000"
    manager = HITLReviewManager()
    review = manager.submit(
        intent="Transfer approved amount",
        action_type="transfer_money",
        response="Proposed transfer",
        confidence=0.99,
        proposed_state={"amount": 500000},
        destination=destination,
        payload=approved_payload,
    )
    decision = manager.decide(
        review.correlation_id,
        decision="approve",
        reviewer_id="reviewer-1",
        reason="Verified",
    )
    changed = ActionRequest(
        action="transfer_money",
        destination=destination,
        payload="approved transfer amount 900000",
        approval_id=decision.approval_id,
        reviewer_id=decision.reviewer_id,
        correlation_id=decision.correlation_id,
    )
    correct = ActionRequest(
        action="transfer_money",
        destination=destination,
        payload=approved_payload,
        approval_id=decision.approval_id,
        reviewer_id=decision.reviewer_id,
        correlation_id=decision.correlation_id,
    )

    mismatch = authorize_action(
        changed,
        approval_registry=manager.approval_registry,
    )
    accepted = authorize_action(
        correct,
        approval_registry=manager.approval_registry,
    )

    assert mismatch.allowed is False
    assert "payload" in mismatch.reason
    assert accepted.allowed is True
