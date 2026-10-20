"""Unit tests for SessionService row serialisation (issue #25, Bug 1a).

_row_to_dict is the single place session rows become dicts for MCP
tool callers (list_sessions, get_session, ...). Anything it omits is
invisible to the orchestrator.
"""
from datetime import datetime
from types import SimpleNamespace

from services.session_service import SessionService


def _row(**overrides):
    now = datetime.utcnow()
    row = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        actor_id="919999999999", identity_type="phone",
        channel="whatsapp", persona="new_volunteer", user_type="new_user",
        volunteer_id=None, coordinator_id=None,
        workflow="new_volunteer_onboarding", active_agent="onboarding",
        status="active", stage="eligibility_screening", sub_state=None,
        context_summary=None, channel_metadata={}, idempotency_key=None,
        created_at=now, updated_at=now, last_message_at=now,
        do_not_disturb=False,
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def test_row_to_dict_exposes_do_not_disturb():
    """Bug 1a proof: the scheduler reads session["do_not_disturb"] to skip
    opted-out volunteers. Before the fix _row_to_dict omitted the field,
    so the check always saw None and 'stop' replies were re-nudged."""
    result = SessionService.__new__(SessionService)._row_to_dict(
        _row(do_not_disturb=True)
    )
    assert result["do_not_disturb"] is True


def test_row_to_dict_do_not_disturb_defaults_false():
    result = SessionService.__new__(SessionService)._row_to_dict(_row())
    assert result["do_not_disturb"] is False
