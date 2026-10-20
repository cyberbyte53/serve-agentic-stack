"""Unit tests for contextual nudge messages (issue #25) and scheduler wiring."""
from datetime import datetime, timedelta

import pytest

from app.service import nudge_messages as nm
from app.service import nudge_scheduler as ns

from conftest import FakeMCP, make_silent_session, wire_scheduler


# ── Bucket resolution ────────────────────────────────────────────────────────

@pytest.mark.parametrize("context,expected", [
    ({"workflow": "new_volunteer_onboarding", "active_agent": "onboarding",
      "stage": "eligibility_screening"}, "onboarding"),
    ({"workflow": "new_volunteer_onboarding", "active_agent": "selection",
      "stage": "onboarding_complete"}, "post_selection"),
    ({"workflow": "returning_volunteer", "active_agent": "engagement",
      "stage": "matching_ready"}, "post_selection"),
    ({"workflow": "returning_volunteer", "active_agent": "fulfillment",
      "stage": "active"}, "matched"),
    ({"workflow": "delivery_support", "active_agent": "delivery_assistant",
      "stage": "delivery_operations"}, "delivery"),
    ({"workflow": "returning_volunteer", "active_agent": "engagement",
      "stage": "matching_ready", "sessions_completed": 2}, "taught_before"),
    ({"workflow": "need_coordination", "active_agent": "need",
      "stage": "drafting_need"}, "generic"),
    ({}, "generic"),
])
def test_resolve_bucket(context, expected):
    assert nm.resolve_bucket(context) == expected


# ── Message content ──────────────────────────────────────────────────────────

def test_generic_fallback_matches_original_copy():
    for n in (1, 2, 3):
        assert nm.build_nudge_message({}, n) == nm.GENERIC_MESSAGES[n]
        assert nm.build_nudge_message({"workflow": "unknown"}, n) == nm.GENERIC_MESSAGES[n]


def test_onboarding_message_references_step_and_first_name():
    ctx = {"workflow": "new_volunteer_onboarding", "active_agent": "onboarding",
           "stage": "eligibility_screening", "volunteer_name": "Asha Verma"}
    msg = nm.build_nudge_message(ctx, 1)
    assert msg.startswith("Hi Asha!")
    assert "background" in msg
    assert "continue" in msg


def test_onboarding_teaching_step_phrase():
    ctx = {"workflow": "new_volunteer_onboarding", "stage": "teaching_profile"}
    assert "subjects" in nm.build_nudge_message(ctx, 1)


def test_post_selection_message_matches_issue_example():
    ctx = {"workflow": "returning_volunteer", "active_agent": "engagement",
           "stage": "matching_ready"}
    msg = nm.build_nudge_message(ctx, 1)
    assert "shown interest in teaching" in msg
    assert msg.startswith("Hi!")  # no name known — no awkward placeholder


def test_taught_before_message_acknowledges_past_teaching():
    ctx = {"workflow": "returning_volunteer", "active_agent": "engagement",
           "stage": "re_engaging", "sessions_completed": 1}
    assert "taught with SERVE before" in nm.build_nudge_message(ctx, 1)


def test_escalation_tone_across_numbers():
    ctx = {"workflow": "new_volunteer_onboarding", "stage": "welcome"}
    m1, m2, m3 = (nm.build_nudge_message(ctx, n) for n in (1, 2, 3))
    assert len({m1, m2, m3}) == 3
    assert "last reminder" in m3
    assert "stop" in m2  # opt-out stays visible on the middle nudge too


def test_nudge_number_is_clamped():
    ctx = {"workflow": "new_volunteer_onboarding", "stage": "welcome"}
    assert nm.build_nudge_message(ctx, 99) == nm.build_nudge_message(ctx, 3)
    assert nm.build_nudge_message(ctx, 0) == nm.build_nudge_message(ctx, 1)


# ── Scheduler wiring ─────────────────────────────────────────────────────────

def _due_onboarding_fake():
    session = make_silent_session()
    fake = FakeMCP([session])
    fake.nudges.append({
        "id": "nudge-1", "session_id": session["id"],
        "volunteer_phone": "919999999999", "volunteer_name": "Asha Verma",
        "nudge_number": 1,
        "scheduled_at": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
        "sent_at": None, "cancelled": False,
    })
    return fake


async def test_scheduler_sends_contextual_message(monkeypatch):
    fake = _due_onboarding_fake()
    wire_scheduler(monkeypatch, fake)
    sent_texts = []

    async def capture_send(to, text):
        sent_texts.append(text)
        return True

    monkeypatch.setattr(ns, "_send_whatsapp", capture_send)
    sent = await ns.process_due_nudges()
    assert sent == 1
    assert "eligibility" not in sent_texts[0]  # stage name never leaked raw
    assert "background" in sent_texts[0]       # contextual step phrase used
    assert "onboarding" in sent_texts[0]


async def test_scheduler_dry_run_logs_without_sending(monkeypatch):
    fake = _due_onboarding_fake()
    wire_scheduler(monkeypatch, fake)
    monkeypatch.setattr(ns, "NUDGE_DRY_RUN", True)

    async def forbidden_send(to, text):  # pragma: no cover - must not be called
        raise AssertionError("WhatsApp send attempted in dry-run mode")

    monkeypatch.setattr(ns, "_send_whatsapp", forbidden_send)
    sent = await ns.process_due_nudges()
    assert sent == 1
    assert fake.nudges[0]["sent_at"] is not None
