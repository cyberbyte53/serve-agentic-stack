"""Unit tests for contextual nudge messages (issue #25) and scheduler wiring."""
from datetime import datetime, timedelta

import pytest

from app.schemas import AgentType
from app.service import nudge_messages as nm
from app.service import nudge_scheduler as ns

from conftest import FakeMCP, make_silent_session, wire_scheduler


# ── Bucket resolution ────────────────────────────────────────────────────────

@pytest.mark.parametrize("context,expected", [
    ({"workflow": "new_volunteer_onboarding", "active_agent": "onboarding",
      "stage": "eligibility_screening"}, AgentType.ONBOARDING),
    ({"workflow": "new_volunteer_onboarding", "active_agent": "selection",
      "stage": "onboarding_complete"}, AgentType.SELECTION),
    ({"workflow": "returning_volunteer", "active_agent": "engagement",
      "stage": "matching_ready"}, AgentType.ENGAGEMENT),
    ({"workflow": "returning_volunteer", "active_agent": "fulfillment",
      "stage": "active"}, AgentType.FULFILLMENT),
    ({"workflow": "delivery_support", "active_agent": "delivery_assistant",
      "stage": "delivery_operations"}, AgentType.DELIVERY_ASSISTANT),
    ({"workflow": "returning_volunteer", "active_agent": "engagement",
      "stage": "matching_ready", "sessions_completed": 2}, nm.TAUGHT_BEFORE),
    ({"workflow": "need_coordination", "active_agent": "need",
      "stage": "drafting_need"}, nm.GENERIC),
    ({}, nm.GENERIC),
])
def test_resolve_bucket(context, expected):
    assert nm.resolve_bucket(context) == expected


def test_onboarding_phrases_cover_registry_stages():
    """The phrase map must cover exactly the onboarding-owned stages in
    WORKFLOW_REGISTRY — if a stage is added/renamed there, this fails
    instead of the copy silently drifting."""
    assert set(nm._ONBOARDING_STEP_PHRASES) == nm._ONBOARDING_STAGES


@pytest.mark.parametrize("context,expected", [
    # No active_agent at all — resolution must come from the registry's
    # stage -> responsible_agent mapping.
    ({"workflow": "new_volunteer_onboarding", "stage": "contact_capture"}, AgentType.ONBOARDING),
    ({"workflow": "returning_volunteer", "stage": "active"}, AgentType.FULFILLMENT),
    ({"workflow": "recommended_volunteer", "stage": "verifying_identity"}, AgentType.ENGAGEMENT),
    ({"workflow": "delivery_support", "stage": "delivery_operations"}, AgentType.DELIVERY_ASSISTANT),
    ({"workflow": "need_coordination", "stage": "drafting_need"}, nm.GENERIC),
])
def test_bucket_resolved_from_workflow_registry(context, expected):
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


# ── Exhaustive stage coverage ────────────────────────────────────────────────

def _all_registry_stages():
    for wf_id, wf in nm.WORKFLOW_REGISTRY.items():
        for stage_id in wf.stages:
            yield wf_id, stage_id


@pytest.mark.parametrize("wf_id,stage_id", list(_all_registry_stages()))
def test_every_registry_stage_produces_a_safe_message(wf_id, stage_id):
    """Every stage a volunteer can sit in must compose a message without
    crashing and without leaking the raw stage id (e.g. 'delivery_operations')
    into volunteer-facing copy."""
    ctx = {"workflow": wf_id, "stage": stage_id, "volunteer_name": "Asha Verma"}
    for n in (1, 2, 3):
        msg = nm.build_nudge_message(ctx, n)
        assert msg
        assert stage_id not in msg, f"raw stage leaked for {wf_id}/{stage_id}"
        if nm.resolve_bucket(ctx) == nm.GENERIC:
            # Non-volunteer journeys (e.g. need coordination) get the
            # original generic copy verbatim — no name personalisation.
            assert msg == nm.GENERIC_MESSAGES[n]
        else:
            assert msg.startswith("Hi Asha!")


def test_registry_stage_count_is_pinned():
    # Guards the parametrize above against silently going empty/partial
    # if the registry import shape changes.
    assert len(list(_all_registry_stages())) >= 40
