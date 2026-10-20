"""Bug proofs for issue #25 (Smart Nudges) — written BEFORE any fix.

Each test asserts the CORRECT behaviour. At commit 452eb1b they failed
against the unmodified code for the reasons in each docstring; they now
serve as the regression suite. (The _row_to_dict proof lives with the
MCP server's own tests: serve-mcp-server/tests/test_session_service.py.)
"""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main
from app.service import nudge_scheduler as ns

from conftest import (
    PHONE, SESSION_ID, FakeMCP, make_silent_session, wire_scheduler,
)


@pytest.fixture
def client():
    return TestClient(main.app)


# ─── Bug 1: `stop` opt-out is broken end to end ─────────────────────────────

async def test_bug1b_mark_do_not_disturb_persists_the_flag(monkeypatch):
    """mark_do_not_disturb must write the flag via an MCP tool.

    Before the fix it only cancelled pending nudges; no MCP tool wrote
    sessions.do_not_disturb, so nothing stopped future scheduling.
    """
    fake = FakeMCP([make_silent_session()])
    monkeypatch.setattr(ns, "_call_mcp_tool", fake.call)

    await ns.mark_do_not_disturb(SESSION_ID)

    dnd_writes = [t for t in fake.tools_called()
                  if "do_not_disturb" in t or "dnd" in t]
    assert dnd_writes, (
        f"no MCP call persists do_not_disturb; calls were {fake.tools_called()}"
    )
    assert fake.sessions[SESSION_ID]["do_not_disturb"] is True


# ─── Bug 2: reply cancellation in the webhook is never awaited ──────────────

def test_bug2_webhook_reply_cancels_pending_nudges(client, monkeypatch):
    """A WhatsApp reply must cancel the volunteer's pending nudges.

    Before the fix, wa_receive called cancel_nudges_for_session without
    await: the coroutine was created and discarded ("coroutine was
    never awaited") and the cancellation never happened. Uses the
    'stop' path, which exits the handler right after the nudge handling
    so the test stays hermetic.
    """
    fake = FakeMCP([make_silent_session()])
    wire_scheduler(monkeypatch, fake)
    monkeypatch.setattr(main, "_wa_send", AsyncMock(return_value=True))
    monkeypatch.setitem(main._wa_sessions, PHONE, SESSION_ID)

    payload = {"entry": [{"changes": [{"value": {"messages": [{
        "type": "text", "from": PHONE, "id": "wamid.1",
        "text": {"body": "stop"},
    }]}}]}]}
    response = client.post("/api/whatsapp/webhook", json=payload)

    assert response.status_code == 200
    assert "nudge_cancel_for_session" in fake.tools_called()
    assert "nudge_set_do_not_disturb" in fake.tools_called()


async def test_bug2_control_awaited_cancel_actually_cancels(monkeypatch):
    """Control: the cancel function itself works when awaited.

    This passed even before the fix, isolating Bug 2 to the call site
    in main.py rather than the cancellation logic.
    """
    fake = FakeMCP([make_silent_session()])
    fake.nudges.append({
        "id": "nudge-x", "session_id": SESSION_ID, "volunteer_phone": PHONE,
        "volunteer_name": "", "nudge_number": 1,
        "scheduled_at": datetime.utcnow().isoformat(),
        "sent_at": None, "cancelled": False,
    })
    monkeypatch.setattr(ns, "_call_mcp_tool", fake.call)

    await ns.cancel_nudges_for_session(SESSION_ID)

    assert "nudge_cancel_for_session" in fake.tools_called()
    assert fake.nudges[0]["cancelled"] is True


# ─── Bug 3: frequency cap leaks and nothing auto-pauses ─────────────────────

async def test_bug3a_no_new_cycle_after_third_nudge(monkeypatch):
    """After nudges #1-#3 have all been sent, a still-silent session must
    not be scheduled a fresh nudge #1.

    Before the fix, check_and_schedule_new_nudges re-scheduled #1 for
    any active silent session; the real nudge_schedule tool only dedups
    UNSENT rows, so the 1h/8h/24h cycle restarted forever. The
    docstring's promise — "After 3 nudges with no reply, session is
    auto-paused" — is what stops this (see Bug 3b).
    """
    fake = FakeMCP([make_silent_session()])
    wire_scheduler(monkeypatch, fake)

    # Scheduler discovers the silent session and queues nudge #1.
    await ns.check_and_schedule_new_nudges()
    # Each process pass sends the due nudge and queues the next one
    # (all scheduled times are in the past: session silent 25h).
    for _ in range(3):
        await ns.process_due_nudges()
    sent_numbers = sorted(
        n["nudge_number"] for n in fake.nudges if n["sent_at"] is not None
    )
    assert sent_numbers == [1, 2, 3], f"setup failed, sent: {sent_numbers}"

    # Next scheduler cycle must not start over.
    await ns.check_and_schedule_new_nudges()
    leftover = fake.unsent_nudges()
    assert leftover == [], (
        f"a new nudge cycle was scheduled after nudge #3: {leftover}"
    )


async def test_bug3c_no_duplicate_first_nudge_between_cycles(monkeypatch):
    """Repeated scheduler scans during one silence episode must not
    re-queue nudge #1 once it has been sent.

    Found by live E2E (delays 1m/2m/3m): nudge #1 was delivered twice,
    because check_and_schedule_new_nudges runs every cycle and the
    server only deduped UNSENT rows. With production delays this
    would re-send #1 every 5 minutes until nudge #2 fell due. The
    server now dedups nudge #1 per silence episode (episode_start).
    """
    fake = FakeMCP([make_silent_session()])
    wire_scheduler(monkeypatch, fake)

    await ns.check_and_schedule_new_nudges()
    await ns.process_due_nudges()          # sends #1, queues #2
    await ns.check_and_schedule_new_nudges()  # next scan, same episode
    await ns.process_due_nudges()          # must NOT send #1 again

    sent_first = [n for n in fake.nudges
                  if n["nudge_number"] == 1 and n["sent_at"] is not None]
    assert len(sent_first) == 1, f"nudge #1 sent {len(sent_first)} times"
    unsent_first = [n for n in fake.unsent_nudges() if n["nudge_number"] == 1]
    assert unsent_first == [], f"duplicate nudge #1 queued: {unsent_first}"


async def test_bug3b_session_paused_after_third_nudge(monkeypatch):
    """Sending nudge #3 must pause the session (scheduler docstring:
    "After 3 nudges with no reply, session is auto-paused").

    Before the fix, process_due_nudges only marked the nudge sent; the
    session stayed active, which also enabled the Bug 3a re-scheduling.
    """
    fake = FakeMCP([make_silent_session()])
    wire_scheduler(monkeypatch, fake)
    fake.nudges.append({
        "id": "nudge-3", "session_id": SESSION_ID, "volunteer_phone": PHONE,
        "volunteer_name": "", "nudge_number": 3,
        "scheduled_at": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
        "sent_at": None, "cancelled": False,
    })

    sent = await ns.process_due_nudges()

    assert sent == 1  # setup sanity: the due nudge was sent
    assert fake.sessions[SESSION_ID]["status"] == "paused", (
        "session still active after final nudge; "
        f"tools called: {fake.tools_called()}"
    )
