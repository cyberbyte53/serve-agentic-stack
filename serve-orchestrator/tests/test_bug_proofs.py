"""Bug proofs for issue #25 (Smart Nudges) — written BEFORE any fix.

Each test asserts the CORRECT behaviour. Against the unmodified code they
are expected to FAIL, for the reasons stated in each docstring. After the
fix they become the regression suite.

Run:  python -m pytest serve-orchestrator/tests/test_bug_proofs.py -q
"""
import ast
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.service import nudge_scheduler as ns

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = "11111111-1111-1111-1111-111111111111"
PHONE = "919999999999"


# ─── Fake MCP transport ──────────────────────────────────────────────────────

class FakeMCP:
    """In-memory stand-in for the MCP tools the scheduler calls.

    `nudge_schedule` mirrors the real tool's dedup rule in
    serve-mcp-server/main.py: it only dedups UNSENT rows for the same
    session + nudge_number. Sent rows do not block a fresh schedule.
    """

    def __init__(self, sessions):
        self.sessions = {s["id"]: dict(s) for s in sessions}
        self.nudges = []          # rows of the fake nudge_queue table
        self.calls = []           # (tool_name, arguments) in order
        self._seq = 0

    async def call(self, tool, arguments):
        self.calls.append((tool, dict(arguments or {})))
        handler = getattr(self, f"tool_{tool}", None)
        if handler is None:
            return {}
        return handler(**(arguments or {}))

    # -- session tools ------------------------------------------------------
    def tool_list_sessions(self, status=None, limit=50):
        rows = [s for s in self.sessions.values()
                if status is None or s.get("status") == status]
        return {"status": "success", "sessions": [dict(r) for r in rows]}

    def tool_get_session(self, session_id):
        s = self.sessions.get(session_id)
        return {"status": "success", "session": dict(s)} if s else {"status": "error"}

    def tool_advance_session_state(self, session_id, new_state, **kw):
        # Mirrors the planned mapping: stage "paused" -> status "paused".
        s = self.sessions.get(session_id)
        if s:
            s["stage"] = new_state
            if new_state == "paused":
                s["status"] = "paused"
        return {"status": "success", "current_state": new_state}

    # -- nudge queue tools ----------------------------------------------------
    def tool_nudge_schedule(self, session_id, volunteer_phone, nudge_number,
                            scheduled_at, volunteer_name=""):
        for n in self.nudges:
            if (n["session_id"] == session_id
                    and n["nudge_number"] == nudge_number
                    and n["sent_at"] is None and not n["cancelled"]):
                return {"status": "already_scheduled", "id": n["id"]}
        self._seq += 1
        row = {
            "id": f"nudge-{self._seq}",
            "session_id": session_id,
            "volunteer_phone": volunteer_phone,
            "volunteer_name": volunteer_name,
            "nudge_number": nudge_number,
            "scheduled_at": scheduled_at,
            "sent_at": None,
            "cancelled": False,
        }
        self.nudges.append(row)
        return {"status": "success", "id": row["id"]}

    def tool_nudge_get_due(self, now=""):
        current = datetime.fromisoformat(now) if now else datetime.utcnow()
        due = [n for n in self.nudges
               if datetime.fromisoformat(n["scheduled_at"]) <= current
               and n["sent_at"] is None and not n["cancelled"]]
        return {"status": "success", "nudges": [
            {k: n[k] for k in ("id", "session_id", "volunteer_phone",
                               "volunteer_name", "nudge_number", "scheduled_at")}
            for n in due]}

    def tool_nudge_mark_sent(self, nudge_id):
        for n in self.nudges:
            if n["id"] == nudge_id:
                n["sent_at"] = datetime.utcnow().isoformat()
        return {"status": "success"}

    def tool_nudge_cancel_for_session(self, session_id):
        count = 0
        for n in self.nudges:
            if (n["session_id"] == session_id and n["sent_at"] is None
                    and not n["cancelled"]):
                n["cancelled"] = True
                count += 1
        return {"status": "success", "cancelled_count": count}

    # -- helpers for assertions ----------------------------------------------
    def tools_called(self):
        return [t for t, _ in self.calls]

    def unsent_nudges(self):
        return [n for n in self.nudges
                if n["sent_at"] is None and not n["cancelled"]]


def _silent_whatsapp_session(**overrides):
    session = {
        "id": SESSION_ID,
        "channel": "whatsapp",
        "status": "active",
        "workflow": "new_volunteer_onboarding",
        "active_agent": "onboarding",
        "stage": "eligibility_screening",
        "persona": "new_volunteer",
        # Silent for 25 hours: past every nudge delay (1h / 8h / 24h).
        "last_message_at": (datetime.utcnow() - timedelta(hours=25)).isoformat(),
        "channel_metadata": {"volunteer_phone": PHONE},
    }
    session.update(overrides)
    return session


def _wire(monkeypatch, fake):
    monkeypatch.setattr(ns, "_call_mcp_tool", fake.call)
    monkeypatch.setattr(ns, "_is_quiet_hour", lambda: False)

    async def fake_send(to, text):
        return True

    monkeypatch.setattr(ns, "_send_whatsapp", fake_send)


# ─── Bug 1: `stop` opt-out is broken end to end ─────────────────────────────

def test_bug1a_row_to_dict_exposes_do_not_disturb():
    """SessionService._row_to_dict must return do_not_disturb.

    Today it omits the field, so the scheduler's `session.get("do_not_disturb")`
    check in check_and_schedule_new_nudges always sees None and a volunteer
    who replied "stop" keeps being nudged.
    """
    from services.session_service import SessionService

    now = datetime.utcnow()
    row = SimpleNamespace(
        id=SESSION_ID, actor_id=PHONE, identity_type="phone",
        channel="whatsapp", persona="new_volunteer", user_type="new_user",
        volunteer_id=None, coordinator_id=None,
        workflow="new_volunteer_onboarding", active_agent="onboarding",
        status="active", stage="eligibility_screening", sub_state=None,
        context_summary=None, channel_metadata={}, idempotency_key=None,
        created_at=now, updated_at=now, last_message_at=now,
        do_not_disturb=True,
    )
    result = SessionService.__new__(SessionService)._row_to_dict(row)
    assert result.get("do_not_disturb") is True


def test_bug1b_mark_do_not_disturb_persists_the_flag(monkeypatch):
    """mark_do_not_disturb must write the flag via an MCP tool.

    Today it only cancels pending nudges; no MCP tool anywhere writes
    sessions.do_not_disturb, so nothing stops future scheduling.
    """
    fake = FakeMCP([_silent_whatsapp_session()])
    monkeypatch.setattr(ns, "_call_mcp_tool", fake.call)

    async def run():
        result = ns.mark_do_not_disturb(SESSION_ID)
        if asyncio.iscoroutine(result):
            await result
        await asyncio.sleep(0.1)  # let any scheduled task run too

    asyncio.run(run())
    dnd_writes = [t for t in fake.tools_called()
                  if "do_not_disturb" in t or "dnd" in t]
    assert dnd_writes, (
        f"no MCP call persists do_not_disturb; calls were {fake.tools_called()}"
    )


# ─── Bug 2: reply cancellation in the webhook is never awaited ──────────────

def test_bug2_webhook_cancel_call_is_awaited():
    """Every cancel_nudges_for_session call in serve-orchestrator/main.py
    must be awaited (or otherwise scheduled).

    Today the WhatsApp handler calls it bare inside an async handler:
    the coroutine object is created and discarded, Python logs
    "coroutine was never awaited", and pending nudges are NOT cancelled
    when a volunteer replies.
    """
    main_py = REPO_ROOT / "serve-orchestrator" / "main.py"
    tree = ast.parse(main_py.read_text())
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "cancel_nudges_for_session"]
    assert calls, "expected at least one cancel_nudges_for_session call in main.py"
    unawaited = [c for c in calls
                 if not isinstance(parents.get(c), ast.Await)]
    assert not unawaited, (
        f"{len(unawaited)} cancel_nudges_for_session call(s) in main.py "
        "are not awaited — the coroutine never runs"
    )


def test_bug2_control_awaited_cancel_actually_cancels(monkeypatch):
    """Control: the cancel function itself works when awaited.

    This passes today, isolating Bug 2 to the call site in main.py.
    """
    fake = FakeMCP([_silent_whatsapp_session()])
    fake.nudges.append({
        "id": "nudge-x", "session_id": SESSION_ID, "volunteer_phone": PHONE,
        "volunteer_name": "", "nudge_number": 1,
        "scheduled_at": datetime.utcnow().isoformat(),
        "sent_at": None, "cancelled": False,
    })
    monkeypatch.setattr(ns, "_call_mcp_tool", fake.call)

    asyncio.run(ns.cancel_nudges_for_session(SESSION_ID))
    assert "nudge_cancel_for_session" in fake.tools_called()
    assert fake.nudges[0]["cancelled"] is True


# ─── Bug 3: frequency cap leaks and nothing auto-pauses ─────────────────────

def test_bug3a_no_new_cycle_after_third_nudge(monkeypatch):
    """After nudges #1-#3 have all been sent, a still-silent session must
    not be scheduled a fresh nudge #1.

    Today check_and_schedule_new_nudges re-schedules #1 for any active
    silent session; the real nudge_schedule tool only dedups UNSENT rows,
    so the 1h/8h/24h cycle restarts forever. The docstring's promise —
    "After 3 nudges with no reply, session is auto-paused" — is what
    should stop this (see Bug 3b).
    """
    fake = FakeMCP([_silent_whatsapp_session()])
    _wire(monkeypatch, fake)

    async def run_full_cycle():
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

    asyncio.run(run_full_cycle())
    leftover = fake.unsent_nudges()
    assert leftover == [], (
        f"a new nudge cycle was scheduled after nudge #3: {leftover}"
    )


def test_bug3b_session_paused_after_third_nudge(monkeypatch):
    """Sending nudge #3 must pause the session (scheduler docstring:
    "After 3 nudges with no reply, session is auto-paused").

    Today process_due_nudges only marks the nudge sent; the session stays
    active, which is also what enables the Bug 3a re-scheduling loop.
    """
    fake = FakeMCP([_silent_whatsapp_session()])
    _wire(monkeypatch, fake)
    fake.nudges.append({
        "id": "nudge-3", "session_id": SESSION_ID, "volunteer_phone": PHONE,
        "volunteer_name": "", "nudge_number": 3,
        "scheduled_at": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
        "sent_at": None, "cancelled": False,
    })

    sent = asyncio.run(ns.process_due_nudges())
    assert sent == 1  # setup sanity: the due nudge was sent
    assert fake.sessions[SESSION_ID]["status"] == "paused", (
        "session still active after final nudge; "
        f"tools called: {fake.tools_called()}"
    )
