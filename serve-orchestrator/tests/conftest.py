"""Shared fixtures and fakes for serve-orchestrator tests.

Mirrors the repo's existing test conventions (see
serve-delivery-agent-service/tests): MCP calls are faked in memory,
no database, network, or LLM is touched.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure the service root is on sys.path so `app.*` and `main` imports work
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.service import nudge_scheduler as ns  # noqa: E402

SESSION_ID = "11111111-1111-1111-1111-111111111111"
PHONE = "919999999999"


def make_silent_session(**overrides):
    """An active WhatsApp session silent for 25h (past every nudge delay)."""
    session = {
        "id": SESSION_ID,
        "channel": "whatsapp",
        "status": "active",
        "workflow": "new_volunteer_onboarding",
        "active_agent": "onboarding",
        "stage": "eligibility_screening",
        "persona": "new_volunteer",
        "last_message_at": (datetime.utcnow() - timedelta(hours=25)).isoformat(),
        "channel_metadata": {"volunteer_phone": PHONE},
    }
    session.update(overrides)
    return session


class FakeMCP:
    """In-memory stand-in for the MCP tools the nudge scheduler calls.

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
        # Mirrors SessionService.advance_state: paused stage -> paused status.
        s = self.sessions.get(session_id)
        if s:
            s["stage"] = new_state
            if new_state == "paused":
                s["status"] = "paused"
        return {"status": "success", "current_state": new_state}

    def tool_nudge_set_do_not_disturb(self, session_id):
        s = self.sessions.get(session_id)
        if s:
            s["do_not_disturb"] = True
        return {"status": "success", "updated": 1 if s else 0}

    # -- nudge queue tools ----------------------------------------------------
    def tool_nudge_schedule(self, session_id, volunteer_phone, nudge_number,
                            scheduled_at, volunteer_name="", episode_start=None):
        for n in self.nudges:
            if (n["session_id"] == session_id
                    and n["nudge_number"] == nudge_number
                    and n["sent_at"] is None and not n["cancelled"]):
                return {"status": "already_scheduled", "id": n["id"]}
        # Episode dedup (mirrors serve-mcp-server): no second nudge #1
        # for the same silence episode.
        if nudge_number == 1 and episode_start:
            for n in self.nudges:
                if (n["session_id"] == session_id and not n["cancelled"]
                        and n["scheduled_at"] >= episode_start):
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


def wire_scheduler(monkeypatch, fake):
    """Point the nudge scheduler at FakeMCP and a always-succeed WhatsApp send."""
    monkeypatch.setattr(ns, "_call_mcp_tool", fake.call)
    monkeypatch.setattr(ns, "_is_quiet_hour", lambda: False)

    async def fake_send(to, text):
        return True

    monkeypatch.setattr(ns, "_send_whatsapp", fake_send)
    return fake
