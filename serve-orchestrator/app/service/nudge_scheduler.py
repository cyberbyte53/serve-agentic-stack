"""
SERVE Orchestrator — Nudge Scheduler (DB-backed)

Background task that sends reminder messages to inactive WhatsApp sessions.

Schedule:
  Nudge #1: 1 hour after last message
  Nudge #2: 8 hours after last message
  Nudge #3: 24 hours after last message (final)

After 3 nudges with no reply, session is auto-paused.
Volunteer can reply "stop" to opt out permanently.

Runs every NUDGE_CHECK_INTERVAL_MINUTES (default 5 min).
Only sends to WhatsApp channel (web UI can't push).
Respects quiet hours (default 9pm-8am IST).

State is persisted in the nudge_queue DB table via MCP tools,
so nudge state survives container restarts.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from app.service.nudge_messages import build_nudge_message

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
NUDGE_ENABLED = os.environ.get("NUDGE_ENABLED", "true").lower() == "true"
NUDGE_DELAY_1_MINUTES = int(os.environ.get("NUDGE_DELAY_1_MINUTES", "60"))
NUDGE_DELAY_2_MINUTES = int(os.environ.get("NUDGE_DELAY_2_MINUTES", "480"))
# 23h45m, not 24h: WhatsApp only allows free-form messages within 24h of
# the volunteer's last message, so the final nudge must land inside that
# window (with margin for the check interval) or Meta rejects it.
NUDGE_DELAY_3_MINUTES = int(os.environ.get("NUDGE_DELAY_3_MINUTES", "1425"))
NUDGE_QUIET_HOURS_START = int(os.environ.get("NUDGE_QUIET_HOURS_START", "21"))
NUDGE_QUIET_HOURS_END = int(os.environ.get("NUDGE_QUIET_HOURS_END", "8"))
NUDGE_CHECK_INTERVAL_MINUTES = int(os.environ.get("NUDGE_CHECK_INTERVAL_MINUTES", "5"))
# Dry-run: compose and log messages without calling the WhatsApp API
# (lets the whole flow be exercised without Meta credentials).
NUDGE_DRY_RUN = os.environ.get("NUDGE_DRY_RUN", "false").lower() == "true"

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://serve-mcp-server:8004")

# WhatsApp config
_WA_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")
_WA_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
_WA_GRAPH_URL = "https://graph.facebook.com/v18.0"

# ── Nudge delays (minutes from last message) ─────────────────────────────────
# Message copy lives in nudge_messages.py (GENERIC_MESSAGES is the fallback).
NUDGE_DELAYS = {
    1: NUDGE_DELAY_1_MINUTES,
    2: NUDGE_DELAY_2_MINUTES,
    3: NUDGE_DELAY_3_MINUTES,
}


# ── MCP Tool Client ──────────────────────────────────────────────────────────

async def _call_mcp_tool(tool_name: str, arguments: Dict) -> Dict:
    """Call an MCP tool via SSE."""
    try:
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client

        wire_args = {"params": arguments} if arguments else {}
        sse_url = f"{MCP_SERVER_URL}/sse"
        async with sse_client(url=sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=wire_args)
                for item in result.content:
                    if hasattr(item, "text"):
                        try:
                            return json.loads(item.text)
                        except (json.JSONDecodeError, ValueError):
                            return {"result": item.text}
        return {}
    except Exception as e:
        logger.error(f"[nudge] MCP tool {tool_name} failed: {e}")
        return {"status": "error", "error": str(e)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_quiet_hour() -> bool:
    """Check if current time is within quiet hours (IST)."""
    now = datetime.utcnow()
    ist_hour = (now.hour + 5) % 24  # rough IST
    if NUDGE_QUIET_HOURS_START > NUDGE_QUIET_HOURS_END:
        return ist_hour >= NUDGE_QUIET_HOURS_START or ist_hour < NUDGE_QUIET_HOURS_END
    return NUDGE_QUIET_HOURS_START <= ist_hour < NUDGE_QUIET_HOURS_END


async def _send_whatsapp(to: str, text: str) -> bool:
    """Send a WhatsApp text message."""
    if not _WA_TOKEN or not _WA_PHONE_NUMBER_ID:
        return False
    url = f"{_WA_GRAPH_URL}/{_WA_PHONE_NUMBER_ID}/messages"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {_WA_TOKEN}", "Content-Type": "application/json"},
                json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}},
            )
            r.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"[nudge] WhatsApp send failed to {to[:6]}***: {e}")
        return False


# ── Public API (called from main.py when volunteer messages) ──────────────────

async def cancel_nudges_for_session(session_id: str) -> None:
    """Cancel all pending nudges for a session (volunteer replied)."""
    await _call_mcp_tool("nudge_cancel_for_session", {"session_id": session_id})


async def mark_do_not_disturb(session_id: str) -> None:
    """Persist the opt-out on the session and cancel any pending nudges."""
    await _call_mcp_tool("nudge_set_do_not_disturb", {"session_id": session_id})
    await cancel_nudges_for_session(session_id)
    logger.info(f"[nudge] Session {session_id[:8]}... marked as DO_NOT_DISTURB")


def is_do_not_disturb(session_id: str) -> bool:
    """Check DND — this is now checked from session.do_not_disturb flag by the orchestrator."""
    # The orchestrator checks this via session state; this function is a no-op placeholder.
    return False


def get_nudge_count(session_id: str) -> int:
    """Not used in DB-backed mode — tracked by nudge_number in DB."""
    return 0


# ── Core Loop Logic ───────────────────────────────────────────────────────────

async def check_and_schedule_new_nudges() -> None:
    """
    Check active WhatsApp sessions for silence and schedule nudge #1.
    Deduplication is handled by the MCP tool (won't create duplicates).
    """
    # Get all active sessions
    result = await _call_mcp_tool("list_sessions", {"status": "active", "limit": 200})
    sessions = result.get("sessions", [])
    now = datetime.utcnow()

    for session in sessions:
        session_id = session.get("id", "")
        channel = session.get("channel", "")
        status = session.get("status", "")
        last_msg = session.get("last_message_at")

        # Only WhatsApp active sessions
        if channel != "whatsapp" or status != "active":
            continue

        # Check do_not_disturb flag on session
        if session.get("do_not_disturb"):
            continue

        if not last_msg:
            continue

        try:
            last_msg_time = datetime.fromisoformat(last_msg.replace("Z", "").replace("+00:00", ""))
        except (ValueError, TypeError):
            continue

        silence_minutes = (now - last_msg_time).total_seconds() / 60

        # Don't nudge sessions silent for more than 72 hours — they're abandoned
        if silence_minutes > 72 * 60:
            continue

        if silence_minutes >= NUDGE_DELAY_1_MINUTES:
            # Get phone from channel_metadata
            ch_meta = session.get("channel_metadata") or {}
            phone = (
                ch_meta.get("volunteer_phone")
                or ch_meta.get("phone_number")
                or session.get("actor_id", "")
            )
            if not phone or phone.startswith("dev-") or phone.startswith("guest_"):
                continue

            # Schedule nudge #1 — MCP tool handles dedup
            send_at = last_msg_time + timedelta(minutes=NUDGE_DELAY_1_MINUTES)
            await _call_mcp_tool("nudge_schedule", {
                "session_id": session_id,
                "volunteer_phone": phone,
                "nudge_number": 1,
                "scheduled_at": send_at.isoformat(),
                "volunteer_name": ch_meta.get("volunteer_name", ""),
                "episode_start": last_msg,
            })


async def process_due_nudges() -> int:
    """Send all due nudges. Returns count sent."""
    if _is_quiet_hour():
        return 0

    result = await _call_mcp_tool("nudge_get_due", {"now": datetime.utcnow().isoformat()})
    nudges = result.get("nudges", [])
    sent_count = 0

    for nudge in nudges:
        nudge_id = nudge["id"]
        session_id = nudge["session_id"]
        phone = nudge["volunteer_phone"]
        nudge_num = nudge["nudge_number"]

        # Compose from the volunteer's CURRENT session context so the
        # message references where they actually left off (issue #25).
        context: Dict = {}
        session_result = await _call_mcp_tool("get_session", {"session_id": session_id})
        if isinstance(session_result, dict):
            context = session_result.get("session") or {}
        if not context.get("volunteer_name") and nudge.get("volunteer_name"):
            context = {**context, "volunteer_name": nudge["volunteer_name"]}
        message = build_nudge_message(context, nudge_num)

        if NUDGE_DRY_RUN:
            logger.info(
                f"[nudge][dry-run] Would send nudge #{nudge_num} to {phone[:6]}*** "
                f"(session {session_id[:8]}...): {message!r}"
            )
            success = True
        else:
            success = await _send_whatsapp(phone, message)

        if success:
            # Mark as sent in DB
            await _call_mcp_tool("nudge_mark_sent", {"nudge_id": nudge_id})
            sent_count += 1
            logger.info(f"[nudge] Sent nudge #{nudge_num} to {phone[:6]}*** (session {session_id[:8]}...)")

            # Schedule next nudge if not the last
            if nudge_num < 3:
                next_num = nudge_num + 1
                # Calculate from original last_message_at (nudge scheduled_at was based on it)
                scheduled_base = datetime.fromisoformat(nudge["scheduled_at"])
                next_delay_from_base = NUDGE_DELAYS[next_num] - NUDGE_DELAYS[nudge_num]
                next_at = scheduled_base + timedelta(minutes=next_delay_from_base)

                await _call_mcp_tool("nudge_schedule", {
                    "session_id": session_id,
                    "volunteer_phone": phone,
                    "nudge_number": next_num,
                    "scheduled_at": next_at.isoformat(),
                    "volunteer_name": nudge.get("volunteer_name", ""),
                })
            else:
                # Final nudge sent with no reply — pause the session so it
                # drops out of the active scan and no new cycle is scheduled.
                await _call_mcp_tool("advance_session_state", {
                    "session_id": session_id,
                    "new_state": "paused",
                })
                logger.info(f"[nudge] Session {session_id[:8]}... auto-paused after final nudge")
        else:
            logger.warning(f"[nudge] Failed to send nudge #{nudge_num} to {phone[:6]}***")

    return sent_count


# ── Background Loop ───────────────────────────────────────────────────────────

async def start_nudge_scheduler() -> None:
    """
    Background loop — runs every NUDGE_CHECK_INTERVAL_MINUTES.
    Checks for sessions needing nudges and sends due nudges.
    """
    if not NUDGE_ENABLED:
        logger.info("[nudge] Nudge scheduler DISABLED (NUDGE_ENABLED=false)")
        return

    if not NUDGE_DRY_RUN and (not _WA_TOKEN or not _WA_PHONE_NUMBER_ID):
        logger.warning("[nudge] Nudge scheduler disabled — WhatsApp not configured")
        return

    logger.info(
        f"[nudge] Nudge scheduler starting (DB-backed) — "
        f"check every {NUDGE_CHECK_INTERVAL_MINUTES}min, "
        f"delays: {NUDGE_DELAY_1_MINUTES}m/{NUDGE_DELAY_2_MINUTES}m/{NUDGE_DELAY_3_MINUTES}m, "
        f"quiet hours: {NUDGE_QUIET_HOURS_START}:00-{NUDGE_QUIET_HOURS_END}:00 IST"
    )

    while True:
        try:
            await check_and_schedule_new_nudges()
            sent = await process_due_nudges()
            if sent:
                logger.info(f"[nudge] Cycle complete: {sent} nudges sent")
        except Exception as e:
            logger.error(f"[nudge] Scheduler error: {e}")

        await asyncio.sleep(NUDGE_CHECK_INTERVAL_MINUTES * 60)
