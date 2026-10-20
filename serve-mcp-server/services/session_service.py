"""
SERVE MCP Server - Session Service
Manages the AI conversation session lifecycle.

All reads/writes go to PostgreSQL (MCP DB).
External Serve Registry calls happen at session boundaries only
(start = identity resolution + profile prefetch; end = write-back).
Falls back to an in-memory store when Postgres is unavailable.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import select, update

from services.database import (
    AgentHandoffLog, ConversationMessage, Session as DBSession,
    TelemetryEvent, get_db, is_db_healthy,
)

logger = logging.getLogger(__name__)


# ─── In-memory fallback ───────────────────────────────────────────────────────

class _InMemoryStore:
    def __init__(self):
        self.sessions:  Dict[str, Dict] = {}
        self.messages:  Dict[str, List] = {}
        self.telemetry: Dict[str, List] = {}

_mem = _InMemoryStore()


# ─── Service ──────────────────────────────────────────────────────────────────

class SessionService:

    # ── Create ────────────────────────────────────────────────────────────────

    async def create_session(
        self,
        channel: str = "web_ui",
        persona: str = "new_volunteer",
        channel_metadata: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        identity_type: Optional[str] = None,
        user_type: Optional[str] = None,
        volunteer_id: Optional[str] = None,
        coordinator_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new session, enriched with identity and Serve Registry references."""
        session_id = str(uuid4())
        now = datetime.utcnow()
        meta = channel_metadata or {}

        # Fall back to extracting actor_id from channel_metadata if not explicit
        if not actor_id:
            actor_id = meta.get("actor_id") or meta.get("keycloak_sub")

        # Derive workflow + agent from persona
        if persona == "need_coordinator":
            workflow     = "need_coordination"
            active_agent = "need"
            stage        = "initiated"
        elif persona == "returning_volunteer":
            workflow     = "returning_volunteer"
            active_agent = "engagement"
            stage        = "re_engaging"
        elif persona == "recommended_volunteer":
            workflow     = "recommended_volunteer"
            active_agent = "engagement"
            stage        = "verifying_identity"
        else:
            workflow     = "new_volunteer_onboarding"
            active_agent = "onboarding"
            stage        = "welcome"

        if is_db_healthy():
            try:
                async with get_db() as db:
                    session = DBSession(
                        id=UUID(session_id),
                        actor_id=actor_id,
                        identity_type=identity_type,
                        channel=channel,
                        persona=persona,
                        user_type=user_type,
                        volunteer_id=volunteer_id,
                        coordinator_id=coordinator_id,
                        workflow=workflow,
                        active_agent=active_agent,
                        status="active",
                        stage=stage,
                        channel_metadata=meta or None,
                        idempotency_key=idempotency_key,
                        registry_checked_at=now if (volunteer_id or coordinator_id) else None,
                        created_at=now,
                        updated_at=now,
                        last_message_at=now,
                    )
                    db.add(session)
                logger.info(f"Session created in DB: {session_id} (persona={persona}, user_type={user_type})")
                return {
                    "status": "success", "session_id": session_id,
                    "stage": stage, "workflow": workflow, "storage": "postgres",
                }
            except Exception as e:
                logger.warning(f"DB create_session failed, using memory: {e}")

        # In-memory fallback
        _mem.sessions[session_id] = {
            "id": session_id, "actor_id": actor_id, "identity_type": identity_type,
            "channel": channel, "persona": persona, "user_type": user_type,
            "volunteer_id": volunteer_id, "coordinator_id": coordinator_id,
            "workflow": workflow, "active_agent": active_agent,
            "status": "active", "stage": stage, "sub_state": None,
            "context_summary": None, "channel_metadata": meta,
            "idempotency_key": idempotency_key,
            "created_at": now.isoformat(), "updated_at": now.isoformat(),
        }
        _mem.messages[session_id]  = []
        _mem.telemetry[session_id] = []
        return {
            "status": "success", "session_id": session_id,
            "stage": stage, "workflow": workflow, "storage": "memory",
        }

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        if is_db_healthy():
            try:
                async with get_db() as db:
                    result = await db.execute(
                        select(DBSession).where(DBSession.id == UUID(session_id))
                    )
                    row = result.scalar_one_or_none()
                    if row:
                        return {"status": "success", "session": self._row_to_dict(row)}
            except Exception as e:
                logger.warning(f"DB get_session failed: {e}")

        row = _mem.sessions.get(session_id)
        if not row:
            return {"status": "error", "error_code": "SESSION_NOT_FOUND",
                    "error_message": f"Session {session_id} does not exist",
                    "recoverable": True,
                    "suggested_action": "Call start_session to create a new session"}
        return {"status": "success", "session": row}

    async def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        if is_db_healthy():
            try:
                async with get_db() as db:
                    query = (
                        select(DBSession)
                        .order_by(DBSession.created_at.desc())
                        .limit(limit)
                    )
                    if status:
                        query = query.where(DBSession.status == status)
                    result = await db.execute(query)
                    rows = result.scalars().all()
                    return {"status": "success",
                            "sessions": [self._row_to_dict(r) for r in rows]}
            except Exception as e:
                logger.warning(f"DB list_sessions failed: {e}")

        sessions = list(_mem.sessions.values())
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        return {"status": "success", "sessions": sessions[:limit]}

    async def resume_context(self, session_id: str) -> Dict[str, Any]:
        """Return session + recent messages for context rebuild."""
        session_result = await self.get_session(session_id)
        if session_result.get("status") != "success":
            return session_result

        messages = await self._get_recent_messages(session_id, limit=10)

        return {
            "status": "success",
            "session": session_result["session"],
            "conversation_history": messages,
            "memory_summary": None,   # populated by MemoryService if needed
        }

    async def find_active_by_actor(self, actor_id: str) -> Dict[str, Any]:
        """
        Find the most recent active or paused session for a given actor_id.
        Returns the session data or {"status": "not_found"}.
        """
        if is_db_healthy():
            try:
                async with get_db() as db:
                    query = (
                        select(DBSession)
                        .where(
                            DBSession.actor_id == actor_id,
                            DBSession.status.in_(["active", "paused"]),
                        )
                        .order_by(DBSession.updated_at.desc())
                        .limit(1)
                    )
                    result = await db.execute(query)
                    row = result.scalars().first()
                    if row:
                        return {
                            "status": "success",
                            "session": self._row_to_dict(row),
                        }
                    return {"status": "not_found"}
            except Exception as e:
                logger.warning(f"DB find_active_by_actor failed: {e}")

        # In-memory fallback
        matches = [
            s for s in _mem.sessions.values()
            if s.get("actor_id") == actor_id and s.get("status") in ("active", "paused")
        ]
        if matches:
            # Sort by updated_at descending
            matches.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
            return {"status": "success", "session": matches[0]}
        return {"status": "not_found"}

    # ── State transitions ─────────────────────────────────────────────────────

    async def advance_state(
        self,
        session_id: str,
        new_state: str,
        sub_state: Optional[str] = None,
        active_agent: Optional[str] = None,
        workflow: Optional[str] = None,
    ) -> Dict[str, Any]:
        _TERMINAL = {
            "need_submitted", "fulfillment_handoff_ready", "approved", "complete",
        }
        if new_state in _TERMINAL:
            new_status = "completed"
        elif new_state == "paused":
            new_status = "paused"
        else:
            new_status = None

        if is_db_healthy():
            try:
                async with get_db() as db:
                    result = await db.execute(
                        select(DBSession).where(DBSession.id == UUID(session_id))
                    )
                    row = result.scalar_one_or_none()
                    if row:
                        old_state = row.stage
                        values: Dict[str, Any] = {
                            "stage":      new_state,
                            "sub_state":  sub_state,
                            "updated_at": datetime.utcnow(),
                        }
                        if new_status:
                            values["status"] = new_status
                        if active_agent:
                            values["active_agent"] = active_agent
                        if workflow:
                            values["workflow"] = workflow
                        await db.execute(
                            update(DBSession)
                            .where(DBSession.id == UUID(session_id))
                            .values(**values)
                        )
                        return {
                            "status": "success",
                            "previous_state": old_state,
                            "current_state":  new_state,
                            "active_agent":   active_agent or row.active_agent,
                            "is_valid": True,
                        }
            except Exception as e:
                logger.warning(f"DB advance_state failed: {e}")

        # In-memory fallback
        if session_id in _mem.sessions:
            old_state = _mem.sessions[session_id].get("stage", "init")
            _mem.sessions[session_id]["stage"]      = new_state
            _mem.sessions[session_id]["sub_state"]  = sub_state
            _mem.sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
            if new_status:
                _mem.sessions[session_id]["status"] = new_status
            if active_agent:
                _mem.sessions[session_id]["active_agent"] = active_agent
            if workflow:
                _mem.sessions[session_id]["workflow"] = workflow
            return {"status": "success", "previous_state": old_state,
                    "current_state": new_state, "active_agent": active_agent, "is_valid": True}

        return {"status": "error", "error_code": "SESSION_NOT_FOUND",
                "error_message": f"Session {session_id} not found"}

    async def update_session_context(
        self,
        session_id: str,
        *,
        sub_state: Optional[str] = None,
        context_summary: Optional[str] = None,
        status: Optional[str] = None,
        active_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch non-stage session fields without advancing the workflow state."""
        values: Dict[str, Any] = {"updated_at": datetime.utcnow()}
        if sub_state is not None:
            values["sub_state"] = sub_state
        if context_summary is not None:
            values["context_summary"] = context_summary
        if status is not None:
            values["status"] = status
        if active_agent is not None:
            values["active_agent"] = active_agent

        if len(values) == 1:
            return {"status": "success", "updated_fields": []}

        if is_db_healthy():
            try:
                async with get_db() as db:
                    result = await db.execute(
                        select(DBSession).where(DBSession.id == UUID(session_id))
                    )
                    row = result.scalar_one_or_none()
                    if row:
                        await db.execute(
                            update(DBSession)
                            .where(DBSession.id == UUID(session_id))
                            .values(**values)
                        )
                        return {
                            "status": "success",
                            "updated_fields": [k for k in values if k != "updated_at"],
                        }
            except Exception as e:
                logger.warning(f"DB update_session_context failed: {e}")

        if session_id in _mem.sessions:
            if sub_state is not None:
                _mem.sessions[session_id]["sub_state"] = sub_state
            if context_summary is not None:
                _mem.sessions[session_id]["context_summary"] = context_summary
            if status is not None:
                _mem.sessions[session_id]["status"] = status
            if active_agent is not None:
                _mem.sessions[session_id]["active_agent"] = active_agent
            _mem.sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
            return {
                "status": "success",
                "updated_fields": [k for k in values if k != "updated_at"],
            }

        return {
            "status": "error",
            "error_code": "SESSION_NOT_FOUND",
            "error_message": f"Session {session_id} not found",
        }

    async def update_actor_id(self, session_id: str, new_actor_id: str) -> Dict[str, Any]:
        """
        Update the actor_id of a session.
        Used to link guest sessions to authenticated users after sign-up.
        """
        if is_db_healthy():
            try:
                async with get_db() as db:
                    result = await db.execute(
                        select(DBSession).where(DBSession.id == UUID(session_id))
                    )
                    row = result.scalar_one_or_none()
                    if row:
                        await db.execute(
                            update(DBSession)
                            .where(DBSession.id == UUID(session_id))
                            .values(actor_id=new_actor_id, updated_at=datetime.utcnow())
                        )
                        return {"status": "success", "actor_id": new_actor_id}
            except Exception as e:
                logger.warning(f"DB update_actor_id failed: {e}")

        # In-memory fallback
        if session_id in _mem.sessions:
            _mem.sessions[session_id]["actor_id"] = new_actor_id
            _mem.sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
            return {"status": "success", "actor_id": new_actor_id}

        return {
            "status": "error",
            "error_code": "SESSION_NOT_FOUND",
            "error_message": f"Session {session_id} not found",
        }

    # ── Messages ──────────────────────────────────────────────────────────────

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent: Optional[str] = None,
        message_metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        message_id = str(uuid4())
        if is_db_healthy():
            try:
                async with get_db() as db:
                    db.add(ConversationMessage(
                        id=UUID(message_id),
                        session_id=UUID(session_id),
                        role=role,
                        content=content,
                        agent=agent,
                        message_metadata=message_metadata,
                        created_at=datetime.utcnow(),
                    ))
                    # Update last_message_at on session
                    await db.execute(
                        update(DBSession)
                        .where(DBSession.id == UUID(session_id))
                        .values(last_message_at=datetime.utcnow())
                    )
                return {"status": "success", "message_id": message_id}
            except Exception as e:
                logger.warning(f"DB save_message failed: {e}")

        if session_id not in _mem.messages:
            _mem.messages[session_id] = []
        _mem.messages[session_id].append({
            "id": message_id, "role": role, "content": content,
            "agent": agent, "timestamp": datetime.utcnow().isoformat(),
        })
        return {"status": "success", "message_id": message_id}

    async def get_conversation(
        self, session_id: str, limit: int = 50
    ) -> Dict[str, Any]:
        messages = await self._get_recent_messages(session_id, limit=limit)
        return {"status": "success", "messages": messages}

    # ── Telemetry ─────────────────────────────────────────────────────────────

    async def log_event(
        self,
        session_id: str,
        event_type: str,
        agent: Optional[str] = None,
        data: Optional[Dict] = None,
        domain: Optional[str] = None,
        source_service: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        event_id = str(uuid4())
        if is_db_healthy():
            try:
                async with get_db() as db:
                    db.add(TelemetryEvent(
                        id=UUID(event_id),
                        session_id=UUID(session_id),
                        event_type=event_type,
                        agent=agent,
                        domain=domain,
                        source_service=source_service,
                        duration_ms=duration_ms,
                        data=data or {},
                        timestamp=datetime.utcnow(),
                    ))
                return {"status": "success", "event_id": event_id}
            except Exception as e:
                logger.warning(f"DB log_event failed: {e}")

        if session_id not in _mem.telemetry:
            _mem.telemetry[session_id] = []
        _mem.telemetry[session_id].append({
            "id": event_id, "event_type": event_type, "agent": agent,
            "data": data or {}, "timestamp": datetime.utcnow().isoformat(),
        })
        return {"status": "success", "event_id": event_id}

    # ── Handoff ───────────────────────────────────────────────────────────────

    async def emit_handoff_event(
        self,
        session_id: str,
        from_agent: str,
        to_agent: str,
        handoff_type: str,
        payload: Optional[Dict] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        event_id = str(uuid4())
        if is_db_healthy():
            try:
                async with get_db() as db:
                    db.add(AgentHandoffLog(
                        id=UUID(event_id),
                        session_id=UUID(session_id),
                        from_agent=from_agent,
                        to_agent=to_agent,
                        handoff_type=handoff_type,
                        payload=payload or {},
                        reason=reason,
                        created_at=datetime.utcnow(),
                    ))
                return {"status": "success", "event_id": event_id}
            except Exception as e:
                logger.warning(f"DB emit_handoff_event failed, falling back to telemetry: {e}")

        # Fallback: log as telemetry event
        return await self.log_event(
            session_id=session_id,
            event_type="handoff",
            agent=from_agent,
            data={"from": from_agent, "to": to_agent, "type": handoff_type,
                  "payload": payload or {}, "reason": reason},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_recent_messages(
        self, session_id: str, limit: int = 10
    ) -> List[Dict]:
        if is_db_healthy():
            try:
                async with get_db() as db:
                    result = await db.execute(
                        select(ConversationMessage)
                        .where(ConversationMessage.session_id == UUID(session_id))
                        .order_by(ConversationMessage.created_at.desc())
                        .limit(limit)
                    )
                    rows = result.scalars().all()
                    return [
                        {
                            "id":      str(m.id),
                            "role":    m.role,
                            "content": m.content,
                            "agent":   m.agent,
                            "timestamp": m.created_at.isoformat() if m.created_at else None,
                        }
                        for m in reversed(rows)
                    ]
            except Exception as e:
                logger.warning(f"DB get_messages failed: {e}")

        return _mem.messages.get(session_id, [])[-limit:]

    def _row_to_dict(self, row: DBSession) -> Dict:
        return {
            "id":              str(row.id),
            "actor_id":        row.actor_id,
            "identity_type":   row.identity_type,
            "channel":         row.channel,
            "persona":         row.persona,
            "user_type":       row.user_type,
            "volunteer_id":    row.volunteer_id,
            "coordinator_id":  row.coordinator_id,
            "workflow":        row.workflow,
            "active_agent":    row.active_agent,
            "status":          row.status,
            "stage":           row.stage,
            "sub_state":       row.sub_state,
            "context_summary": row.context_summary,
            "channel_metadata": row.channel_metadata,
            "idempotency_key": row.idempotency_key,
            "created_at":      row.created_at.isoformat() if row.created_at else None,
            "updated_at":      row.updated_at.isoformat() if row.updated_at else None,
            "last_message_at": row.last_message_at.isoformat() if row.last_message_at else None,
        }
