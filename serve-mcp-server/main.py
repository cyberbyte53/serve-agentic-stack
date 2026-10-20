"""
SERVE AI - MCP Server
Model Context Protocol server exposing volunteer management and need coordination
tools to any MCP-compatible LLM client (Claude, Cursor, etc.).

External data sources:
  Serve Volunteering Service → user identity, volunteer profiles
  Serve Need Service         → entities (schools), needs

MCP DB (this server owns):
  sessions, volunteer_profiles (cache), conversation_messages,
  memory_summaries, telemetry_events, actor_registry_cache,
  identity_resolution_log, agent_handoff_log, need_drafts
"""
import logging
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Import config and services ─────────────────────────────────────────────────
from config import MCP_PORT, MCP_HOST

from services.session_service     import SessionService
from services.profile_service     import ProfileService
from services.memory_service      import MemoryService
from services.engagement_service  import engagement_service
from services.coordinator_service import coordinator_service
from services.school_service      import school_service
from services.need_service        import need_service
from services.identity_service    import identity_service
from services.serve_registry_client import volunteering_client
from services.volunteer_service   import volunteer_service

# ── Import Pydantic input schemas ─────────────────────────────────────────────
from schemas import (
    LookupActorInput,
    StartSessionInput, GetSessionInput, ResumeSessionInput, FindSessionByActorInput,
    UpdateSessionActorInput,
    AdvanceSessionStateInput, ListSessionsInput,
    NudgeScheduleInput, NudgeGetDueInput, NudgeMarkSentInput,
    NudgeCancelForSessionInput, NudgeSetDoNotDisturbInput,
    GetMissingFieldsInput, SaveVolunteerFieldsInput, EvaluateReadinessInput,
    SaveMessageInput, GetConversationInput,
    SaveMemorySummaryInput, GetMemorySummaryInput,
    EngagementSaveConfirmedSignalsInput,
    EngagementUpdateVolunteerStatusInput, EngagementPrepareFulfillmentHandoffInput,
    LogEventInput, EmitHandoffEventInput,
    ResolveCoordinatorInput, CreateCoordinatorInput, MapCoordinatorToSchoolInput,
    ResolveSchoolContextInput, CreateSchoolContextInput, FetchPreviousNeedContextInput,
    CreateOrUpdateNeedDraftInput, SubmitNeedInput, UpdateNeedStatusInput,
    PrepareHandoffInput, PauseNeedSessionInput,
    SaveNeedMessageInput, LogNeedEventInput, EmitNeedHandoffInput,
    GetSessionAnalyticsInput,
    # Engagement + Fulfillment agent tools
    GetVolunteerFulfillmentHistoryInput, CheckActiveNominationsInput,
    GetEngagementContextInput, GetEngagementContextByEmailInput,
    NominateVolunteerInput, ConfirmNominationInput,
    GetNominationsForNeedInput, GetRecommendedVolunteersInput,
    GetNeedsForEntityInput, GetNeedDetailsInput,
    # Delivery assistant tools
    DeliveryStartActivationInput, DeliveryGetContextInput,
    DeliveryConfirmAcknowledgementInput, DeliveryConfirmFirstSessionReadinessInput,
    DeliveryCompleteActivationInput, DeliveryCreateScheduledSessionInput,
    DeliveryGetScheduledSessionsInput, DeliveryGetDueRemindersInput,
    DeliveryMarkReminderInput, DeliveryRecordSessionOutcomeInput,
    DeliveryLogBlockerInput, DeliveryCaptureRescheduleInput,
    DeliveryUpdateStatusInput, DeliveryEvaluateEscalationInput,
    # Delivery assistant — full-spec expansion
    DeliveryReadAssignmentContextInput, DeliveryReadActivationContextInput,
    DeliveryReadScheduleContextInput, DeliveryReadSessionContextInput,
    DeliveryReadDeliveryHistoryInput, DeliveryExtractSignalsInput,
    DeliveryDetectBlockersInput, DeliveryGetMissingSignalsInput,
    DeliveryEvaluateActivationInput, DeliveryEvaluateReadinessInput,
    DeliverySetReadinessDimensionInput, DeliveryEvaluateDeliveryHealthInput,
    DeliveryEvaluateNextActionInput, DeliveryGetActivationContentInput,
    DeliveryNotifyStakeholderInput, DeliverySetCoordinatorPhoneInput,
    DeliverySendSessionDayReminderInput,
    DeliverySendPreSessionReminderInput, DeliverySendCompletionCheckInput,
    DeliverySendFollowupNudgeInput, DeliveryReadReminderHistoryInput,
    DeliverySuppressReminderInput, DeliveryStartCheckinInput,
    DeliveryCloseCheckinInput, DeliveryUpdateBlockerInput,
    DeliveryGetSupportGuidanceInput, DeliveryCreateOpsSupportRequestInput,
    DeliverySubmitRescheduleInput, DeliveryReadRescheduleStatusInput,
    DeliveryResolveRescheduleInput, DeliveryRaiseRiskInput,
    DeliveryPrepareOpsHandoffInput, DeliveryWriteSessionSummaryInput,
    DeliveryWriteDeliverySummaryInput,
)
from services.delivery_service import delivery_service
session_service = SessionService()
profile_service = ProfileService()
memory_service  = MemoryService()


# ── Server lifespan ────────────────────────────────────────────────────────────

_db_initialized = False

@asynccontextmanager
async def _lifespan(server: FastMCP):
    global _db_initialized
    if not _db_initialized:
        from services.database import init_db, check_db_health
        try:
            await init_db()
            await check_db_health()
            _db_initialized = True
            logger.info("MCP Server ready — DB initialised and health cached")
        except Exception as e:
            logger.warning(f"DB init failed (in-memory fallback active): {e}")
    yield


mcp = FastMCP(
    name="SERVE AI MCP Server",
    instructions=(
        "MCP server for eVidyaloka volunteer onboarding and need coordination. "
        "Use lookup_actor at the start of every session to identify the user before "
        "calling any workflow tools."
    ),
    host=MCP_HOST,
    port=MCP_PORT,
    lifespan=_lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def lookup_actor(params: LookupActorInput) -> dict:
    """
    Resolve who this actor is across all systems (Serve Registry + MCP history).
    MUST be called at session start before any workflow tool.

    Args:
        actor_id:      Channel-native identity (email, phone number, temp session id)
        channel:       whatsapp | web_ui | api | mobile | scheduler
        identity_type: email | phone | session_id | system (inferred if omitted)
        session_id:    MCP session UUID if already created (for log linking)

    Returns:
        user_type:          S1=new_user | S2=registry_known | S3=returning_ai_user
                            | S4=coordinator | S5=anonymous
        actor_type:         volunteer | coordinator | system
        serve_entity_id:    Serve Registry osid (null if brand new)
        has_prior_session:  bool
        last_session_id:    UUID of most recent MCP session (if any)
        profile_available:  True when Serve Registry has existing profile data
        suggested_workflow: new_volunteer_onboarding | returning_volunteer
                            | need_coordination | system_triggered
    """
    result = await identity_service.resolve(
        actor_id=params.actor_id,
        channel=params.channel,
        identity_type=params.identity_type,
        session_id=params.session_id,
    )
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def start_session(params: StartSessionInput) -> dict:
    """
    Start a new conversation session.

    Args:
        channel:          web_ui | whatsapp | api | scheduler | mobile
        persona:          new_volunteer | returning_volunteer | need_coordinator
        channel_metadata: Channel-specific context {actor_id, trigger_type, …}
        actor_id:         Channel-native identity (from lookup_actor result)
        identity_type:    email | phone | session_id | system
        user_type:        From lookup_actor: new_user | registry_known | returning_ai_user
                          | coordinator | anonymous
        volunteer_id:     Serve Registry osid (from lookup_actor serve_entity_id)
        coordinator_id:   Serve Registry coordinator osid (for need_coordinator persona)
        idempotency_key:  Deduplication key (WhatsApp message_id / wamid)

    Returns:
        session_id, stage, workflow
    """
    result = await session_service.create_session(
        channel=params.channel,
        persona=params.persona,
        channel_metadata=params.channel_metadata,
        actor_id=params.actor_id,
        identity_type=params.identity_type,
        user_type=params.user_type,
        volunteer_id=params.volunteer_id,
        coordinator_id=params.coordinator_id,
        idempotency_key=params.idempotency_key,
    )

    # For S2/S3 users with existing Serve Registry profile → pre-populate profile cache
    session_id = result.get("session_id")
    if session_id and params.volunteer_id and params.user_type in ("registry_known", "returning_ai_user"):
        await profile_service.prefetch_from_registry(
            session_id=session_id,
            volunteer_id=params.volunteer_id,
        )
        result["profile_prefetched"] = True

    logger.info(f"start_session: {session_id} (persona={params.persona}, user_type={params.user_type})")
    return result


@mcp.tool()
async def get_session(params: GetSessionInput) -> dict:
    """
    Retrieve current state of a session.

    Args:
        session_id: UUID of the session

    Returns:
        Full session object including user_type, volunteer_id, stage, status
    """
    return await session_service.get_session(params.session_id)


@mcp.tool()
async def resume_session(params: ResumeSessionInput) -> dict:
    """
    Resume an existing session with full context (session state + conversation history).

    Args:
        session_id: UUID of the session to resume

    Returns:
        session, conversation_history (last 10 messages), memory_summary
    """
    result = await session_service.resume_context(params.session_id)

    # Enrich with memory summary
    memory = await memory_service.get_summary(params.session_id)
    result["memory_summary"] = memory.get("data")

    logger.info(f"resume_session: {params.session_id[:8]}…")
    return result


@mcp.tool()
async def find_session_by_actor(params: FindSessionByActorInput) -> dict:
    """
    Find the most recent active or paused session for a given actor_id.
    Used to resume sessions after page refresh when the user re-authenticates.

    Args:
        actor_id: Keycloak sub UUID or other stable identity

    Returns:
        Session data if found, or {"status": "not_found"}
    """
    return await session_service.find_active_by_actor(params.actor_id)


@mcp.tool()
async def update_session_actor(params: UpdateSessionActorInput) -> dict:
    """
    Update the actor_id of an existing session.
    Used to link a guest (anonymous) session to an authenticated Keycloak user
    after they sign up.

    Args:
        session_id: UUID of the session to update
        actor_id:   New actor_id (Keycloak sub) to link to this session

    Returns:
        {"status": "success"} or {"status": "error", "error_message": "..."}
    """
    return await session_service.update_actor_id(params.session_id, params.actor_id)


@mcp.tool()
async def advance_session_state(params: AdvanceSessionStateInput) -> dict:
    """
    Advance a session to a new workflow stage.

    Args:
        session_id:   UUID of the session
        new_state:    Target stage (e.g. intent_discovery, onboarding_complete)
        sub_state:    Optional finer-grained sub-state
        active_agent: Updated owning agent after a handoff

    Returns:
        previous_state, current_state, active_agent, is_valid

    Side-effect:
        When new_state = "onboarding_complete", the volunteer profile is
        automatically synced back to Serve Registry.
    """
    result = await session_service.advance_state(
        session_id=params.session_id,
        new_state=params.new_state,
        sub_state=params.sub_state,
        active_agent=params.active_agent,
        workflow=params.workflow,
    )

    # Write-back to Serve Registry on onboarding completion
    if params.new_state == "onboarding_complete":
        logger.info(f"[{params.session_id}] ── REGISTRATION START ── onboarding_complete triggered")
        from services.database import get_db, Session as DBSession, is_db_healthy
        from sqlalchemy import update as sa_update
        from uuid import UUID

        session_result = await session_service.get_session(params.session_id)
        volunteer_id   = session_result.get("session", {}).get("volunteer_id")
        logger.info(f"[{params.session_id}] existing volunteer_id on session: {volunteer_id}")

        if not volunteer_id:
            # Check if a volunteer with this email already exists in the registry
            profile_result = await profile_service.get_profile(params.session_id)
            profile        = profile_result.get("profile", {})
            email = profile.get("email")
            phone = profile.get("phone")
            full_name = profile.get("full_name", "")
            logger.info(f"[{params.session_id}] profile for registration: name={full_name}, email={email}, phone={phone}, qualification={profile.get('qualification')}")

            # ── Step 0: Create Keycloak account ────────────────────────────────
            if email:
                keycloak_user_id = await volunteering_client.create_keycloak_user(
                    email=email,
                    full_name=full_name,
                    phone=phone,
                )
                if keycloak_user_id:
                    logger.info(f"[{params.session_id}] Keycloak user created: {keycloak_user_id}")
                else:
                    logger.warning(f"[{params.session_id}] Keycloak user creation failed (continuing with Serve Registry)")

            # ── Step 1: Serve Registry — check duplicate + create ──────────────
            if email:
                existing = await volunteering_client.lookup_by_email(email)
                logger.info(f"[{params.session_id}] lookup_by_email({email}) result: {existing}")
                if existing and existing.get("osid"):
                    volunteer_id = existing["osid"]
                    logger.info(f"[{params.session_id}] Existing volunteer found by email: {volunteer_id}")

            if not volunteer_id:
                logger.info(f"[{params.session_id}] Creating new volunteer in Serve Registry...")
                new_vid = await volunteering_client.create_volunteer(
                    full_name=full_name,
                    email=email,
                    phone=profile.get("phone"),
                    city=profile.get("location"),
                )
                logger.info(f"[{params.session_id}] create_volunteer result: {new_vid}")
                if new_vid:
                    volunteer_id = new_vid

            if volunteer_id:
                if is_db_healthy():
                    try:
                        async with get_db() as db:
                            await db.execute(
                                sa_update(DBSession)
                                .where(DBSession.id == UUID(params.session_id))
                                .values(volunteer_id=volunteer_id)
                            )
                        logger.info(f"[{params.session_id}] volunteer_id={volunteer_id} linked to session")
                    except Exception as e:
                        logger.warning(f"[{params.session_id}] Could not update volunteer_id on session: {e}")
            else:
                logger.warning(f"[{params.session_id}] No volunteer_id obtained — registration may have failed")

        if volunteer_id:
            # For new volunteers, skip the PUT update_user (just created with correct data).
            # Only create the user-profile (API 2).
            profile_ok = await volunteering_client.save_volunteer_profile(
                volunteer_id=volunteer_id,
                profile_data={},  # uses hardcoded defaults in _build_profile_payload
                existing_profile_id=None,
            )
            result["registry_sync"] = {"profile_created": profile_ok}
            logger.info(f"[{params.session_id}] ── REGISTRATION DONE ── volunteer_id={volunteer_id}, profile_created={profile_ok}")
        else:
            logger.warning(f"[{params.session_id}] ── REGISTRATION SKIPPED ── no volunteer_id available")

    return result


@mcp.tool()
async def list_sessions(params: ListSessionsInput) -> dict:
    """
    List sessions, optionally filtered by status.

    Args:
        status: active | paused | completed | abandoned (omit for all)
        limit:  Max sessions to return (default 50)
    """
    return await session_service.list_sessions(status=params.status, limit=params.limit)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE TOOLS (reads/writes MCP DB working copy)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_missing_fields(params: GetMissingFieldsInput) -> dict:
    """
    Get fields still needed from the volunteer.
    Reads from MCP DB profile cache (fast, no API call).

    Returns:
        missing_fields, confirmed_fields, completion_percentage
    """
    return await profile_service.get_missing_fields(params.session_id)


@mcp.tool()
async def save_volunteer_fields(params: SaveVolunteerFieldsInput) -> dict:
    """
    Save confirmed volunteer profile fields to MCP DB working copy.
    Does NOT call Serve Registry — that happens automatically at onboarding_complete.

    Args:
        session_id: UUID of the session
        fields:     Dict of field names → values (see schema for supported keys)

    Returns:
        saved_fields list
    """
    return await profile_service.save_fields(session_id=params.session_id, fields=params.fields)


@mcp.tool()
async def get_volunteer_profile(params: GetSessionInput) -> dict:
    """
    Get the complete volunteer profile from MCP DB cache.

    Returns:
        Full profile dict including Serve Registry source info
    """
    return await profile_service.get_profile(params.session_id)


@mcp.tool()
async def evaluate_readiness(params: EvaluateReadinessInput) -> dict:
    """
    Evaluate if a volunteer is ready for the selection phase.

    Returns:
        ready_for_selection (bool), missing_fields, recommendation, reason
    """
    return await profile_service.evaluate_readiness(params.session_id)


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def save_message(params: SaveMessageInput) -> dict:
    """
    Save a conversation message to MCP DB.

    Args:
        session_id:       UUID of the session
        role:             user | assistant
        content:          Message text
        agent:            Agent identifier (onboarding, need, selection…)
        message_metadata: Optional {token_count, latency_ms, model}
    """
    return await session_service.save_message(
        session_id=params.session_id,
        role=params.role,
        content=params.content,
        agent=params.agent,
        message_metadata=params.message_metadata,
    )


@mcp.tool()
async def get_conversation(params: GetConversationInput) -> dict:
    """
    Get conversation history for a session.

    Args:
        session_id: UUID of the session
        limit:      Max messages to return (default 50)
    """
    return await session_service.get_conversation(session_id=params.session_id, limit=params.limit)


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def save_memory_summary(params: SaveMemorySummaryInput) -> dict:
    """
    Save a compressed conversation memory summary (persisted to DB).

    Args:
        session_id:   UUID of the session
        summary_text: Summarised conversation context
        key_facts:    List of extracted key facts about the volunteer
        volunteer_id: Serve Registry osid (if known)

    Returns:
        summary_id, version, key_facts_count
    """
    return await memory_service.save_summary(
        session_id=params.session_id,
        summary_text=params.summary_text,
        key_facts=params.key_facts or [],
        volunteer_id=params.volunteer_id,
    )


@mcp.tool()
async def get_memory_summary(params: GetMemorySummaryInput) -> dict:
    """
    Retrieve the latest memory summary for a session (from DB).

    Returns:
        data: {summary_id, summary_text, key_facts, version, created_at}
              or null if no summary exists
    """
    return await memory_service.get_summary(params.session_id)


# ─────────────────────────────────────────────────────────────────────────────
# ENGAGEMENT HYBRID TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def engagement_save_confirmed_signals(params: EngagementSaveConfirmedSignalsInput) -> dict:
    """
    Persist confirmed engagement signals into the session sub_state.
    """
    return await engagement_service.save_confirmed_signals(
        session_id=params.session_id,
        signals=params.signals,
    )


@mcp.tool()
async def engagement_update_volunteer_status(params: EngagementUpdateVolunteerStatusInput) -> dict:
    """
    Persist the current engagement status and reason into the session state.
    """
    return await engagement_service.update_volunteer_status(
        session_id=params.session_id,
        volunteer_status=params.volunteer_status,
        reason=params.reason,
        signals=params.signals,
    )


@mcp.tool()
async def engagement_prepare_fulfillment_handoff(params: EngagementPrepareFulfillmentHandoffInput) -> dict:
    """
    Build and persist the fulfillment handoff payload for the current engagement session.
    """
    return await engagement_service.prepare_fulfillment_handoff(
        session_id=params.session_id,
        signals=params.signals,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def log_event(params: LogEventInput) -> dict:
    """
    Log a telemetry/audit event (persisted to DB).

    Args:
        session_id:     UUID of the session
        event_type:     state_transition | agent_response | mcp_call | error | …
        agent:          Agent identifier
        data:           Event payload
        domain:         volunteer | need | system
        source_service: orchestrator | onboarding_agent | need_agent
        duration_ms:    Operation duration for performance tracking
    """
    return await session_service.log_event(
        session_id=params.session_id,
        event_type=params.event_type,
        agent=params.agent,
        data=params.data or {},
        domain=params.domain,
        source_service=params.source_service,
        duration_ms=params.duration_ms,
    )


@mcp.tool()
async def emit_handoff_event(params: EmitHandoffEventInput) -> dict:
    """
    Record an agent handoff in the dedicated agent_handoff_log table.

    Args:
        session_id:   UUID of the session
        from_agent:   Source agent (onboarding, need, …)
        to_agent:     Target agent
        handoff_type: agent_transition | resume | escalation | pause
        payload:      Handoff context data
        reason:       Human-readable reason
    """
    return await session_service.emit_handoff_event(
        session_id=params.session_id,
        from_agent=params.from_agent,
        to_agent=params.to_agent,
        handoff_type=params.handoff_type,
        payload=params.payload,
        reason=params.reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEED COORDINATION TOOLS — Coordinator (delegates to Serve Registry)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def resolve_coordinator_identity(params: ResolveCoordinatorInput) -> dict:
    """
    Resolve coordinator identity from Serve Registry.
    Provide email for reliable lookup. WhatsApp-only resolution requires email
    to be collected first.

    Args:
        whatsapp_number: WhatsApp contact number
        email:           Email address (preferred for lookup)
        name:            Name hint for disambiguation

    Returns:
        status: linked | unlinked | ambiguous
        coordinator: Serve Registry coordinator data if found
        linked_schools: List of linked entity IDs
    """
    return await coordinator_service.resolve_identity(
        whatsapp_number=params.whatsapp_number,
        email=params.email,
        name=params.name,
    )


@mcp.tool()
async def create_coordinator(params: CreateCoordinatorInput) -> dict:
    """
    Register a new coordinator in Serve Registry.

    Args:
        name:            Full name of the coordinator
        whatsapp_number: WhatsApp contact number
        email:           Email address

    Returns:
        Created coordinator profile with Serve Registry ID
    """
    return await coordinator_service.create_coordinator(
        name=params.name,
        whatsapp_number=params.whatsapp_number,
        email=params.email,
    )


@mcp.tool()
async def map_coordinator_to_school(params: MapCoordinatorToSchoolInput) -> dict:
    """
    Link a coordinator to a school (entity) in Serve Need Service.

    Args:
        coordinator_id: Serve Registry coordinator osid
        school_id:      Serve Need Service entity UUID
    """
    return await school_service.link_coordinator(
        school_id=params.school_id,
        coordinator_id=params.coordinator_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEED COORDINATION TOOLS — School / Entity (delegates to Serve Need Service)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def resolve_school_context(params: ResolveSchoolContextInput) -> dict:
    """
    Resolve school context from Serve Need Service.

    Args:
        coordinator_id: Serve Registry coordinator osid (fetches linked entities)
        school_hint:    School name / partial name for search

    Returns:
        status: existing | new | ambiguous | multiple
        school: Entity data if found
        previous_needs: List of past needs for this school
    """
    return await school_service.resolve_context(
        coordinator_id=params.coordinator_id,
        school_hint=params.school_hint,
    )


@mcp.tool()
async def create_school_context(params: CreateSchoolContextInput) -> dict:
    """
    Create a new school (entity) in Serve Need Service.

    Args:
        name:            School name
        location:        Location description (village/town/district)
        contact_number:  Contact phone
        coordinator_id:  Coordinator to auto-link (optional)
        district:        District name
        state:           State name

    Returns:
        Created entity with Serve Need Service ID
    """
    return await school_service.create_school(
        name=params.name,
        location=params.location,
        contact_number=params.contact_number,
        coordinator_id=params.coordinator_id,
        district=params.district,
        state=params.state,
    )


@mcp.tool()
async def fetch_previous_need_context(params: FetchPreviousNeedContextInput) -> dict:
    """
    Fetch school details and previous needs from Serve Need Service.

    Args:
        school_id: Serve Need Service entity UUID

    Returns:
        school, previous_needs (list), needs_count
    """
    return await school_service.fetch_previous_needs(params.school_id)


# ─────────────────────────────────────────────────────────────────────────────
# NEED COORDINATION TOOLS — Need Lifecycle (MCP DB draft → Serve Need Service)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def start_need_session(
    channel: str = "web_ui",
    whatsapp_number: Optional[str] = None,
    channel_metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Start a new need coordination session (wrapper — prefer start_session directly).
    """
    return await need_service.start_session(
        channel=channel,
        whatsapp_number=whatsapp_number,
        channel_metadata=channel_metadata,
    )


@mcp.tool()
async def resume_need_context(params: ResumeSessionInput) -> dict:
    """Resume a need session with its current draft."""
    return await need_service.resume_context(params.session_id)


@mcp.tool()
async def advance_need_state(params: AdvanceSessionStateInput) -> dict:
    """Advance a need session to a new workflow state."""
    return await need_service.advance_state(
        session_id=params.session_id,
        new_state=params.new_state,
        sub_state=params.sub_state,
    )


@mcp.tool()
async def create_or_update_need_draft(params: CreateOrUpdateNeedDraftInput) -> dict:
    """
    Create or update the working need draft (stored in MCP DB).
    The draft is only pushed to Serve Need Service when submit_need_for_approval is called.

    Args:
        session_id:           UUID of the need session
        subjects:             List of subject names (mathematics, science, english…)
        grade_levels:         List of grades ("1"-"12")
        student_count:        Number of students
        time_slots:           [{day, startTime, endTime}] or simple string list
        start_date:           ISO date string (YYYY-MM-DD)
        duration_weeks:       Number of weeks
        schedule_preference:  Schedule description / frequency
        special_requirements: Any special notes
        coordinator_osid:     Serve Registry coordinator osid (set when resolved)
        entity_id:            Serve Need Service entity UUID (set when school resolved)
    """
    need_data: Dict[str, Any] = {}
    if params.subjects            is not None: need_data["subjects"]             = params.subjects
    if params.grade_levels        is not None: need_data["grade_levels"]         = params.grade_levels
    if params.student_count       is not None: need_data["student_count"]        = params.student_count
    if params.time_slots          is not None: need_data["time_slots"]           = params.time_slots
    if params.start_date          is not None: need_data["start_date"]           = params.start_date
    if params.duration_weeks      is not None: need_data["duration_weeks"]       = params.duration_weeks
    if params.schedule_preference is not None: need_data["schedule_preference"]  = params.schedule_preference
    if params.grade_schedule      is not None: need_data["grade_schedule"]       = params.grade_schedule
    if params.skipped_grades      is not None: need_data["skipped_grades"]       = params.skipped_grades
    if params.special_requirements is not None: need_data["special_requirements"] = params.special_requirements
    if params.coordinator_osid    is not None: need_data["coordinator_osid"]     = params.coordinator_osid
    if params.entity_id           is not None: need_data["entity_id"]            = params.entity_id

    return await need_service.save_need_draft(
        session_id=params.session_id,
        need_data=need_data,
    )


@mcp.tool()
async def get_missing_need_fields(params: GetMissingFieldsInput) -> dict:
    """
    Get mandatory need fields still missing from the draft.

    Returns:
        missing_fields, confirmed_fields, completion_percentage
    """
    return await need_service.get_missing_fields(params.session_id)


@mcp.tool()
async def evaluate_need_submission_readiness(params: GetMissingFieldsInput) -> dict:
    """
    Evaluate whether the need draft is ready for submission.

    Returns:
        is_ready (bool), missing_fields, warnings, completion_percentage, recommendation
    """
    return await need_service.evaluate_readiness(params.session_id)


@mcp.tool()
async def submit_need_for_approval(params: SubmitNeedInput) -> dict:
    """
    Submit the completed need draft to Serve Need Service (POST /need/raise).
    The draft must have coordinator_osid and entity_id set.

    Args:
        need_id: MCP DB draft UUID (from create_or_update_need_draft response)

    Returns:
        serve_need_id (Serve Need Service UUID), status, need details
    """
    return await need_service.submit_for_approval(params.need_id)


@mcp.tool()
async def update_need_status(params: UpdateNeedStatusInput) -> dict:
    """
    Update the status of a need in both MCP DB and Serve Need Service.

    Args:
        need_id:  MCP DB draft UUID
        status:   draft | pending_approval | approved | refinement_required
                  | paused | rejected | submitted
        comments: Admin/reviewer comments
    """
    return await need_service.update_status(
        need_id=params.need_id,
        status=params.status,
        comments=params.comments,
    )


@mcp.tool()
async def prepare_fulfillment_handoff(params: PrepareHandoffInput) -> dict:
    """
    Prepare a handoff payload for the fulfillment agent.
    Assembles need details + school + coordinator from Serve Need Service.

    Args:
        need_id: MCP DB draft UUID

    Returns:
        Complete handoff payload: need_details, school, coordinator_osid, approval_status
    """
    return await need_service.prepare_fulfillment_handoff(params.need_id)


@mcp.tool()
async def pause_need_session(params: PauseNeedSessionInput) -> dict:
    """Pause a need session for later resumption."""
    return await need_service.pause_session(session_id=params.session_id, reason=params.reason)


@mcp.tool()
async def save_need_message(params: SaveNeedMessageInput) -> dict:
    """Save a conversation message in a need session (persisted to DB)."""
    return await need_service.save_message(
        session_id=params.session_id, role=params.role,
        content=params.content, agent=params.agent
    )


@mcp.tool()
async def log_need_event(params: LogNeedEventInput) -> dict:
    """Log a need lifecycle audit event (persisted to telemetry_events table)."""
    return await need_service.log_event(
        session_id=params.session_id, event_type=params.event_type, data=params.data
    )


@mcp.tool()
async def emit_need_handoff_event(params: EmitNeedHandoffInput) -> dict:
    """Emit a handoff event in the need workflow (persisted to agent_handoff_log)."""
    return await need_service.emit_handoff_event(
        session_id=params.session_id,
        from_agent=params.from_agent,
        to_agent=params.to_agent,
        payload=params.payload,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVABILITY TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_server_health() -> dict:
    """
    Get MCP server health status including DB and Serve Registry connectivity.
    """
    from services.database import is_db_healthy, check_db_health
    import httpx
    from config import VOLUNTEERING_SERVICE_URL

    db_ok = await check_db_health()

    registry_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{VOLUNTEERING_SERVICE_URL}/user/status",
                                  params={"status": "ACTIVE"})
            registry_ok = r.status_code < 500
    except Exception:
        registry_ok = False

    return {
        "status":             "healthy" if db_ok else "degraded",
        "db_healthy":         db_ok,
        "registry_reachable": registry_ok,
        "mcp_port":           MCP_PORT,
    }


@mcp.tool()
async def get_session_analytics(params: GetSessionAnalyticsInput) -> dict:
    """
    Session analytics summary: counts by status, channel, persona and user_type.
    Useful for monitoring platform adoption and engagement trends.

    Args:
        date_from: ISO date YYYY-MM-DD (defaults to last 30 days)
        date_to:   ISO date YYYY-MM-DD (defaults to today)

    Returns:
        total_sessions, by_status, by_channel, by_persona, by_user_type,
        avg_messages_per_session, onboarding_completion_rate
    """
    from datetime import date, timedelta
    from services.database import get_db, is_db_healthy, Session as DBSession
    from services.database import ConversationMessage
    from sqlalchemy import select, func, and_

    date_to_val   = date.fromisoformat(params.date_to)   if params.date_to   else date.today()
    date_from_val = date.fromisoformat(params.date_from) if params.date_from else date_to_val - timedelta(days=30)

    if not is_db_healthy():
        return {"status": "error", "error_message": "Database not available"}

    try:
        from datetime import datetime
        dt_from = datetime.combine(date_from_val, datetime.min.time())
        dt_to   = datetime.combine(date_to_val,   datetime.max.time())

        async with get_db() as db:
            # Total sessions in window
            total_q = await db.execute(
                select(func.count()).select_from(DBSession)
                .where(and_(DBSession.created_at >= dt_from, DBSession.created_at <= dt_to))
            )
            total = total_q.scalar() or 0

            # Group by status
            status_q = await db.execute(
                select(DBSession.status, func.count().label("cnt"))
                .where(and_(DBSession.created_at >= dt_from, DBSession.created_at <= dt_to))
                .group_by(DBSession.status)
            )
            by_status = {row.status: row.cnt for row in status_q.fetchall()}

            # Group by channel
            channel_q = await db.execute(
                select(DBSession.channel, func.count().label("cnt"))
                .where(and_(DBSession.created_at >= dt_from, DBSession.created_at <= dt_to))
                .group_by(DBSession.channel)
            )
            by_channel = {row.channel: row.cnt for row in channel_q.fetchall()}

            # Group by persona
            persona_q = await db.execute(
                select(DBSession.persona, func.count().label("cnt"))
                .where(and_(DBSession.created_at >= dt_from, DBSession.created_at <= dt_to))
                .group_by(DBSession.persona)
            )
            by_persona = {row.persona: row.cnt for row in persona_q.fetchall()}

            # Group by user_type
            utype_q = await db.execute(
                select(DBSession.user_type, func.count().label("cnt"))
                .where(and_(DBSession.created_at >= dt_from, DBSession.created_at <= dt_to))
                .group_by(DBSession.user_type)
            )
            by_user_type = {(row.user_type or "unknown"): row.cnt for row in utype_q.fetchall()}

            # Average messages per session
            msg_q = await db.execute(
                select(
                    ConversationMessage.session_id,
                    func.count().label("msg_cnt")
                )
                .where(and_(
                    ConversationMessage.created_at >= dt_from,
                    ConversationMessage.created_at <= dt_to,
                ))
                .group_by(ConversationMessage.session_id)
            )
            msg_counts = [row.msg_cnt for row in msg_q.fetchall()]
            avg_messages = round(sum(msg_counts) / len(msg_counts), 1) if msg_counts else 0.0

            completed = by_status.get("completed", 0)
            completion_rate = round((completed / total * 100), 1) if total else 0.0

        return {
            "status":                   "success",
            "period":                   {"from": str(date_from_val), "to": str(date_to_val)},
            "total_sessions":           total,
            "by_status":                by_status,
            "by_channel":               by_channel,
            "by_persona":               by_persona,
            "by_user_type":             by_user_type,
            "avg_messages_per_session": avg_messages,
            "onboarding_completion_rate_pct": completion_rate,
        }
    except Exception as e:
        logger.error(f"get_session_analytics error: {e}")
        return {"status": "error", "error_message": str(e)}


@mcp.tool()
async def get_registry_sync_status() -> dict:
    """
    Show volunteer sessions where profile has not yet been synced to Serve Registry.
    Helps detect stuck onboarding flows.

    Returns:
        unsynced_count, sessions (list of session_id, actor_id, created_at, stage),
        synced_today_count
    """
    from services.database import get_db, is_db_healthy
    from services.database import VolunteerProfile, Session as DBSession
    from sqlalchemy import select, and_, func
    from datetime import date, datetime

    if not is_db_healthy():
        return {"status": "error", "error_message": "Database not available"}

    try:
        async with get_db() as db:
            # Profiles never synced
            unsynced_q = await db.execute(
                select(
                    VolunteerProfile.session_id,
                    VolunteerProfile.created_at,
                )
                .where(VolunteerProfile.registry_synced_at.is_(None))
                .order_by(VolunteerProfile.created_at.desc())
                .limit(100)
            )
            unsynced_rows = unsynced_q.fetchall()

            # Enrich with session data
            unsynced_sessions = []
            for row in unsynced_rows:
                sess_q = await db.execute(
                    select(DBSession.actor_id, DBSession.stage, DBSession.status)
                    .where(DBSession.id == row.session_id)
                )
                sess = sess_q.fetchone()
                unsynced_sessions.append({
                    "session_id": str(row.session_id),
                    "actor_id":   sess.actor_id if sess else None,
                    "stage":      sess.stage    if sess else None,
                    "status":     sess.status   if sess else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                })

            # Synced today
            today_start = datetime.combine(date.today(), datetime.min.time())
            synced_today_q = await db.execute(
                select(func.count()).select_from(VolunteerProfile)
                .where(VolunteerProfile.registry_synced_at >= today_start)
            )
            synced_today = synced_today_q.scalar() or 0

        return {
            "status":           "success",
            "unsynced_count":   len(unsynced_sessions),
            "synced_today_count": synced_today,
            "unsynced_sessions": unsynced_sessions,
        }
    except Exception as e:
        logger.error(f"get_registry_sync_status error: {e}")
        return {"status": "error", "error_message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MCP PROMPTS  — reusable system prompt templates for each agent
# ─────────────────────────────────────────────────────────────────────────────

@mcp.prompt()
def volunteer_onboarding_prompt() -> str:
    """
    System prompt for the Volunteer Onboarding Agent.
    Guides LLMs through collecting volunteer profile data conversationally.
    """
    return """You are the Serve Volunteer Onboarding Assistant.

Your goal is to warmly and conversationally collect the volunteer's profile information.
Always call `lookup_actor` first to identify the user, then `start_session` before proceeding.

## Required fields to collect (in natural conversation order):
1. full_name — "What is your full name?"
2. email — "What is your email address?"
3. phone — "What is your mobile number?"
4. location — City or district they are based in
5. skills — Subjects or skills they can teach (e.g. mathematics, science, english, coding)
6. availability — How many hours per week, which days/times

## Optional enrichment (ask after required fields):
- gender, dob, languages, interests, qualification, years_of_experience,
  employment_status, motivation, experience_level

## Rules:
- Ask ONE question at a time. Do not ask multiple questions in one message.
- Confirm values before saving: "Just to confirm, your name is [X] — is that right?"
- After confirmation call `save_volunteer_fields` with ONLY the confirmed fields.
- Call `get_missing_fields` before each question to know what is still needed.
- When all required fields are done call `advance_session_state` with new_state="onboarding_complete".
- Never make up or assume field values.
- If the user wants to stop, call `advance_session_state` with new_state="paused".
"""


@mcp.prompt()
def need_coordinator_prompt() -> str:
    """
    System prompt for the Need Coordination Agent.
    Guides LLMs through collecting school need information conversationally.
    """
    return """You are the Serve Need Coordination Assistant.

Your goal is to help school coordinators raise a teaching need for their school.
Always call `start_session` with persona="need_coordinator" at the beginning.

## Workflow:
1. **Identify coordinator** → call `resolve_coordinator_identity` with email or WhatsApp number.
   - If not found → call `create_coordinator` and note the returned ID.
2. **Identify school** → call `resolve_school_context` with coordinator_id.
   - If not found → call `create_school_context` after collecting name and location.
   - Call `fetch_previous_need_context` to show history.
3. **Collect need details** → call `create_or_update_need_draft` incrementally.
   - subjects: which subjects are needed
   - grade_levels: which grades (e.g. ["6", "7", "8"])
   - student_count: approx number of students
   - time_slots: preferred days/times
   - start_date: when to start (YYYY-MM-DD)
   - duration_weeks: for how many weeks
4. **Review & submit** → call `evaluate_need_submission_readiness`.
   - If ready → call `submit_need_for_approval`.
5. **Handoff** → call `prepare_fulfillment_handoff` and `emit_need_handoff_event`.

## Rules:
- Ask ONE question at a time.
- Confirm critical details (student_count, subjects) before saving.
- Never invent data. If unsure, ask the coordinator to confirm.
- The needTypeId for Online Teaching is fixed — the system sets it automatically.
"""


@mcp.prompt()
def memory_summarizer_prompt() -> str:
    """
    System prompt for the Memory Summarizer Agent.
    Guides LLMs in compressing conversation history into a reusable summary.
    """
    return """You are the Serve Conversation Memory Summarizer.

Given a conversation history, create a concise summary for future sessions.

## Output format — call `save_memory_summary` with:
- summary_text: 2-4 sentences covering the key outcome and volunteer/coordinator intent.
- key_facts: A list of 5-10 bullet-point facts (name, skills, availability, location,
  school name, grade levels, subjects raised, current workflow stage, etc.)

## Rules:
- Only include facts explicitly stated in the conversation. Do not infer.
- Include the current workflow stage and what the next step should be.
- Keep summary_text under 150 words.
- Include any blockers or reasons for pausing if the session was paused.

## When to summarize:
- When the session state advances to "paused", "completed", or "handoff_pending".
- When the agent is about to be replaced by a different agent.
"""


@mcp.prompt()
def returning_volunteer_prompt() -> str:
    """
    System prompt for the Returning Volunteer Engagement Agent.
    Guides LLMs in re-engaging volunteers who have previously onboarded.
    """
    return """You are the Serve Volunteer Engagement Assistant.

The volunteer you are speaking with has already completed onboarding.
Always call `resume_session` or `get_memory_summary` first to load their profile context.

## Your goals:
1. Welcome the volunteer back warmly. Reference their name and skills from the loaded context.
2. Understand why they are returning (new availability, want to change skills, find opportunities).
3. If they want to update their profile → collect new values and call `save_volunteer_fields`.
4. If they need to see opportunities → use the selection/fulfillment workflow.
5. If they want to pause → call `advance_session_state` with new_state="paused"
   and then `save_memory_summary`.

## Rules:
- Do not ask for information already in their profile unless they indicate it has changed.
- Confirm any changes before saving.
- Be encouraging and appreciate their commitment to volunteering.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ENGAGEMENT & FULFILLMENT AGENT TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_volunteer_fulfillment_history(params: GetVolunteerFulfillmentHistoryInput) -> dict:
    """
    Fetch a volunteer's completed/closed fulfillment history, enriched with
    need details (school, subject, grade, schedule).

    Used by: Engagement Agent (to build personalised greeting context)

    Args:
        volunteer_id: Serve Registry volunteer osid
        page:         Page number (default 0)
        size:         Page size (default 50)

    Returns:
        fulfillments: list of {fulfillment_id, need_id, school_name, subject,
                               grade_levels, schedule, start_date, end_date,
                               fulfillment_status}
        total: count of completed/closed fulfillments
    """
    from services.serve_registry_client import fulfillment_client, need_service_client

    COMPLETED_STATUSES = {"Completed", "Closed"}

    raw = await fulfillment_client.get_fulfillments_for_volunteer(
        params.volunteer_id, page=params.page, size=params.size
    )

    enriched = []
    for f in raw:
        status = f.get("fulfillmentStatus", "")
        if status not in COMPLETED_STATUSES:
            continue
        need_id = f.get("needId", "")
        need_detail = {}
        if need_id:
            try:
                need_detail = await need_service_client.get_need_details(need_id) or {}
            except Exception:
                pass
        enriched.append({
            "fulfillment_id":    f.get("id"),
            "need_id":           need_id,
            "school_name":       need_detail.get("name", ""),
            "subjects":          need_detail.get("subjects", []),
            "grade_levels":      need_detail.get("grade_levels", []),
            "schedule":          need_detail.get("days", ""),
            "start_date":        need_detail.get("start_date", ""),
            "end_date":          need_detail.get("end_date", ""),
            "fulfillment_status": status,
        })

    return {"status": "success", "fulfillments": enriched, "total": len(enriched)}


@mcp.tool()
async def check_active_nominations(params: CheckActiveNominationsInput) -> dict:
    """
    Check whether a volunteer already has an active nomination in the pipeline.

    Used by: Engagement Agent (to skip re-engagement if already nominated)

    Args:
        volunteer_id: Serve Registry volunteer osid

    Returns:
        has_active_nomination: bool
        nominations: list of active nomination objects
    """
    from services.serve_registry_client import nomination_client

    ACTIVE_STATUSES = {"Nominated", "Approved", "Proposed"}

    all_nominations = await nomination_client.get_nominations_for_volunteer(params.volunteer_id)
    active = [n for n in all_nominations if n.get("nominationStatus") in ACTIVE_STATUSES]

    return {
        "status":                "success",
        "has_active_nomination": len(active) > 0,
        "nominations":           active,
    }


@mcp.tool()
async def get_engagement_context(params: GetEngagementContextInput) -> dict:
    """
    Convenience tool — looks up volunteer by phone, then fetches fulfillment history
    + volunteer profile in a single call. Engagement agent calls this once at session start.

    Used by: Engagement Agent

    Args:
        phone: Volunteer's WhatsApp/mobile number

    Returns:
        volunteer_id:       resolved Serve Registry osid
        volunteer_name:     full name from registry
        fulfillment_history: list of enriched past engagements (school, subject, grade, slots)
        volunteer_profile:  profile dict from Serve Registry (may be null)
    """
    return await engagement_service.get_engagement_context(params.phone)


@mcp.tool()
async def get_engagement_context_by_email(params: GetEngagementContextByEmailInput) -> dict:
    """
    Fallback — looks up volunteer by email when phone lookup fails.
    Returns the same structure as get_engagement_context.

    Used by: Engagement Agent (when volunteer's WhatsApp number doesn't match registry)

    Args:
        email: Volunteer's email address used during eVidyaloka registration
    """
    return await engagement_service.get_engagement_context_by_email(params.email)


@mcp.tool()
async def get_needs_for_entity(params: GetNeedsForEntityInput) -> dict:
    """
    Get open needs for a school/entity.

    Used by: Fulfillment Agent

    Args:
        entity_id: Serve Need Service entity UUID
        page: page number
        size: page size

    Returns:
        needs: list of needs for the entity
    """
    from services.serve_registry_client import need_service_client

    needs = await need_service_client.get_needs_for_entity(
        params.entity_id,
        page=params.page,
        size=params.size,
    )
    return {"status": "success", "needs": needs, "total": len(needs)}


@mcp.tool()
async def get_all_entities() -> dict:
    """
    Get all schools/entities from Serve Need Service.

    Used by: Fulfillment Agent (fallback when preferred school has no time-matching needs)

    Returns:
        entities: list of schools with entity_id, name, district, state
    """
    from services.serve_registry_client import need_service_client
    raw = await need_service_client.search_entities()
    # Ensure entity_id is explicitly set alongside id
    entities = [{**e, "entity_id": e.get("id")} for e in raw]
    return {"status": "success", "entities": entities, "total": len(entities)}


@mcp.tool()
async def search_approved_needs() -> dict:
    """
    Bulk-fetch all approved needs across all schools with enriched details.

    Used by: Fulfillment Agent MatchFinder for efficient matching at scale.
    Returns pre-filtered, enriched needs with school_name, subjects, grades, time_slots.
    """
    from services.serve_registry_client import need_service_client
    needs = await need_service_client.get_approved_needs_bulk()
    return {"status": "success", "needs": needs, "total": len(needs)}


@mcp.tool()
async def get_need_details(params: GetNeedDetailsInput) -> dict:
    """
    Get enriched details for a need.

    Used by: Fulfillment Agent

    Args:
        need_id: Serve Need Service need UUID

    Returns:
        need_details: flattened need detail object
    """
    from services.serve_registry_client import need_service_client

    details = await need_service_client.get_need_details(params.need_id)
    if not details:
        return {"status": "error", "error": f"Need {params.need_id} not found"}
    return {"status": "success", "need_details": details}


@mcp.tool()
async def nominate_volunteer_for_need(params: NominateVolunteerInput) -> dict:
    """
    Nominate a volunteer for a need.

    Used by: Fulfillment Agent (after matching volunteer to a need)

    Args:
        need_id:      Serve Need Service need UUID
        volunteer_id: Serve Registry volunteer osid

    Returns:
        nomination object with nominationStatus='Nominated'
    """
    from services.serve_registry_client import nomination_client

    result = await nomination_client.nominate_volunteer(
        need_id=params.need_id,
        volunteer_id=params.volunteer_id,
    )
    if result:
        return {"status": "success", "nomination": result}
    return {"status": "error", "error": "Nomination failed — check need_id and volunteer_id"}


@mcp.tool()
async def confirm_nomination(params: ConfirmNominationInput) -> dict:
    """
    Confirm or reject a nomination (Approved / Rejected / Backfill etc.).

    Used by: Fulfillment Agent

    Args:
        volunteer_id:   Serve Registry volunteer osid
        nomination_id:  Nomination UUID
        status:         Nominated | Approved | Proposed | Backfill | Rejected

    Returns:
        Updated nomination object
    """
    from services.serve_registry_client import nomination_client

    result = await nomination_client.confirm_nomination(
        volunteer_id=params.volunteer_id,
        nomination_id=params.nomination_id,
        status=params.status,
    )
    if result:
        return {"status": "success", "nomination": result}
    return {"status": "error", "error": "Confirmation failed"}


@mcp.tool()
async def get_nominations_for_need(params: GetNominationsForNeedInput) -> dict:
    """
    Get all nominations for a need, optionally filtered by status.

    Used by: Fulfillment Agent (to see who's nominated for a need)

    Args:
        need_id: Serve Need Service need UUID
        status:  Optional filter — Nominated | Approved | Proposed | Backfill | Rejected

    Returns:
        nominations: list of nomination objects
    """
    from services.serve_registry_client import nomination_client

    if params.status:
        nominations = await nomination_client.get_nominations_for_need_by_status(
            params.need_id, params.status
        )
    else:
        nominations = await nomination_client.get_nominations_for_need(params.need_id)

    return {"status": "success", "nominations": nominations, "total": len(nominations)}


@mcp.tool()
async def get_recommended_volunteers(params: GetRecommendedVolunteersInput) -> dict:
    """
    Get recommended volunteers — either not yet nominated or already nominated.

    Used by: Fulfillment Agent (to find candidates for open needs)

    Args:
        already_nominated: False → volunteers not yet nominated (default)
                           True  → volunteers already nominated

    Returns:
        volunteers: list of UserResponse objects
    """
    from services.serve_registry_client import nomination_client

    if params.already_nominated:
        volunteers = await nomination_client.get_recommended_nominated()
    else:
        volunteers = await nomination_client.get_recommended_not_nominated()

    return {"status": "success", "volunteers": volunteers, "total": len(volunteers)}


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY ASSISTANT TOOLS — post-handshake delivery journey
# Used by: Delivery Agent (serve-delivery-agent-service)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def delivery_start_activation(params: DeliveryStartActivationInput) -> dict:
    """
    Create a post-handshake delivery record for an approved volunteer↔need.
    Reuses an existing active delivery for the same volunteer+need if present.

    Used by: Delivery Agent / ops trigger when a nomination is approved.
    """
    return await delivery_service.start_activation(**params.model_dump())


@mcp.tool()
async def delivery_get_context(params: DeliveryGetContextInput) -> dict:
    """
    Fetch full delivery context (assignment + activation + scheduled sessions +
    blockers + reschedule requests) by delivery_id or session_id.
    """
    return await delivery_service.get_delivery_context(
        delivery_id=params.delivery_id, session_id=params.session_id
    )


@mcp.tool()
async def delivery_confirm_acknowledgement(params: DeliveryConfirmAcknowledgementInput) -> dict:
    """Record that the volunteer or coordinator acknowledged the assignment."""
    return await delivery_service.confirm_acknowledgement(params.delivery_id, params.party)


@mcp.tool()
async def delivery_confirm_first_session_readiness(params: DeliveryConfirmFirstSessionReadinessInput) -> dict:
    """Record that the volunteer is ready for the first scheduled session."""
    return await delivery_service.confirm_first_session_readiness(params.delivery_id)


@mcp.tool()
async def delivery_complete_activation(params: DeliveryCompleteActivationInput) -> dict:
    """
    Mark activation complete. GATED — refuses unless the volunteer has
    acknowledged AND first-session readiness is confirmed.
    """
    return await delivery_service.complete_activation(params.delivery_id)


@mcp.tool()
async def delivery_create_scheduled_session(params: DeliveryCreateScheduledSessionInput) -> dict:
    """Create a scheduled teaching session within a delivery."""
    return await delivery_service.create_scheduled_session(**params.model_dump())


@mcp.tool()
async def delivery_get_scheduled_sessions(params: DeliveryGetScheduledSessionsInput) -> dict:
    """List a delivery's scheduled sessions (optionally today's only)."""
    return await delivery_service.get_scheduled_sessions(params.delivery_id, params.today_only)


@mcp.tool()
async def delivery_get_due_reminders(params: DeliveryGetDueRemindersInput) -> dict:
    """
    Return candidate reminder state for active deliveries: each non-terminal
    session plus which reminder types were already recorded. The delivery agent's
    policy engine decides which reminders are actually due.
    """
    return await delivery_service.get_due_reminders(delivery_id=params.delivery_id, now=params.now)


@mcp.tool()
async def delivery_mark_reminder(params: DeliveryMarkReminderInput) -> dict:
    """
    Record a reminder as sent/suppressed/responded. Idempotent — the
    (session, reminder_type) unique constraint makes duplicates a no-op.
    """
    return await delivery_service.mark_reminder(
        scheduled_session_id=params.scheduled_session_id,
        reminder_type=params.reminder_type,
        status=params.status,
        suppressed_reason=params.suppressed_reason,
    )


@mcp.tool()
async def delivery_record_session_outcome(params: DeliveryRecordSessionOutcomeInput) -> dict:
    """
    Record a verified session outcome (completed | partially_completed | missed |
    disrupted | unverified | cancelled | reschedule_requested | support_needed).
    Updates completed_sessions. Optional structured evidence (attendance_count,
    duration_minutes, disruption_type) may be included.
    """
    return await delivery_service.record_session_outcome(
        scheduled_session_id=params.scheduled_session_id,
        outcome=params.outcome, reason=params.reason, reported_by=params.reported_by,
        attendance_count=params.attendance_count, duration_minutes=params.duration_minutes,
        disruption_type=params.disruption_type,
    )


@mcp.tool()
async def delivery_log_blocker(params: DeliveryLogBlockerInput) -> dict:
    """Record a structured operational blocker on a delivery or session."""
    return await delivery_service.log_blocker(**params.model_dump())


@mcp.tool()
async def delivery_capture_reschedule(params: DeliveryCaptureRescheduleInput) -> dict:
    """Capture a reschedule request as PENDING — never auto-approved."""
    return await delivery_service.capture_reschedule_request(**params.model_dump())


@mcp.tool()
async def delivery_update_status(params: DeliveryUpdateStatusInput) -> dict:
    """Update the delivery-level status (active | at_risk | paused | escalated | ...)."""
    return await delivery_service.update_delivery_status(
        params.delivery_id, params.delivery_status, params.status_reason
    )


@mcp.tool()
async def delivery_evaluate_escalation(params: DeliveryEvaluateEscalationInput) -> dict:
    """
    Return deterministic continuity signals (consecutive missed/unverified, open
    blockers, reschedule count) plus a convenience escalate verdict from thresholds.
    """
    return await delivery_service.evaluate_escalation(params.delivery_id)


# ── Full-spec expansion: granular reads ────────────────────────────────────────

@mcp.tool()
async def delivery_read_assignment_context(params: DeliveryReadAssignmentContextInput) -> dict:
    """Focused read: volunteer/need/entity/coordinator/programme assignment details only."""
    return await delivery_service.read_assignment_context(params.delivery_id)


@mcp.tool()
async def delivery_read_activation_context(params: DeliveryReadActivationContextInput) -> dict:
    """Focused read: acknowledgement, readiness, and activation progress only."""
    return await delivery_service.read_activation_context(params.delivery_id)


@mcp.tool()
async def delivery_read_schedule_context(params: DeliveryReadScheduleContextInput) -> dict:
    """Focused read: upcoming/today/missed sessions."""
    return await delivery_service.read_schedule_context(params.delivery_id)


@mcp.tool()
async def delivery_read_session_context(params: DeliveryReadSessionContextInput) -> dict:
    """Focused read: one scheduled session plus its reminder history."""
    return await delivery_service.read_session_context(params.scheduled_session_id)


@mcp.tool()
async def delivery_read_delivery_history(params: DeliveryReadDeliveryHistoryInput) -> dict:
    """Focused read: outcome history, blockers, reschedules, cancelled sessions."""
    return await delivery_service.read_delivery_history(params.delivery_id)


# ── Full-spec expansion: signal processing & evaluation ────────────────────────

@mcp.tool()
async def delivery_extract_signals(params: DeliveryExtractSignalsInput) -> dict:
    """
    Deterministic, keyword-based structured guess at what a free-form message
    signals (completion/reschedule/blocker/readiness/availability/disengagement/
    support-request). Advisory only — never the authoritative recording path.
    """
    return delivery_service.extract_delivery_signals(params.text)


@mcp.tool()
async def delivery_detect_blockers(params: DeliveryDetectBlockersInput) -> dict:
    """Deterministic keyword classifier for blocker type. Advisory pre-check only."""
    return delivery_service.detect_blockers(params.text)


@mcp.tool()
async def delivery_get_missing_signals(params: DeliveryGetMissingSignalsInput) -> dict:
    """What evidence is still required before a given target can be recorded."""
    return await delivery_service.get_missing_signals(params.delivery_id, params.target)


@mcp.tool()
async def delivery_evaluate_activation(params: DeliveryEvaluateActivationInput) -> dict:
    """Dry-run of the complete_activation gate — reports readiness without committing."""
    return await delivery_service.evaluate_activation(params.delivery_id)


@mcp.tool()
async def delivery_evaluate_readiness(params: DeliveryEvaluateReadinessInput) -> dict:
    """Evaluate the multi-dimension readiness checklist (volunteer/coordinator/session/classroom/material/meeting_link/infrastructure)."""
    return await delivery_service.evaluate_readiness(params.delivery_id)


@mcp.tool()
async def delivery_set_readiness_dimension(params: DeliverySetReadinessDimensionInput) -> dict:
    """Set one dimension of the readiness checklist."""
    return await delivery_service.set_readiness_dimension(params.delivery_id, params.dimension, params.value)


@mcp.tool()
async def delivery_evaluate_delivery_health(params: DeliveryEvaluateDeliveryHealthInput) -> dict:
    """
    Richer than delivery_evaluate_escalation — adds blocker-age and
    reschedule-pattern signals on top of consecutive-miss/unverified counts.
    """
    return await delivery_service.evaluate_delivery_health(params.delivery_id)


@mcp.tool()
async def delivery_evaluate_next_action(params: DeliveryEvaluateNextActionInput) -> dict:
    """
    Read-only advisory: what the deterministic flow would do next. Reporting
    only — never grants tool-call authority to change state directly.
    """
    return await delivery_service.evaluate_next_action(params.delivery_id)


# ── Full-spec expansion: activation content & coordinator notification ─────────

@mcp.tool()
async def delivery_get_activation_content(params: DeliveryGetActivationContentInput) -> dict:
    """Fixed, deterministic template text (intro | instructions | resources) — never LLM-authored."""
    return await delivery_service.get_activation_content(params.delivery_id, params.content_type)


@mcp.tool()
async def delivery_notify_linked_stakeholder(params: DeliveryNotifyStakeholderInput) -> dict:
    """
    Notify the coordinator via WhatsApp using fixed template text. Honest about
    outcome — status reflects sent / failed / no_contact_on_file, never a
    false confirmation.
    """
    return await delivery_service.notify_linked_stakeholder(
        params.delivery_id, params.stakeholder, params.reason
    )


@mcp.tool()
async def delivery_set_coordinator_phone(params: DeliverySetCoordinatorPhoneInput) -> dict:
    """
    Self-healing capture: cache a coordinator's phone the first time they
    message in on their own, since there is no reliable coordinator_id -> phone
    directory anywhere in this stack.
    """
    return await delivery_service.set_coordinator_phone(params.delivery_id, params.phone)


# ── Full-spec expansion: reminder wrappers, history, suppression ───────────────

@mcp.tool()
async def delivery_send_session_day_reminder(params: DeliverySendSessionDayReminderInput) -> dict:
    """Named wrapper for the session-day reminder type."""
    return await delivery_service.send_session_day_reminder(params.scheduled_session_id, params.now)


@mcp.tool()
async def delivery_send_pre_session_reminder(params: DeliverySendPreSessionReminderInput) -> dict:
    """Named wrapper for the pre-session reminder type."""
    return await delivery_service.send_pre_session_reminder(params.scheduled_session_id, params.now)


@mcp.tool()
async def delivery_send_completion_check(params: DeliverySendCompletionCheckInput) -> dict:
    """Named wrapper for the completion-check reminder type."""
    return await delivery_service.send_completion_check(params.scheduled_session_id, params.now)


@mcp.tool()
async def delivery_send_followup_nudge(params: DeliverySendFollowupNudgeInput) -> dict:
    """Named wrapper for the follow-up nudge reminder type."""
    return await delivery_service.send_followup_nudge(params.scheduled_session_id, params.now)


@mcp.tool()
async def delivery_read_reminder_history(params: DeliveryReadReminderHistoryInput) -> dict:
    """Full reminder history for a delivery (all sessions, all types, all statuses)."""
    return await delivery_service.read_reminder_history(params.delivery_id)


@mcp.tool()
async def delivery_suppress_reminder(params: DeliverySuppressReminderInput) -> dict:
    """Explicit ops override — suppress one reminder type for one session, distinct from automatic policy suppression."""
    return await delivery_service.suppress_reminder(
        params.scheduled_session_id, params.reminder_type, params.reason
    )


# ── Full-spec expansion: session check-in ───────────────────────────────────────

@mcp.tool()
async def delivery_start_session_checkin(params: DeliveryStartCheckinInput) -> dict:
    """Mark a session's check-in window as active."""
    return await delivery_service.start_session_checkin(params.scheduled_session_id)


@mcp.tool()
async def delivery_close_checkin(params: DeliveryCloseCheckinInput) -> dict:
    """Close a session's operational check-in without closing the whole delivery."""
    return await delivery_service.close_checkin(params.scheduled_session_id)


# ── Full-spec expansion: blocker resolution & support ───────────────────────────

@mcp.tool()
async def delivery_update_blocker(params: DeliveryUpdateBlockerInput) -> dict:
    """Update a blocker's status/owner/resolution — closes the write-only-sink gap."""
    return await delivery_service.update_blocker(
        params.blocker_id, params.status, params.owner, params.resolution_notes
    )


@mcp.tool()
async def delivery_get_support_guidance(params: DeliveryGetSupportGuidanceInput) -> dict:
    """Static, deterministic guidance lookup for a given blocker type."""
    return delivery_service.get_support_guidance(params.blocker_type)


@mcp.tool()
async def delivery_create_ops_support_request(params: DeliveryCreateOpsSupportRequestInput) -> dict:
    """Lighter-weight than full escalation — flags something for ops without treating it as an emergency."""
    return await delivery_service.create_ops_support_request(
        params.delivery_id, params.reason, params.urgency
    )


# ── Full-spec expansion: reschedule resolution ──────────────────────────────────

@mcp.tool()
async def delivery_submit_reschedule(params: DeliverySubmitRescheduleInput) -> dict:
    """Move a captured reschedule request from pending to submitted (ops review)."""
    return await delivery_service.submit_reschedule_request(params.reschedule_request_id)


@mcp.tool()
async def delivery_read_reschedule_status(params: DeliveryReadRescheduleStatusInput) -> dict:
    """Read reschedule request(s) — by delivery_id (all) or a specific reschedule_request_id."""
    return await delivery_service.read_reschedule_status(params.delivery_id, params.reschedule_request_id)


@mcp.tool()
async def delivery_resolve_reschedule(params: DeliveryResolveRescheduleInput) -> dict:
    """
    Ops-only: move a reschedule request to approved/rejected — the piece that
    lets a request finally reach a real outcome. Not exposed to the
    volunteer-facing conversational agent.
    """
    return await delivery_service.resolve_reschedule_request(
        params.reschedule_request_id, params.status, params.resolution_notes
    )


# ── Full-spec expansion: risk, handoff enrichment, summaries ───────────────────

@mcp.tool()
async def delivery_raise_risk(params: DeliveryRaiseRiskInput) -> dict:
    """Set a delivery's risk_level, distinct from full escalation."""
    return await delivery_service.raise_delivery_risk(params.delivery_id, params.risk_level, params.reason)


@mcp.tool()
async def delivery_prepare_ops_handoff(params: DeliveryPrepareOpsHandoffInput) -> dict:
    """Build the fuller handoff bundle (recent outcomes, open blockers, pending reschedules) for use as emit_handoff_event's payload."""
    return await delivery_service.prepare_ops_handoff(params.delivery_id, params.reason)


@mcp.tool()
async def delivery_write_session_summary(params: DeliveryWriteSessionSummaryInput) -> dict:
    """Generate and store a deterministic, template-based session-level summary. Never LLM-generated."""
    return await delivery_service.write_session_summary(params.scheduled_session_id)


@mcp.tool()
async def delivery_write_delivery_summary(params: DeliveryWriteDeliverySummaryInput) -> dict:
    """Generate and store a deterministic, template-based delivery-level summary. Never LLM-generated."""
    return await delivery_service.write_delivery_summary(params.delivery_id)


# ─────────────────────────────────────────────────────────────────────────────
# VOLUNTEER FACT-STORE TOOLS
# ─────────────────────────────────────────────────────────────────────────────

from schemas import (
    FindVolunteerInput, CreateVolunteerInput,
    MergeVolunteerFactsInput, GetVolunteerFactsInput,
    CheckVolunteerCredentialInput, UpdateVolunteerRegistryStatusInput,
)


@mcp.tool()
async def find_volunteer(params: FindVolunteerInput) -> dict:
    """
    Find a volunteer in the platform fact-store by email, phone, or Serve Registry ID.
    Returns the volunteer record with all facts, or {"status": "not_found"}.

    Args:
        email: Email address to search
        phone: Phone number to search
        serve_registry_id: Serve Registry osid to search
    """
    result = await volunteer_service.find_volunteer(
        email=params.email,
        phone=params.phone,
        serve_registry_id=params.serve_registry_id,
    )
    if result:
        return {"status": "success", "volunteer": result}
    return {"status": "not_found"}


@mcp.tool()
async def create_volunteer_record(params: CreateVolunteerInput) -> dict:
    """
    Create a new volunteer in the platform fact-store.
    Called when onboarding completes and the volunteer is registered.

    Args:
        full_name: Volunteer's full name
        phone: Phone number
        email: Email address
        serve_registry_id: Serve Registry osid (if already registered)
        facts: Initial facts dict (eligibility, platform_status, etc.)
    """
    result = await volunteer_service.create_volunteer(
        full_name=params.full_name,
        phone=params.phone,
        email=params.email,
        serve_registry_id=params.serve_registry_id,
        facts=params.facts,
    )
    if result:
        return {"status": "success", "volunteer": result}
    return {"status": "error", "error": "Failed to create volunteer record"}


@mcp.tool()
async def merge_volunteer_facts(params: MergeVolunteerFactsInput) -> dict:
    """
    Merge new facts into a volunteer's existing fact-set.
    Used by agents to persist their outputs (credentials, preferences, commitments).

    Merge behavior:
    - Top-level keys: overwrite
    - credentials: merge per-category (add/update category, don't remove others)
    - preferences: overwrite entirely (latest wins)
    - commitments: append to list

    Args:
        volunteer_id: Platform volunteer UUID
        facts: Facts to merge
    """
    result = await volunteer_service.merge_facts(
        volunteer_id=params.volunteer_id,
        new_facts=params.facts,
    )
    if result is not None:
        return {"status": "success", "facts": result}
    return {"status": "error", "error": "Failed to merge facts"}


@mcp.tool()
async def get_volunteer_facts(params: GetVolunteerFactsInput) -> dict:
    """
    Get the full fact-set for a volunteer.
    Used by the orchestrator for gap analysis before routing.

    Args:
        volunteer_id: Platform volunteer UUID

    Returns:
        All stored facts (eligibility, credentials, preferences, commitments)
    """
    result = await volunteer_service.get_facts(params.volunteer_id)
    if result is not None:
        return {"status": "success", "facts": result}
    return {"status": "not_found"}


@mcp.tool()
async def check_volunteer_credential(params: CheckVolunteerCredentialInput) -> dict:
    """
    Check if a volunteer has a specific credential at the required status.
    Used by fulfillment agent to verify eligibility before nomination.

    Args:
        volunteer_id: Platform volunteer UUID
        category: Credential category (english_teaching, hindi_teaching, etc.)
        required_status: Status to check for (default: recommended)

    Returns:
        has_credential (bool), credential details if found
    """
    has_it = await volunteer_service.has_credential(
        volunteer_id=params.volunteer_id,
        category=params.category,
        required_status=params.required_status,
    )
    credential = await volunteer_service.get_credential(params.volunteer_id, params.category)
    return {
        "status": "success",
        "has_credential": has_it,
        "credential": credential,
    }


@mcp.tool()
async def update_volunteer_registry_status(params: UpdateVolunteerRegistryStatusInput) -> dict:
    """
    Update a volunteer's status in the external Serve Registry.
    Called after selection evaluation to mark volunteer as Recommended or OnHold.

    Args:
        volunteer_id: Serve Registry volunteer osid (the external registry ID, not platform UUID)
        status: New status — must be 'Recommended' or 'OnHold'

    Returns:
        success (bool) and the applied status
    """
    allowed = {"Recommended", "OnHold"}
    if params.status not in allowed:
        return {"status": "error", "error": f"Status must be one of {allowed}, got '{params.status}'"}

    from services.serve_registry_client import volunteering_client
    success = await volunteering_client.update_volunteer_status(
        volunteer_id=params.volunteer_id,
        status=params.status,
    )
    if success:
        return {"status": "success", "volunteer_id": params.volunteer_id, "new_status": params.status}
    return {"status": "error", "error": f"Failed to update volunteer {params.volunteer_id} status to {params.status}"}


# ─────────────────────────────────────────────────────────────────────────────
# NUDGE QUEUE TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
async def nudge_schedule(params: NudgeScheduleInput) -> dict:
    """Schedule a nudge reminder for an inactive WhatsApp session."""
    session_id, volunteer_phone = params.session_id, params.volunteer_phone
    nudge_number, scheduled_at = params.nudge_number, params.scheduled_at
    volunteer_name = params.volunteer_name
    from datetime import datetime
    from services.database import get_db, NudgeQueue, is_db_healthy
    from uuid import UUID
    if not is_db_healthy():
        return {"status": "error", "error": "DB not available"}
    try:
        async with get_db() as db:
            # Check for existing unsent nudge for this session+number
            from sqlalchemy import select, and_
            existing = (await db.execute(
                select(NudgeQueue).where(and_(
                    NudgeQueue.session_id == UUID(session_id),
                    NudgeQueue.nudge_number == nudge_number,
                    NudgeQueue.sent_at.is_(None),
                    NudgeQueue.cancelled == False,
                ))
            )).scalars().first()
            if existing:
                return {"status": "already_scheduled", "id": str(existing.id)}

            nudge = NudgeQueue(
                session_id=UUID(session_id),
                volunteer_phone=volunteer_phone,
                volunteer_name=volunteer_name,
                nudge_number=nudge_number,
                scheduled_at=datetime.fromisoformat(scheduled_at),
            )
            db.add(nudge)
            return {"status": "success", "id": str(nudge.id)}
    except Exception as e:
        logger.error(f"nudge_schedule failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def nudge_get_due(params: NudgeGetDueInput) -> dict:
    """Get all nudges that are due to be sent (scheduled_at <= now, not sent, not cancelled)."""
    now = params.now
    from datetime import datetime
    from services.database import get_db, NudgeQueue, is_db_healthy
    from sqlalchemy import select, and_
    if not is_db_healthy():
        return {"status": "error", "error": "DB not available"}
    try:
        current = datetime.fromisoformat(now) if now else datetime.utcnow()
        async with get_db() as db:
            rows = (await db.execute(
                select(NudgeQueue).where(and_(
                    NudgeQueue.scheduled_at <= current,
                    NudgeQueue.sent_at.is_(None),
                    NudgeQueue.cancelled == False,
                )).order_by(NudgeQueue.scheduled_at)
                .limit(50)
            )).scalars().all()
            return {
                "status": "success",
                "nudges": [{
                    "id": str(r.id),
                    "session_id": str(r.session_id),
                    "volunteer_phone": r.volunteer_phone,
                    "volunteer_name": r.volunteer_name,
                    "nudge_number": r.nudge_number,
                    "scheduled_at": r.scheduled_at.isoformat(),
                } for r in rows],
            }
    except Exception as e:
        logger.error(f"nudge_get_due failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def nudge_mark_sent(params: NudgeMarkSentInput) -> dict:
    """Mark a nudge as sent."""
    nudge_id = params.nudge_id
    from datetime import datetime
    from services.database import get_db, NudgeQueue, is_db_healthy
    from sqlalchemy import update as sa_update
    from uuid import UUID
    if not is_db_healthy():
        return {"status": "error", "error": "DB not available"}
    try:
        async with get_db() as db:
            await db.execute(
                sa_update(NudgeQueue)
                .where(NudgeQueue.id == UUID(nudge_id))
                .values(sent_at=datetime.utcnow())
            )
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def nudge_cancel_for_session(params: NudgeCancelForSessionInput) -> dict:
    """Cancel all pending nudges for a session (volunteer replied)."""
    session_id = params.session_id
    from services.database import get_db, NudgeQueue, is_db_healthy
    from sqlalchemy import update as sa_update, and_
    from uuid import UUID
    if not is_db_healthy():
        return {"status": "error", "error": "DB not available"}
    try:
        async with get_db() as db:
            result = await db.execute(
                sa_update(NudgeQueue)
                .where(and_(
                    NudgeQueue.session_id == UUID(session_id),
                    NudgeQueue.sent_at.is_(None),
                    NudgeQueue.cancelled == False,
                ))
                .values(cancelled=True)
            )
        return {"status": "success", "cancelled_count": result.rowcount}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def nudge_set_do_not_disturb(params: NudgeSetDoNotDisturbInput) -> dict:
    """Persist a volunteer's nudge opt-out on their session.

    Sets sessions.do_not_disturb so the nudge scheduler skips this
    session in future cycles (the flag is returned by list_sessions).
    """
    session_id = params.session_id
    from services.database import get_db, Session as DBSession, is_db_healthy
    from sqlalchemy import update as sa_update
    from uuid import UUID
    if not is_db_healthy():
        return {"status": "error", "error": "DB not available"}
    try:
        async with get_db() as db:
            result = await db.execute(
                sa_update(DBSession)
                .where(DBSession.id == UUID(session_id))
                .values(do_not_disturb=True)
            )
        return {"status": "success", "updated": result.rowcount}
    except Exception as e:
        logger.error(f"nudge_set_do_not_disturb failed: {e}")
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HTTP ENDPOINTS
# Must be registered BEFORE mcp.run() is called.
# ─────────────────────────────────────────────────────────────────────────────

from starlette.requests import Request as _Request
from starlette.responses import JSONResponse as _JSONResponse

from config import DASHBOARD_API_KEY
from services.dashboard_service import get_dashboard_stats, get_conversation_for_session, get_session_detail, get_analytics
from services.auth import validate_token


async def _check_dashboard_auth(request: _Request) -> bool:
    """
    Validate access to dashboard endpoints.
    Accepts:
    1. X-Admin-Token header matching DASHBOARD_API_KEY or ADMIN_TOKEN env var
    2. Bearer token matching DASHBOARD_API_KEY (legacy)
    3. JWT with coordinator/admin roles (Keycloak, if available)
    """
    import os
    admin_token = os.environ.get("ADMIN_TOKEN", "")

    # Check X-Admin-Token header (new simple auth)
    x_admin_token = request.headers.get("X-Admin-Token", "")
    if x_admin_token:
        if (DASHBOARD_API_KEY and x_admin_token == DASHBOARD_API_KEY):
            return True
        if (admin_token and x_admin_token == admin_token):
            return True

    # Check Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

        # Static API key check
        if DASHBOARD_API_KEY and token == DASHBOARD_API_KEY:
            return True
        if admin_token and token == admin_token:
            return True

        # JWT validation (Keycloak)
        try:
            claims = await validate_token(auth_header)
            allowed_roles = {"vCoordinator", "nCoordinator", "sAdmin", "nAdmin", "vAdmin"}
            if any(role in claims.roles for role in allowed_roles):
                return True
        except Exception:
            pass

    return False


@mcp.custom_route("/api/dashboard/stats", methods=["GET"])
async def dashboard_stats(request: _Request) -> _JSONResponse:
    if not await _check_dashboard_auth(request):
        return _JSONResponse({"error": "Unauthorized"}, status_code=401)
    page = int(request.query_params.get("page", 1))
    page_size = int(request.query_params.get("page_size", 25))
    data = await get_dashboard_stats(page=page, page_size=page_size)
    return _JSONResponse(data)


@mcp.custom_route("/api/dashboard/analytics", methods=["GET"])
async def dashboard_analytics(request: _Request) -> _JSONResponse:
    if not await _check_dashboard_auth(request):
        return _JSONResponse({"error": "Unauthorized"}, status_code=401)
    data = await get_analytics()
    return _JSONResponse(data)


@mcp.custom_route("/api/dashboard/conversation/{session_id}", methods=["GET"])
async def dashboard_conversation(request: _Request) -> _JSONResponse:
    if not await _check_dashboard_auth(request):
        return _JSONResponse({"error": "Unauthorized"}, status_code=401)
    session_id = request.path_params.get("session_id", "")
    limit = int(request.query_params.get("limit", 50))
    data = await get_conversation_for_session(session_id, limit)
    return _JSONResponse(data)


@mcp.custom_route("/api/dashboard/session/{session_id}", methods=["GET"])
async def dashboard_session_detail(request: _Request) -> _JSONResponse:
    if not await _check_dashboard_auth(request):
        return _JSONResponse({"error": "Unauthorized"}, status_code=401)
    session_id = request.path_params.get("session_id", "")
    data = await get_session_detail(session_id)
    return _JSONResponse(data)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    transport = "sse" if "--http" in sys.argv else "stdio"
    logger.info(f"Starting SERVE AI MCP Server (transport={transport}, port={MCP_PORT})")

    # The FastMCP SSE transport does NOT invoke the `lifespan` passed to the
    # FastMCP() constructor (confirmed with mcp 1.28.1) — so the _lifespan hook
    # above never fires and init_db() (create_all + column migrations) would
    # never run, leaving the schema uncreated on a fresh deploy. Run it
    # explicitly here so the DB is always initialised regardless of transport,
    # then dispose the engine so its connection pool is rebuilt inside
    # mcp.run()'s own event loop (avoids asyncpg "attached to a different loop"
    # errors). init_db is idempotent, so if a future mcp version does fire the
    # lifespan, the double-run is harmless.
    async def _startup_init():
        from services.database import init_db, engine
        try:
            await init_db()
        except Exception as e:
            logger.warning(f"init_db at startup failed (in-memory fallback active): {e}")
        finally:
            await engine.dispose()
    asyncio.run(_startup_init())

    mcp.run(transport=transport)
