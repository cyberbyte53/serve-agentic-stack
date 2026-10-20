"""
SERVE Orchestrator — Contextual Nudge Messages (pure functions, no I/O)

Builds the WhatsApp nudge text for an inactive volunteer from their
session context (workflow / active_agent / stage / persona), so nudges
reference where the volunteer actually left off instead of sending the
same generic copy to everyone (issue #25).

Deterministic templates on purpose — like the delivery agent's policy
engine, reminder copy is never an LLM decision. Personalisation is
limited to an optional first name.

Message hooks are keyed by the responsible agent (AgentType, per
WORKFLOW_REGISTRY) — no new bucket vocabulary is invented:
  ONBOARDING        — dropped mid-onboarding; references the step left at
  SELECTION /
  ENGAGEMENT        — finished initial steps, not yet matched
  FULFILLMENT       — matched, awaiting first session
  DELIVERY_ASSISTANT — in the delivery journey with a class coming up
Plus two non-agent cases, defined once below as constants:
  TAUGHT_BEFORE     — has completed sessions before, now inactive again
  GENERIC           — fallback: the original pre-#25 copy, unchanged
"""
from typing import Dict, Optional, Union

# Non-agent hook keys (agent-keyed hooks use AgentType members directly).
TAUGHT_BEFORE = "taught_before"
GENERIC = "generic"

# ── Original generic copy (fallback, unchanged) ──────────────────────────────

GENERIC_MESSAGES = {
    1: (
        "Hi! We had a brief technical issue on our end — apologies for the interruption. "
        "Your progress is fully saved. Just reply here to continue right where you left off. 🙏\n\n"
        "Reply:\n"
        "▶️ *continue* — pick up where you left off\n"
        "⏰ *later* — remind me in a few hours\n"
        "🚫 *stop* — don't send reminders"
    ),
    2: (
        "Hey! Just a quick check-in — your progress is saved and we're ready "
        "whenever you are. Reply anytime to continue, or 'stop' for no more reminders."
    ),
    3: (
        "Hi! We're still here whenever you're ready. Your progress is saved — "
        "just message anytime. This is our last reminder. 🙂"
    ),
}

# ── Stage knowledge comes from the orchestrator's workflow registry ────────
# WORKFLOW_REGISTRY (workflow_validator.py) is the source of truth for
# which stages exist and which agent is responsible for each. We only
# add volunteer-friendly phrasing on top — registry display names are
# admin labels (e.g. "Legacy Teaching Profile") unfit for a message.

from app.schemas import AgentType, OnboardingState, WorkflowType
from app.service.workflow_validator import (
    NEW_VOLUNTEER_ONBOARDING_WORKFLOW,
    WORKFLOW_REGISTRY,
)


def _parse_onboarding_state(context: Dict) -> Optional[OnboardingState]:
    """Session stage string -> OnboardingState, or None if not onboarding."""
    try:
        return OnboardingState(context.get("stage") or "")
    except ValueError:
        return None


# Onboarding-owned stages of the new-volunteer workflow, derived from
# the registry so this can never drift from the real workflow.
_ONBOARDING_STAGES = {
    state
    for stage_id, stage in NEW_VOLUNTEER_ONBOARDING_WORKFLOW.stages.items()
    if stage.responsible_agent == AgentType.ONBOARDING.value
    for state in [_parse_onboarding_state({"stage": stage_id})]
    if state is not None
} - {OnboardingState.PAUSED, OnboardingState.HUMAN_REVIEW}

# Volunteer-friendly phrase for each onboarding stage, keyed by the
# OnboardingState enum (never raw stage strings). Keys must cover
# _ONBOARDING_STAGES exactly — a test enforces this against the registry.
_ONBOARDING_STEP_PHRASES = {
    OnboardingState.WELCOME: "getting started",
    OnboardingState.ORIENTATION_VIDEO: "watching the short orientation video",
    OnboardingState.ELIGIBILITY_SCREENING: "a few quick details about your background",
    OnboardingState.CONTACT_CAPTURE: "sharing your contact details",
    OnboardingState.TEACHING_PROFILE: "telling us the subjects and classes you'd like to teach",
    OnboardingState.REGISTRATION_REVIEW: "reviewing your profile before registration",
}


def _responsible_agent(context: Dict) -> Optional[AgentType]:
    """Agent responsible for the session's current stage, per the registry.

    Falls back to the session's active_agent when the workflow/stage is
    unknown to the registry (e.g. the volunteer_engagement workflow,
    which the router knows but the validator registry does not).
    Returns None for agents outside AgentType (e.g. helpline).
    """
    raw = None
    workflow = WORKFLOW_REGISTRY.get(context.get("workflow") or "")
    if workflow is not None:
        stage = workflow.stages.get(context.get("stage") or "")
        if stage is not None:
            raw = stage.responsible_agent
    if raw is None:
        raw = context.get("active_agent") or ""
    try:
        return AgentType(raw)
    except ValueError:
        return None


# ── Bucket resolution ────────────────────────────────────────────────────────

def _first_name(context: Dict) -> Optional[str]:
    name = (
        context.get("volunteer_name")
        or context.get("first_name")
        or context.get("full_name")
        or (context.get("channel_metadata") or {}).get("volunteer_name")
    )
    if name and isinstance(name, str) and name.strip():
        return name.strip().split()[0]
    return None


def _sessions_completed(context: Dict) -> int:
    for key in ("sessions_completed", "classes_taught", "completed_sessions"):
        value = context.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def resolve_bucket(context: Dict) -> Union[AgentType, str]:
    """Map session context to a hook key.

    Returns the AgentType responsible for the current stage (per
    WORKFLOW_REGISTRY), or TAUGHT_BEFORE / GENERIC for the non-agent
    cases. Taught-before takes precedence for engagement-stage
    volunteers who have already taught.
    """
    workflow = context.get("workflow") or ""
    agent = _responsible_agent(context)

    if workflow == WorkflowType.DELIVERY_SUPPORT.value or agent is AgentType.DELIVERY_ASSISTANT:
        return AgentType.DELIVERY_ASSISTANT
    if _sessions_completed(context) > 0:
        return TAUGHT_BEFORE
    if agent in (
        AgentType.ONBOARDING,
        AgentType.SELECTION,
        AgentType.ENGAGEMENT,
        AgentType.FULFILLMENT,
        AgentType.DELIVERY_ASSISTANT,
    ):
        return agent
    if _parse_onboarding_state(context) in _ONBOARDING_STAGES:
        return AgentType.ONBOARDING
    return GENERIC


def _hook(bucket: Union[AgentType, str], context: Dict) -> str:
    """One sentence referencing where the volunteer left off."""
    if bucket is AgentType.ONBOARDING:
        step = _ONBOARDING_STEP_PHRASES.get(_parse_onboarding_state(context), "getting started")
        return f"You had started your volunteer onboarding with SERVE and left it at {step}."
    if bucket in (AgentType.SELECTION, AgentType.ENGAGEMENT):
        return "You had shown interest in teaching with SERVE and completed your initial steps."
    if bucket is AgentType.FULFILLMENT:
        return "You've been matched with a teaching opportunity with SERVE, and we're holding your place."
    if bucket == TAUGHT_BEFORE:
        return "You taught with SERVE before — you showed up, and it mattered to your students."
    if bucket is AgentType.DELIVERY_ASSISTANT:
        return "You have a class coming up with SERVE, and we want to make sure you're all set for it."
    return ""


# ── Public API ───────────────────────────────────────────────────────────────

def build_nudge_message(context: Dict, nudge_number: int) -> str:
    """
    Compose the nudge text for a session context and nudge number (1-3).

    Nudge numbers outside 1-3 are clamped. Unknown contexts fall back to
    the original generic copy so behaviour never regresses.
    """
    number = min(3, max(1, int(nudge_number)))
    bucket = resolve_bucket(context or {})
    if bucket == GENERIC:
        return GENERIC_MESSAGES[number]

    name = _first_name(context)
    greeting = f"Hi {name}!" if name else "Hi!"
    hook = _hook(bucket, context)

    if number == 1:
        return (
            f"{greeting} {hook} Just checking in — would you like to continue? "
            "Reply here and we'll pick up right where you left off. 🙏\n\n"
            "Reply:\n"
            "▶️ *continue* — pick up where you left off\n"
            "⏰ *later* — remind me in a few hours\n"
            "🚫 *stop* — don't send reminders"
        )
    if number == 2:
        return (
            f"{greeting} {hook} Your progress is fully saved and we're ready "
            "whenever you are. Reply anytime to continue, or 'stop' for no more reminders."
        )
    return (
        f"{greeting} {hook} This is our last reminder — just message anytime "
        "you're ready and we'll be here. 🙂"
    )
