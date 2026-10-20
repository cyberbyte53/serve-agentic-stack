"""
SERVE Orchestrator — Contextual Nudge Messages (pure functions, no I/O)

Builds the WhatsApp nudge text for an inactive volunteer from their
session context (workflow / active_agent / stage / persona), so nudges
reference where the volunteer actually left off instead of sending the
same generic copy to everyone (issue #25).

Deterministic templates on purpose — like the delivery agent's policy
engine, reminder copy is never an LLM decision. Personalisation is
limited to an optional first name.

Buckets:
  onboarding      — dropped mid-onboarding; references the step left at
  post_selection  — finished initial steps / in engagement, not yet matched
  matched         — with the fulfillment agent, awaiting first session
  taught_before   — has completed sessions before, now inactive again
  delivery        — in the delivery journey with a class to show up for
  generic         — fallback: the original pre-#25 copy, unchanged
"""
from typing import Dict, Optional

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

# ── Onboarding step phrases (stage -> what the volunteer was doing) ─────────

_ONBOARDING_STEPS = {
    "welcome": "getting started",
    "orientation_video": "watching the short orientation video",
    "intent_discovery": "telling us why you'd like to volunteer",
    "purpose_orientation": "learning how SERVE works",
    "eligibility_screening": "a few quick details about your background",
    "eligibility_confirmation": "a few quick details about your background",
    "contact_capture": "sharing your contact details",
    "capability_discovery": "telling us the subjects you'd like to teach",
    "teaching_profile": "telling us the subjects and classes you'd like to teach",
    "registration_review": "reviewing your profile before registration",
    "profile_confirmation": "reviewing your profile before registration",
}

_POST_SELECTION_STAGES = {
    "onboarding_complete", "selection_conversation", "gathering_preferences",
    "re_engaging", "profile_refresh", "matching_ready", "verifying_identity",
    "not_registered",
}

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


def resolve_bucket(context: Dict) -> str:
    """Map session context to a message bucket."""
    workflow = context.get("workflow") or ""
    agent = context.get("active_agent") or ""
    stage = context.get("stage") or ""

    if workflow == "delivery_support" or agent in ("delivery_assistant", "delivery"):
        return "delivery"
    if _sessions_completed(context) > 0:
        return "taught_before"
    if agent == "fulfillment" or workflow == "fulfillment":
        return "matched"
    if (
        agent in ("engagement", "selection")
        or workflow in ("returning_volunteer", "volunteer_engagement", "recommended_volunteer")
        or stage in _POST_SELECTION_STAGES
    ):
        return "post_selection"
    if workflow == "new_volunteer_onboarding" or agent == "onboarding" or stage in _ONBOARDING_STEPS:
        return "onboarding"
    return "generic"


def _hook(bucket: str, context: Dict) -> str:
    """One sentence referencing where the volunteer left off."""
    if bucket == "onboarding":
        step = _ONBOARDING_STEPS.get(context.get("stage") or "", "getting started")
        return f"You had started your volunteer onboarding with SERVE and left it at {step}."
    if bucket == "post_selection":
        return "You had shown interest in teaching with SERVE and completed your initial steps."
    if bucket == "matched":
        return "You've been matched with a teaching opportunity with SERVE, and we're holding your place."
    if bucket == "taught_before":
        return "You taught with SERVE before — you showed up, and it mattered to your students."
    if bucket == "delivery":
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
    if bucket == "generic":
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
