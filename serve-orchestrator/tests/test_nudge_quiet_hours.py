"""Boundary tests for _is_quiet_hour (real IST via zoneinfo).

The previous implementation used `(utcnow().hour + 5) % 24  # rough IST`,
dropping IST's :30 — so quiet hours actually began/ended up to 30 minutes
late. These tests pin the exact boundaries, including that regression.

IST wall-clock times are converted to aware UTC datetimes, exercising the
same conversion path production uses.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.service import nudge_scheduler as ns


def _at_ist(hour: int, minute: int = 0) -> datetime:
    ist_wall = datetime(2026, 10, 20, hour, minute, tzinfo=ns.IST)
    return ist_wall.astimezone(timezone.utc)


@pytest.mark.parametrize("hour,minute,expected", [
    (20, 59, False),  # just before quiet hours start
    (21, 0, True),    # quiet hours start (9pm IST)
    (21, 10, True),   # :30 regression — old "+5" math said False until 21:40
    (23, 59, True),
    (0, 30, True),    # window wraps past midnight
    (7, 59, True),
    (8, 0, False),    # quiet hours end (8am IST)
    (8, 10, False),   # :30 regression — old math stayed "quiet" until 08:40
    (12, 0, False),   # midday
])
def test_quiet_hour_boundaries_ist(hour, minute, expected):
    assert ns._is_quiet_hour(at=_at_ist(hour, minute)) is expected


def test_naive_input_is_treated_as_utc():
    # 15:30 UTC == 21:00 IST (quiet-hours start)
    assert ns._is_quiet_hour(at=datetime(2026, 10, 20, 15, 30)) is True
    # 02:30 UTC == 08:00 IST (quiet-hours end)
    assert ns._is_quiet_hour(at=datetime(2026, 10, 20, 2, 30)) is False


def test_non_wrapping_window(monkeypatch):
    monkeypatch.setattr(ns, "NUDGE_QUIET_HOURS_START", 1)
    monkeypatch.setattr(ns, "NUDGE_QUIET_HOURS_END", 5)
    assert ns._is_quiet_hour(at=_at_ist(3)) is True
    assert ns._is_quiet_hour(at=_at_ist(12)) is False


def test_equal_start_end_disables_quiet_hours(monkeypatch):
    # START == END is the documented off-switch (used in local E2E runs).
    monkeypatch.setattr(ns, "NUDGE_QUIET_HOURS_START", 0)
    monkeypatch.setattr(ns, "NUDGE_QUIET_HOURS_END", 0)
    for hour in (0, 3, 12, 22):
        assert ns._is_quiet_hour(at=_at_ist(hour)) is False


def test_default_argument_uses_current_time():
    # Smoke: no-arg call must agree with evaluating "now" explicitly.
    now_utc = datetime.now(timezone.utc)
    expected = ns._is_quiet_hour(at=now_utc)
    # Allow one boundary race if the hour rolled over between calls.
    assert ns._is_quiet_hour() in (expected, ns._is_quiet_hour(at=now_utc + timedelta(minutes=1)))
