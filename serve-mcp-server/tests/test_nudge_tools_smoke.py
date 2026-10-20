"""Execution smoke tests for the nudge MCP tools (issue #25, Bug 4b).

The transport contract tests pin the argument shape; these pin that
the tools actually execute. nudge_schedule and nudge_mark_sent
previously referenced `datetime` without importing it, so every call
returned {"status": "error", "error": "name 'datetime' is not
defined"} — a failure invisible until something really called them.

A fake async DB session stands in for Postgres (the tools import
get_db from services.database at call time, so patching that module
is enough).
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest

import main
from schemas import NudgeMarkSentInput, NudgeScheduleInput
import services.database as database


class FakeDB:
    def __init__(self):
        self.added = []
        self.executed = []

    async def execute(self, statement):
        self.executed.append(statement)
        result = SimpleNamespace()
        result.scalars = lambda: SimpleNamespace(first=lambda: None)
        result.rowcount = 1
        return result

    def add(self, obj):
        self.added.append(obj)


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDB()

    @asynccontextmanager
    async def fake_get_db():
        yield db

    monkeypatch.setattr(database, "get_db", fake_get_db)
    monkeypatch.setattr(database, "is_db_healthy", lambda: True)
    return db


def test_nudge_schedule_executes(fake_db):
    result = asyncio.run(main.nudge_schedule(NudgeScheduleInput(
        session_id="11111111-1111-1111-1111-111111111111",
        volunteer_phone="919999999999",
        nudge_number=1,
        scheduled_at="2026-10-20T10:00:00",
        volunteer_name="Asha",
    )))
    assert result["status"] == "success", result
    # scheduled_at must have been parsed to a datetime (the NameError site)
    assert fake_db.added, "no NudgeQueue row was added"
    assert isinstance(fake_db.added[0].scheduled_at, datetime)


def test_nudge_mark_sent_executes(fake_db):
    result = asyncio.run(main.nudge_mark_sent(NudgeMarkSentInput(
        nudge_id="22222222-2222-2222-2222-222222222222",
    )))
    assert result["status"] == "success", result
    assert fake_db.executed, "no UPDATE was executed"
