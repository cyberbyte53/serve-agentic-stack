"""Test bootstrap for serve-orchestrator tests.

Puts the orchestrator package and the MCP server package on sys.path, and
stubs `services.database` (which creates a real Postgres engine on import)
so `services.session_service` can be imported without a database or the
asyncpg driver installed.
"""
import sys
import types
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_DIR = TESTS_DIR.parent
REPO_ROOT = ORCHESTRATOR_DIR.parent
MCP_DIR = REPO_ROOT / "serve-mcp-server"

for p in (str(ORCHESTRATOR_DIR), str(MCP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _stub_mcp_database() -> None:
    """Replace services.database with inert stand-ins before it is imported."""
    if "services.database" in sys.modules:
        return

    # Package stub with a real __path__ so submodule imports (e.g.
    # services.session_service) load from disk without executing the real
    # services/__init__.py (which pulls in every service + the DB engine).
    pkg = types.ModuleType("services")
    pkg.__path__ = [str(MCP_DIR / "services")]
    sys.modules.setdefault("services", pkg)

    db = types.ModuleType("services.database")
    for name in ("AgentHandoffLog", "ConversationMessage", "Session", "TelemetryEvent"):
        setattr(db, name, type(name, (), {}))
    db.get_db = lambda *a, **k: None
    db.is_db_healthy = lambda: False
    sys.modules["services.database"] = db


_stub_mcp_database()
