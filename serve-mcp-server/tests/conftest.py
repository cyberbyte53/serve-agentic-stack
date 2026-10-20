"""Shared fixtures for serve-mcp-server tests."""
import sys
from pathlib import Path

# Ensure the service root is on sys.path so `services.*` and `config` imports work
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
