"""Transport contract tests for the nudge MCP tools (issue #25, Bug 4).

Every MCP client in this repo (domain_client, the nudge scheduler)
calls tools with arguments wrapped as {"params": {...}}, matching the
params-model style used by all other tools. The nudge_* tools were
originally defined with flat arguments, so every scheduler call failed
pydantic validation ("Field required: session_id ...") and no nudge
was ever scheduled — invisible to unit tests that fake the transport.

These tests pin the contract: each nudge tool must expose a single
`params` input, like the rest of the server.
"""
import asyncio

import main

NUDGE_TOOLS = [
    "nudge_schedule",
    "nudge_get_due",
    "nudge_mark_sent",
    "nudge_cancel_for_session",
    "nudge_set_do_not_disturb",
]


def test_nudge_tools_accept_params_wrapped_arguments():
    tools = {t.name: t for t in asyncio.run(main.mcp.list_tools())}
    for name in NUDGE_TOOLS:
        assert name in tools, f"{name} not registered"
        properties = tools[name].inputSchema.get("properties", {})
        assert list(properties.keys()) == ["params"], (
            f"{name} takes {list(properties.keys())}, but repo MCP clients "
            "wrap arguments as {'params': {...}} — calls will fail validation"
        )
