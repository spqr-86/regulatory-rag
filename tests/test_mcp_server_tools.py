"""Регистрация тулов MCP-сервера.

Ловит смену API SDK: на mcp 2.x модуль `mcp.server.fastmcp` удалён, а
`FastMCP` переименован в `MCPServer`. Импорт `mcp_server` и список тулов
падают при такой смене раньше, чем это увидит клиент.
"""

import asyncio

import pytest

EXPECTED_TOOLS = {"retrieve_chunks", "get_chunk", "get_norm", "collection_info"}


@pytest.mark.unit
def test_server_exposes_expected_tools():
    import mcp_server

    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


@pytest.mark.unit
def test_server_name():
    import mcp_server

    assert mcp_server.mcp.name == "regulatory-mcp"
