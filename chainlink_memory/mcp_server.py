"""
ChainLink MCP Server
====================
Model Context Protocol server that exposes ChainLink memory tools
directly to AI assistants (Claude Code, Cursor, Windsurf, etc).

Install:
    pip install chainlink-memory[mcp]

Run:
    chainlink-mcp

Configure in Claude Code (~/.claude.json or project .mcp.json):
    {
        "mcpServers": {
            "chainlink-memory": {
                "command": "chainlink-mcp",
                "env": {"ANTHROPIC_API_KEY": "sk-ant-..."}
            }
        }
    }

Tools exposed:
- store_memory: Save a fact/preference/context
- store_memories: Batch store multiple memories
- query_memory: Search with chain reasoning (finds connections vector search misses)
- list_memories: See all stored memories
- remove_memory: Delete a memory by ID
- clear_memories: Remove all memories
- memory_stats: Get memory count and status
"""

import json
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("chainlink_memory.mcp")

# Lazy-load ChainLink to keep startup fast (sentence-transformers is slow)
_chainlink_instance = None
_persist_path = None


def _get_chainlink():
    """Get or create the ChainLink instance. Lazy-loaded on first tool call."""
    global _chainlink_instance, _persist_path

    if _chainlink_instance is None:
        from chainlink_memory.sdk import ChainLink

        _persist_path = os.environ.get(
            "CHAINLINK_PERSIST_PATH",
            str(Path.home() / ".chainlink" / "memories.json"),
        )

        Path(_persist_path).parent.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = os.environ.get("CHAINLINK_MODEL", "claude-haiku-4-5-20251001")

        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set. store_memory will work but "
                "query_memory chain reasoning will fail. Set the key in your "
                "MCP config env or export it in your shell."
            )

        _chainlink_instance = ChainLink(
            api_key=api_key if api_key else None,
            persist_path=_persist_path,
            model=model,
        )

    return _chainlink_instance


def _make_response(data: dict) -> list:
    """Build an MCP text content response from a dict."""
    from mcp.types import TextContent
    return [TextContent(type="text", text=json.dumps(data, default=str))]


def _error_response(message: str) -> list:
    """Build an MCP error response."""
    from mcp.types import TextContent
    return [TextContent(type="text", text=json.dumps({"error": message}))]


def create_server():
    """Create and configure the MCP server with all tools."""
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("chainlink-memory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="store_memory",
                description=(
                    "Store a memory in ChainLink. Memories are persisted to disk "
                    "and used for chain reasoning queries. Store any fact, preference, "
                    "event, observation, or context the user shares. Examples: "
                    "'User has a severe shellfish allergy', 'Meeting with Bob at 3pm Thursday', "
                    "'Project deadline moved to March 20'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory text to store",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata tags",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="store_memories",
                description=(
                    "Store multiple memories at once. More efficient than "
                    "calling store_memory repeatedly."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of memory texts to store",
                        },
                    },
                    "required": ["texts"],
                },
            ),
            Tool(
                name="query_memory",
                description=(
                    "Search memories using chain reasoning. Unlike simple vector search, "
                    "this finds INDIRECT connections through multi-hop reasoning chains. "
                    "Example: querying 'plan Friday dinner' finds 'shellfish allergy' even "
                    "if dinner and allergies were never mentioned together — because ChainLink "
                    "traces the chain: Thai dinner -> shrimp paste -> shellfish. "
                    "Use this whenever you need to recall relevant context about a user, "
                    "project, or situation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_memories",
                description="List all stored memories with their IDs and text.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="remove_memory",
                description="Remove a specific memory by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "The ID of the memory to remove",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
            Tool(
                name="clear_memories",
                description="Remove ALL stored memories. Use with caution.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="memory_stats",
                description="Get memory count, persist path, and model info.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            chainlink = _get_chainlink()

            if name == "store_memory":
                text = arguments.get("text", "").strip()
                if not text:
                    return _error_response("text is required and cannot be empty")
                metadata = arguments.get("metadata")
                memory_id = chainlink.add(text, metadata=metadata)
                return _make_response({
                    "status": "stored",
                    "memory_id": memory_id,
                    "text": text,
                    "total_memories": chainlink.count(),
                })

            elif name == "store_memories":
                texts = arguments.get("texts", [])
                texts = [t.strip() for t in texts if t.strip()]
                if not texts:
                    return _error_response("texts must be a non-empty list of strings")
                ids = chainlink.add_many(texts)
                return _make_response({
                    "status": "stored",
                    "memory_ids": ids,
                    "count": len(ids),
                    "total_memories": chainlink.count(),
                })

            elif name == "query_memory":
                query = arguments.get("query", "").strip()
                if not query:
                    return _error_response("query is required and cannot be empty")
                top_k = arguments.get("top_k", 5)

                if chainlink.count() == 0:
                    return _make_response({
                        "query": query,
                        "results": [],
                        "count": 0,
                        "message": "No memories stored yet. Use store_memory first.",
                    })

                # Run chain reasoning in a thread (CPU/IO bound)
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, lambda: chainlink.query(query, top_k=top_k)
                )

                return _make_response({
                    "query": query,
                    "results": [
                        {
                            "text": r.text,
                            "score": r.score,
                            "reason": r.reason,
                            "source": r.source,
                            "is_chain": r.is_chain,
                        }
                        for r in results
                    ],
                    "count": len(results),
                })

            elif name == "list_memories":
                memories = chainlink.get_all()
                return _make_response({
                    "memories": [
                        {"id": m.id, "text": m.text, "metadata": m.metadata}
                        for m in memories
                    ],
                    "count": len(memories),
                })

            elif name == "remove_memory":
                memory_id = arguments.get("memory_id")
                if memory_id is None:
                    return _error_response("memory_id is required")
                removed = chainlink.remove(int(memory_id))
                return _make_response({
                    "status": "removed" if removed else "not_found",
                    "memory_id": memory_id,
                    "total_memories": chainlink.count(),
                })

            elif name == "clear_memories":
                chainlink.clear()
                return _make_response({
                    "status": "cleared",
                    "total_memories": 0,
                })

            elif name == "memory_stats":
                return _make_response({
                    "total_memories": chainlink.count(),
                    "persist_path": str(_persist_path),
                    "model": chainlink._engine.model,
                    "api_key_set": bool(chainlink._engine._api_key),
                })

            else:
                return _error_response(f"Unknown tool: {name}")

        except ValueError as e:
            return _error_response(str(e))
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return _error_response(f"Internal error: {type(e).__name__}: {e}")

    return server


async def main():
    """Run the MCP server over stdio."""
    from mcp.server.stdio import stdio_server

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


def cli_entry():
    """CLI entry point for `chainlink-mcp` command."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_entry()
