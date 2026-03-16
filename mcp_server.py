"""
ChainLink MCP Server
====================
Model Context Protocol server that exposes ChainLink memory tools
directly to AI assistants (Claude Code, Cursor, Codex, etc).

Install:
    pip install chainlink-memory[mcp]

Run:
    chainlink-mcp

Add to Claude Code config (~/.claude/claude_desktop_config.json):
    {
        "mcpServers": {
            "chainlink": {
                "command": "chainlink-mcp",
                "env": {"ANTHROPIC_API_KEY": "sk-ant-..."}
            }
        }
    }

This gives AI assistants direct access to:
- store_memory: Save a memory
- store_memories: Save multiple memories at once
- query_memory: Find relevant memories with chain reasoning
- list_memories: See all stored memories
- remove_memory: Delete a memory by ID
- clear_memories: Remove all memories
- memory_stats: Get memory count and status
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from typing import Any

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

# Lazy-load ChainLink to avoid slow import on startup
_chainlink_instance = None
_persist_path = None


def _get_chainlink():
    """Get or create the ChainLink instance."""
    global _chainlink_instance, _persist_path

    if _chainlink_instance is None:
        # Import here to avoid slow startup from sentence-transformers
        sys.path.insert(0, str(Path(__file__).parent))
        from sdk import ChainLink

        # Default persist path: ~/.chainlink/memories.json
        _persist_path = os.environ.get(
            "CHAINLINK_PERSIST_PATH",
            str(Path.home() / ".chainlink" / "memories.json"),
        )

        # Ensure directory exists
        Path(_persist_path).parent.mkdir(parents=True, exist_ok=True)

        _chainlink_instance = ChainLink(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            persist_path=_persist_path,
            model=os.environ.get("CHAINLINK_MODEL", "claude-haiku-4-5-20251001"),
        )

    return _chainlink_instance


# Create the MCP server
server = Server("chainlink-memory")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose ChainLink tools to AI assistants."""
    return [
        Tool(
            name="store_memory",
            description=(
                "Store a memory in ChainLink. ChainLink finds implicit connections "
                "between memories that vector search misses \u2014 like connecting "
                "'Thai dinner' \u2192 'shrimp paste' \u2192 'shellfish allergy'. "
                "Use this to store any fact, preference, event, or context the user shares."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memory text to store (e.g. 'User has a severe shellfish allergy')",
                    },
                    "metadata": {
                        "type": "object",
                        "description": 'Optional key-value metadata (e.g. {"source": "chat", "topic": "health"})',
                        "additionalProperties": true,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="store_memories",
            description=(
                "Store multiple memories at once in ChainLink. More efficient than "
                "calling store_memory repeatedly. Use when onboarding a user or "
                "importing context from another source."
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
        ),        Tool(
            name="query_memory",
            description=(
                "Search memories with chain reasoning. Unlike vector search, this finds "
                "INDIRECT connections through multi-hop reasoning. For example, querying "
                "'plan Friday dinner' will find 'shellfish allergy' even if the user "
                "never mentioned allergies and dinner together \u2014 because ChainLink traces "
                "the chain: Thai dinner \u2192 shrimp paste \u2192 shellfish allergy. "
                "96.9% chain recall vs 68.8% for vector search alone."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g. 'anything to worry about for Friday?')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_memories",
            description="List all stored memories with their IDs, text, and metadata.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
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
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="memory_stats",
            description="Get statistics about stored memories: count, persist path, model info.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls from AI assistants."""

    chainlink = _get_chainlink()

    if name == "store_memory":
        text = arguments["text"]
        metadata = arguments.get("metadata")
        memory_id = chainlink.add(text, metadata=metadata)
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "stored",
                "memory_id": memory_id,
                "text": text,
                "total_memories": chainlink.count(),
            }),
        )]

    elif name == "store_memories":
        texts = arguments["texts"]
        ids = chainlink.add_many(texts)
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "stored",
                "memory_ids": ids,
                "count": len(ids),
                "total_memories": chainlink.count(),
            }),
        )]

    elif name == "query_memory":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        # Run in thread pool since chain reasoning is CPU/IO bound
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: chainlink.query(query, top_k=top_k)
        )

        return [TextContent(
            type="text",
            text=json.dumps({
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
            }),
        )]

    elif name == "list_memories":
        memories = chainlink.get_all()
        return [TextContent(
            type="text",
            text=json.dumps({
                "memories": [
                    {
                        "id": m.id,
                        "text": m.text,
                        "metadata": m.metadata,
                    }
                    for m in memories
                ],
                "count": len(memories),
            }),
        )]

    elif name == "remove_memory":
        memory_id = arguments["memory_id"]
        removed = chainlink.remove(memory_id)
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "removed" if removed else "not_found",
                "memory_id": memory_id,
                "total_memories": chainlink.count(),
            }),
        )]

    elif name == "clear_memories":
        chainlink.clear()
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "cleared",
                "total_memories": 0,
            }),
        )]

    elif name == "memory_stats":
        return [TextContent(
            type="text",
            text=json.dumps({
                "total_memories": chainlink.count(),
                "persist_path": str(_persist_path),
                "model": chainlink._engine.model,
            }),
        )]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}),
        )]


async def main():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


def cli_entry():
    """CLI entry point for `chainlink-mcp` command."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_entry()

