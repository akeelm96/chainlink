"""
ChainLink Memory — Find connections your vector search misses.

Chain-aware memory for AI agents. One import, three methods.

Quick start:
    from chainlink_memory import ChainLink

    memory = ChainLink()
    memory.add("User loves Thai food")
    memory.add("User has a severe shellfish allergy")
    memory.add("Thai curries often contain shrimp paste")

    results = memory.query("plan Friday dinner")
    # Finds: Thai food -> shrimp paste -> shellfish allergy chain
"""

from chainlink_memory.sdk import ChainLink, QueryResult, Memory

__version__ = "0.1.0"
__all__ = ["ChainLink", "QueryResult", "Memory"]
