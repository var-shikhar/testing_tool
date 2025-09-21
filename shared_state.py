import asyncio

# Store client-specific asyncio.Queues for SSE
client_queues: dict[str, asyncio.Queue] = {}

