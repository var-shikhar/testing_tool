# # memory_stdio_server.py  —  Python 3.12+
# """
# A pure-stdio MCP server:

# * Reads **one JSON-RPC 2.0 object per line** from STDIN.
# * Writes **one JSON-RPC 2.0 object per line** to STDOUT.
# * Lets mcp-proxy translate everything to HTTP + SSE.

# Run bare (for tests)       : python -u memory_stdio_server.py
# Run behind open-mcp-proxy  : mcp-proxy --sse-port 65432 -- python -u memory_stdio_server.py
# """
# # import aiohttp
# from __future__ import annotations
# import sys, json, logging, asyncio
# from dataclasses import dataclass, field
# from typing import Any, Dict, Callable

# # ─────────────────────────────  logging to STDERR  ───────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  [%(levelname)s]  %(message)s",
#     stream=sys.stderr,              # <-- NEVER write logs to STDOUT
# )

# # ─────────────────────────────  in-memory store  ─────────────────────────────
# STORE: dict[str, Any] = {}
# STORE_LOCK = asyncio.Lock()

# # ─────────────────────────────  helper I/O  ──────────────────────────────────
# def write(obj: dict) -> None:
#     """Emit one JSON-RPC object to STDOUT (newline-delimited)."""
#     sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
#     sys.stdout.flush()


# from tools import API_TOOLS, API_TOOL_IMPLS

# # Define all available tools
# AVAILABLE_TOOLS = {"tools": API_TOOLS}

# print("availaible tools",AVAILABLE_TOOLS)

# #Map tool-name → call-able  (all return either result or raise Exception)
# TOOL_IMPLS: Dict[str, Callable[..., Any]] = API_TOOL_IMPLS.copy()


# # ─── Add just below the imports ─────────────────────────────────────────────
# def make_result(text: str | dict | list, *, error: bool = False,headers: dict = None) -> dict:
#     """
#     Wrap raw tool output in the MCP-required structure:

#         {
#           "content": [{ "type": "text", "text": <stringified text> }],
#           "isError": <bool>
#         }
#     """
#     # if isinstance(text, (dict, list)):
#     #     text = json.dumps(text, separators=(",", ":"))
#     # return {
#     #     "content": [{"type": "text", "text": str(text)}],
#     #     "isError": error,
#     # }
#     if isinstance(text, (dict, list)):
#         text = json.dumps(text, separators=(",", ":"))
    
#     result = {
#         "content": [{"type": "text", "text": str(text)}],
#         "isError": error,
#     }
    
#     # If headers are provided, include them in the result
#     if headers:
#         result["headers"] = headers
#     logging.info("result1",result)
#     return result
# # ────────────────────────────────────────────────────────────────────────────
# # ─────────────────────────────  JSON-RPC handler  ────────────────────────────
# @dataclass
# class JsonRpc:
#     jsonrpc: str = "2.0"
#     id: int | str | None = None
#     method: str | None = None
#     params: dict[str, Any] = field(default_factory=dict)
#     result: Any | None = None
#     error: Any | None = None

# async def handle_rpc(msg: dict):
#     if msg.get("jsonrpc") != "2.0" or "id" not in msg:
#         logging.warning("Skipping invalid JSON-RPC: %s", msg)
#         return
#     logging.info(f"message: {msg}")
#     rpc = JsonRpc(**msg)         # type: ignore[arg-type]
#     try:
#         # 0️⃣  Hand-shake --------------------------------------------------
#         if rpc.method == "initialize":
#             rpc.result = {
#                 "protocolVersion": "2024-11-05",
#                 "capabilities": {
#                     "tools": {
#                         "list": True,
#                         "call": True
#                     }
#                 },
#                 "serverInfo": {
#                     "name":    "memory-mcp",
#                     "version": "0.1.0"
#                 }
#             }

#         # 1️⃣  List tools --------------------------------------------------
#         elif rpc.method == "tools/list":
#             rpc.result = AVAILABLE_TOOLS

#         # 2️⃣  Call a tool -------------------------------------------------
#         elif rpc.method == "tools/call":
#             # params = {"name": "...", "arguments": {...}}
#             tool_name = rpc.params["name"]
#             kwargs    = rpc.params.get("arguments", {})
#             contractor_token = rpc.params["contract_token"]
#             # contractor_token = kwargs.pop("x-contractor-token", None)
#             # contractor_token = kwargs.pop("x-contractor-token", None) or (
#             #     kwargs.get("headers", {}).get("x-contractor-token")
#             # )
#             # if not kwargs.get('headers'):
#             #     kwargs['headers'] = {}
#             # if contractor_token:
#             #     kwargs['headers']['x-contractor-token'] = contractor_token
            
#             contractor_token = kwargs.pop("contractor_token", None) \
#                 or kwargs.pop("x-contractor-token", None) \
#                 or kwargs.get("headers", {}).get("x-contractor-token")

#             # Don't auto-create headers anymore
#             if contractor_token:
#                 kwargs["contractor_token"] = contractor_token

            
#             # kwargs['contractor_token'] = CONTRACTOR_TOKEN
#             impl = TOOL_IMPLS.get(tool_name)
#             # logging.info("implementation",impl)
#             if not impl:
#                 raise ValueError(f"Unknown tool {tool_name}")

#             # result-data = await impl(**kwargs)
#             rpc.result = make_result(await impl(**kwargs))               

#         # -----------------------------------------------------------------
#         else:
#             raise ValueError(f"Unknown method {rpc.method}")

#     except Exception as exc:      # -> structured JSON-RPC error
#         logging.exception("call failed")
#         rpc.error = {"code": -32000, "message": str(exc)}
#         rpc.result = make_result(str(exc), error=True)

#     if rpc.error is not None:
#         write({"jsonrpc": "2.0", "id": rpc.id, "error": rpc.error})
#     else:
#         write({"jsonrpc": "2.0", "id": rpc.id, "result": rpc.result})



# # ─────────────────────────────  main event-loop  ─────────────────────────────
# async def main():
#     loop = asyncio.get_running_loop()
#     reader = asyncio.StreamReader()
#     protocol = asyncio.StreamReaderProtocol(reader)
#     await loop.connect_read_pipe(lambda: protocol, sys.stdin)

#     while True:
#         line = await reader.readline()
#         if not line:
#             break                # EOF → proxy shut down

#         try:
#             msg = json.loads(line)
#             await handle_rpc(msg)
#         except json.JSONDecodeError:
#             logging.warning("Non-JSON input ignored: %s", line.rstrip())

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         pass


# memory_stdio_server.py  —  Python 3.12+
"""
A pure-stdio MCP server:

* Reads **one JSON-RPC 2.0 object per line** from STDIN.
* Writes **one JSON-RPC 2.0 object per line** to STDOUT.
* Lets mcp-proxy translate everything to HTTP + SSE.

Run bare (for tests)       : python -u memory_stdio_server.py
Run behind open-mcp-proxy  : mcp-proxy --sse-port 65432 -- python -u memory_stdio_server.py
"""
# import aiohttp
from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

# ─────────────────────────────  logging to STDERR  ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    stream=sys.stderr,  # <-- NEVER write logs to STDOUT
)

# ─────────────────────────────  in-memory store  ─────────────────────────────
STORE: dict[str, Any] = {}
STORE_LOCK = asyncio.Lock()


# ─────────────────────────────  helper I/O  ──────────────────────────────────
def write(obj: dict) -> None:
    """Emit one JSON-RPC object to STDOUT (newline-delimited)."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()



# from contracting_plus.tools import API_TOOL_IMPLS, API_TOOLS
from ..tools import API_TOOLS, API_TOOL_IMPLS

# Define all available tools
AVAILABLE_TOOLS = {"tools": API_TOOLS}

print("availaible tools", AVAILABLE_TOOLS)

# Map tool-name → call-able  (all return either result or raise Exception)
TOOL_IMPLS: Dict[str, Callable[..., Any]] = API_TOOL_IMPLS.copy()


# ─── Add just below the imports ─────────────────────────────────────────────
def make_result(text: str | dict | list, *, error: bool = False) -> dict:
    """
    Wrap raw tool output in the MCP-required structure:

        {
          "content": [{ "type": "text", "text": <stringified text> }],
          "isError": <bool>
        }
    """
    if isinstance(text, (dict, list)):
        text = json.dumps(text, separators=(",", ":"))
    return {
        "content": [{"type": "text", "text": str(text)}],
        "isError": error,
    }


# ────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────  JSON-RPC handler  ────────────────────────────
@dataclass
class JsonRpc:
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    error: Any | None = None


async def handle_rpc(msg: dict):
    if msg.get("jsonrpc") != "2.0" or "id" not in msg:
        logging.warning("Skipping invalid JSON-RPC: %s", msg)
        return

    rpc = JsonRpc(**msg)  # type: ignore[arg-type]
    try:
        # 0️⃣  Hand-shake --------------------------------------------------
        if rpc.method == "initialize":
            rpc.result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"list": True, "call": True}},
                "serverInfo": {"name": "memory-mcp", "version": "0.1.0"},
            }

        # 1️⃣  List tools --------------------------------------------------
        elif rpc.method == "tools/list":
            rpc.result = AVAILABLE_TOOLS

        # 2️⃣  Call a tool -------------------------------------------------
        elif rpc.method == "tools/call":
            # params = {"name": "...", "arguments": {...}}
            tool_name = rpc.params["name"]
            kwargs = rpc.params.get("arguments", {})

            impl = TOOL_IMPLS.get(tool_name)
            if not impl:
                raise ValueError(f"Unknown tool {tool_name}")

            # result-data = await impl(**kwargs)
            rpc.result = make_result(await impl(**kwargs))

        # -----------------------------------------------------------------
        else:
            raise ValueError(f"Unknown method {rpc.method}")

    except Exception as exc:  # -> structured JSON-RPC error
        logging.exception("call failed")
        rpc.error = {"code": -32000, "message": str(exc)}
        rpc.result = make_result(str(exc), error=True)

    if rpc.error is not None:
        write({"jsonrpc": "2.0", "id": rpc.id, "error": rpc.error})
    else:
        write({"jsonrpc": "2.0", "id": rpc.id, "result": rpc.result})


# ─────────────────────────────  main event-loop  ─────────────────────────────
async def main():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break  # EOF → proxy shut down

        try:
            msg = json.loads(line)
            await handle_rpc(msg)
        except json.JSONDecodeError:
            logging.warning("Non-JSON input ignored: %s", line.rstrip())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
