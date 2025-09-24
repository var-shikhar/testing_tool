# browser_stdio_server.py  —  Python 3.12+
"""
A pure-stdio MCP server for browser automation.

* Reads **one JSON-RPC 2.0 object per line** from STDIN.
* Writes **one JSON-RPC 2.0 object per line** to STDOUT.
* Lets mcp-proxy translate everything to HTTP + SSE.

Run bare (for tests)       : python -u browser_stdio_server.py
Run behind open-mcp-proxy  : mcp-proxy --sse-port 65433 -- python -u browser_stdio_server.py
"""
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

# ─────────────────────────────  browser setup  ─────────────────────────────
from playwright.async_api import async_playwright
from typing import Optional, Any

browser_instance: Optional[Any] = None
page_instance: Optional[Any] = None

async def init_browser():
    global browser_instance, page_instance
    playwright = await async_playwright().start()
    browser_instance = await playwright.chromium.launch(headless=False)
    page_instance = await browser_instance.new_page()

async def close_browser():
    if page_instance:
        await page_instance.close()
    if browser_instance:
        await browser_instance.close()

# ─────────────────────────────  helper I/O  ──────────────────────────────────
def write(obj: dict) -> None:
    """Emit one JSON-RPC object to STDOUT (newline-delimited)."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()

# ─────────────────────────────  browser tools  ─────────────────────────────
async def navigate(url: str) -> str:
    if not page_instance:
        await init_browser()
    assert page_instance is not None
    await page_instance.goto(url)
    return f"Navigated to {url}"

async def click(selector: str) -> str:
    if not page_instance:
        await init_browser()
    assert page_instance is not None
    await page_instance.click(selector)
    return f"Clicked on {selector}"

async def type_text(selector: str, text: str) -> str:
    if not page_instance:
        await init_browser()
    assert page_instance is not None
    await page_instance.fill(selector, text)
    return f"Typed '{text}' into {selector}"

async def screenshot() -> str:
    if not page_instance:
        await init_browser()
    assert page_instance is not None
    await page_instance.screenshot(path="screenshot.png")
    return "Screenshot saved as screenshot.png"

async def get_text(selector: str) -> str:
    if not page_instance:
        await init_browser()
    assert page_instance is not None
    element = await page_instance.query_selector(selector)
    if element:
        text = await element.inner_text()
        return text
    return "Element not found"

async def upload_file(selector: str, file_path: str) -> str:
    """
    Upload a file to a file input element.
    Note: This will trigger the file upload dialog for manual selection.
    The actual file upload happens through user interaction.
    """
    if not page_instance:
        await init_browser()
    assert page_instance is not None

    # Check if the selector exists and is a file input
    element = await page_instance.query_selector(selector)
    if not element:
        return f"Element not found: {selector}"

    # Check if it's a file input
    input_type = await element.get_attribute('type')
    if input_type != 'file':
        return f"Element is not a file input (type: {input_type})"

    # Set up file chooser handler
    file_chooser = None

    def handle_file_chooser(chooser):
        nonlocal file_chooser
        file_chooser = chooser

    page_instance.on('filechooser', handle_file_chooser)

    try:
        # Click the file input to open the file dialog
        await element.click()

        # Wait for file chooser to appear (this requires manual intervention)
        await asyncio.sleep(0.5)  # Give time for dialog to open

        if file_chooser:
            # If we have a file chooser, we could set files programmatically
            # But for manual intervention, we'll just acknowledge the dialog opened
            return f"File upload dialog opened for selector: {selector}. Please manually select and upload the file: {file_path}"
        else:
            return f"File input clicked for selector: {selector}, but no file chooser detected. Manual file selection required."

    except Exception as e:
        return f"Error triggering file upload: {e}"
    finally:
        page_instance.remove_listener('filechooser', handle_file_chooser)

# Define all available tools
AVAILABLE_TOOLS = {
    "tools": [
        {
            "name": "navigate",
            "description": "Navigate to a URL",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to"}
                },
                "required": ["url"]
            }
        },
        {
            "name": "click",
            "description": "Click on an element by selector",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the element"}
                },
                "required": ["selector"]
            }
        },
        {
            "name": "type_text",
            "description": "Type text into an input field",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the input"},
                    "text": {"type": "string", "description": "Text to type"}
                },
                "required": ["selector", "text"]
            }
        },
        {
            "name": "screenshot",
            "description": "Take a screenshot of the current page",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_text",
            "description": "Get text from an element",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the element"}
                },
                "required": ["selector"]
            }
        },
        {
            "name": "upload_file",
            "description": "Trigger file upload dialog for a file input element (requires manual file selection)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the file input element"},
                    "file_path": {"type": "string", "description": "Path to the file (for reference, manual selection required)"}
                },
                "required": ["selector", "file_path"]
            }
        }
    ]
}

# Map tool-name → callable
TOOL_IMPLS: Dict[str, Callable[..., Any]] = {
    "navigate": navigate,
    "click": click,
    "type_text": type_text,
    "screenshot": screenshot,
    "get_text": get_text,
    "upload_file": upload_file,
}

print("Available browser tools", AVAILABLE_TOOLS)

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
                "serverInfo": {"name": "browser-mcp", "version": "0.1.0"},
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
    await init_browser()
    try:
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
    finally:
        await close_browser()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass