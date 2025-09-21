# mcp_handler.py
import asyncio
import json
import logging
import copy
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Tuple, Optional

from mcp.client.sse import sse_client
from mcp import ClientSession # Assuming this is the correct import path for ClientSession

logger = logging.getLogger(__name__)

# Default timeout values, can be overridden via environment variables
DEFAULT_MCP_TOOL_FETCH_TIMEOUT = 5.0
DEFAULT_MCP_TOOL_EXEC_TIMEOUT = 30.0
DEFAULT_MCP_CONNECTION_TIMEOUT = 5.0

def _fix_openapi_schema_recursive(schema_node: Any, path: str = "") -> Any:
    """
    Recursively fixes common issues in OpenAPI schemas,
    such as ensuring 'items' is defined for arrays.
    """
    if not isinstance(schema_node, dict):
        return schema_node
    
    if schema_node.get("type") == "array":
        if schema_node.get("items") is None or \
           (isinstance(schema_node.get("items"), dict) and not schema_node.get("items")):
            logger.debug(f"Schema Fix: Default 'items' for array at '{path}'. Setting to {{'type': 'string'}}")
            schema_node["items"] = {"type": "string"} # Default to string items if not specified

    for key, value in schema_node.items():
        new_path = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            _fix_openapi_schema_recursive(value, new_path)
        elif isinstance(value, list):
            for i, item_in_list in enumerate(value):
                if isinstance(item_in_list, dict):
                    _fix_openapi_schema_recursive(item_in_list, f"{new_path}[{i}]")
    return schema_node

class MCPManager:
    """
    Manages connections, tool discovery, and tool execution for multiple MCP instances.
    """
    def __init__(self,
                 mcp_endpoints: Dict[str, str],
                 tool_fetch_timeout: float = DEFAULT_MCP_TOOL_FETCH_TIMEOUT,
                 tool_exec_timeout: float = DEFAULT_MCP_TOOL_EXEC_TIMEOUT,
                 connection_timeout: float = DEFAULT_MCP_CONNECTION_TIMEOUT):
        self.mcp_endpoints_config = mcp_endpoints
        self.tool_fetch_timeout = tool_fetch_timeout
        self.tool_exec_timeout = tool_exec_timeout
        self.connection_timeout = connection_timeout

        self._sessions: Dict[str, ClientSession] = {}
        self._managed_resources: Dict[str, List[Any]] = {} 
        self._all_openai_tools: List[Dict] = []
        self._stack: Optional[AsyncExitStack] = None
        self._is_active = False # Tracks if the manager is within its async context

    async def __aenter__(self):
        """Enters the async context, initializes AsyncExitStack and performs initial connections."""
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        self._is_active = True
        logger.info("MCPManager entered, performing initial connection and tool fetch.")
        if self.mcp_endpoints_config: # Only attempt if endpoints are configured
            await self.refresh_connections_and_tools()
        else:
            logger.info("MCPManager: No MCP endpoints configured. Skipping initial connection.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exits the async context, ensuring all MCP resources are cleaned up."""
        logger.info("MCPManager exiting, cleaning up all MCP resources.")
        self._is_active = False
        if self._stack:
            for category in list(self._sessions.keys()): # Make a copy for safe iteration
                await self._cleanup_session_resources(category)
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
            self._stack = None
        self._sessions.clear()
        self._managed_resources.clear()
        self._all_openai_tools.clear()
        logger.info("MCPManager exited and resources cleaned up.")

    async def _establish_session_resources(self, category: str, url: str) -> Tuple[Optional[ClientSession], List[Any]]:
        """Establishes a session with a single MCP and manages its resources via the stack."""
        if not self._stack:
            logger.error("AsyncExitStack not available. Cannot establish session resources.")
            return None, []

        logger.debug(f"Attempting to establish MCP session for '{category}' at {url} (timeout: {self.connection_timeout}s)...")
        entered_cms_for_this_session = []
        session_object = None
        try:
            async def connect_sequence():
                nonlocal session_object
                sse_conn_mgr = sse_client(url)
                # mypy can sometimes struggle with enter_async_context, type: ignore might be needed depending on environment
                rx_tx = await self._stack.enter_async_context(sse_conn_mgr) 
                entered_cms_for_this_session.append(sse_conn_mgr)
                
                session_mgr_cm = ClientSession(rx_tx[0], rx_tx[1])
                sess_obj_inner = await self._stack.enter_async_context(session_mgr_cm)
                entered_cms_for_this_session.append(session_mgr_cm)
                
                await sess_obj_inner.initialize()
                session_object = sess_obj_inner

            await asyncio.wait_for(connect_sequence(), timeout=self.connection_timeout)
            logger.info(f"MCP session for '{category}' established and initialized.")
            return session_object, entered_cms_for_this_session
        except asyncio.TimeoutError:
            logger.error(f"Timeout establishing MCP session for '{category}' at {url}.")
        except ConnectionRefusedError:
            logger.error(f"Connection refused for MCP '{category}' at {url}.")
        except Exception as e:
            logger.error(f"Failed to connect/initialize MCP '{category}' at {url}: {e}", exc_info=False)
        
        # Cleanup resources if establishment failed
        logger.debug(f"Cleaning up {len(entered_cms_for_this_session)} resources from failed attempt for '{category}'.")
        for cm_to_pop in reversed(entered_cms_for_this_session):
            try:
                await self._stack.pop_aclose(cm_to_pop)
            except Exception as pop_exc:
                logger.error(f"Error cleaning up resource {type(cm_to_pop).__name__} for '{category}': {pop_exc}")
        return None, []

    async def _cleanup_session_resources(self, category: str):
        """Cleans up resources for a specific MCP category."""
        if not self._stack:
            logger.warning(f"AsyncExitStack not available for cleaning '{category}'. Resources might leak.")
            if category in self._sessions: del self._sessions[category]
            if category in self._managed_resources: del self._managed_resources[category]
            return

        logger.info(f"Cleaning up resources for MCP '{category}'.")
        if category in self._managed_resources:
            for cm in reversed(self._managed_resources[category]):
                try:
                    logger.debug(f"Closing CM: {type(cm).__name__} for {category}")
                    await self._stack.pop_aclose(cm)
                except Exception as e:
                    logger.error(f"Error closing resource {type(cm).__name__} for '{category}': {e}", exc_info=True)
            del self._managed_resources[category]
        if category in self._sessions:
            del self._sessions[category]
        logger.info(f"Finished cleaning up MCP '{category}'.")

    async def _fetch_tools_for_session(self, session: ClientSession, category_prefix: str) -> Tuple[List[Any], List[Dict]]:
        """Fetches and formats tools from a single active MCP session."""
        logger.info(f"Fetching tools from MCP '{category_prefix}' (timeout: {self.tool_fetch_timeout}s)...")
        try:
            resp = await asyncio.wait_for(session.list_tools(), timeout=self.tool_fetch_timeout)
            logger.debug(f"Raw response from list_tools for '{category_prefix}': {resp}")
            
            mcp_native_tools = []
            oa_tools_for_this_mcp = []

            if not hasattr(resp, 'tools') or not isinstance(resp.tools, list):
                logger.warning(f"No 'tools' list found or in unexpected format in response from '{category_prefix}'. Resp: {resp}")
                return [], []

            for t in resp.tools:
                if hasattr(t, 'name') and hasattr(t, 'inputSchema'):
                    prefixed_tool_name = f"{category_prefix}_{t.name}"
                    
                    # Ensure inputSchema is a dict, default if not
                    original_parameters_schema = t.inputSchema
                    if not isinstance(original_parameters_schema, dict):
                        logger.warning(f"Tool '{prefixed_tool_name}' has non-dict inputSchema '{original_parameters_schema}'. Defaulting to empty schema.")
                        parameters_schema = {"type": "object", "properties": {}}
                    else:
                        parameters_schema = copy.deepcopy(original_parameters_schema)

                    parameters_schema = _fix_openapi_schema_recursive(parameters_schema, prefixed_tool_name)
                    
                    # Ensure basic structure for OpenAI
                    if "type" not in parameters_schema:
                        parameters_schema["type"] = "object"
                    if parameters_schema["type"] == "object" and "properties" not in parameters_schema:
                        parameters_schema["properties"] = {}
                    
                    oa_tools_for_this_mcp.append({
                        "type": "function",
                        "function": {
                            "name": prefixed_tool_name,
                            "description": f"({category_prefix}) {t.description or 'Tool from '+category_prefix}",
                            "parameters": parameters_schema,
                        },
                    })
                    mcp_native_tools.append(t)
                else:
                    logger.warning(f"Skipping tool from '{category_prefix}' due to missing 'name' or 'inputSchema': {t}")
            return mcp_native_tools, oa_tools_for_this_mcp
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching tools from MCP '{category_prefix}'.")
            raise
        except Exception as e:
            logger.error(f"Failed to get tools from MCP '{category_prefix}': {e}", exc_info=True)
            raise

    async def refresh_connections_and_tools(self) -> None:
        """
        Refreshes all MCP connections and updates the list of available OpenAI tools.
        Manages connections based on current `mcp_endpoints_config`.
        """
        if not self._is_active:
            logger.warning("MCPManager is not active. Cannot refresh connections.")
            return
        if not self.mcp_endpoints_config:
            logger.info("No MCP endpoints configured. Clearing existing sessions and tools if any.")
            for category_to_remove in list(self._sessions.keys()):
                await self._cleanup_session_resources(category_to_remove)
            self._all_openai_tools = []
            return

        logger.info("--- Refreshing MCP connections and tools ---")
        
        current_config = self.mcp_endpoints_config.copy()
        
        # Remove sessions for endpoints that are no longer configured
        for category in list(self._sessions.keys()):
            if category not in current_config:
                logger.info(f"MCP '{category}' removed from configuration. Cleaning up.")
                await self._cleanup_session_resources(category)

        # Establish new sessions or re-establish failed ones
        for category, url in current_config.items():
            if category not in self._sessions: # Or add a health check: or not await self._is_session_healthy(category)
                logger.info(f"MCP '{category}' not connected or needs reconnection. Attempting...")
                # Defensive cleanup if resources somehow exist for a non-session category
                if category in self._managed_resources:
                     await self._cleanup_session_resources(category)

                new_session, entered_cms = await self._establish_session_resources(category, url)
                if new_session:
                    self._sessions[category] = new_session
                    self._managed_resources[category] = entered_cms
                # If new_session is None, _establish_session_resources already logged the error and cleaned up partial resources.
        
        # Fetch tools from all currently active sessions
        refreshed_oa_tools: List[Dict] = []
        tool_fetch_tasks = []
        active_categories = list(self._sessions.keys()) # iterate over a copy

        for category in active_categories:
            if category in self._sessions: # Check again, session might have been removed by another task
                session = self._sessions[category]
                # Create task to fetch tools for this session
                task = asyncio.create_task(
                    self._fetch_tools_for_session(session, category),
                    name=f"fetch_tools_{category}"
                )
                tool_fetch_tasks.append((category, task))
            
        for category, task in tool_fetch_tasks:
            try:
                _, oa_tools_for_mcp = await task # native_tools are not used further for now
                if oa_tools_for_mcp:
                    refreshed_oa_tools.extend(oa_tools_for_mcp)
                    logger.debug(f"Successfully fetched {len(oa_tools_for_mcp)} tools for '{category}'.")
            except Exception: # Errors are logged in _fetch_tools_for_session
                logger.error(f"Failed to fetch tools for MCP '{category}' during refresh. It will be unavailable.")
                # This session might be problematic, clean it up so it's retried next refresh
                await self._cleanup_session_resources(category)
        
        self._all_openai_tools = refreshed_oa_tools
        logger.info(f"MCP refresh complete. Active sessions: {list(self._sessions.keys())}. Total OpenAI tools: {len(self._all_openai_tools)}.")

    async def execute_tool(self, tool_call: Any) -> Dict[str, Any]: # tool_call: openai.types.chat.ChatCompletionMessageToolCall
        """Executes a tool call on the appropriate MCP."""
        prefixed_tool_name = tool_call.function.name
        tool_call_id = tool_call.id
        logger.info(f"Attempting tool call: {prefixed_tool_name} (ID: {tool_call_id}, Timeout: {self.tool_exec_timeout}s)")

        try:
            category, original_tool_name = prefixed_tool_name.split("_", 1)
        except ValueError:
            logger.error(f"Invalid tool name format: {prefixed_tool_name}. Cannot extract category.")
            return {"role": "tool", "tool_call_id": tool_call_id, "name": prefixed_tool_name,
                    "content": f"[TOOL ERROR] Invalid tool name format: {prefixed_tool_name}"}

        if category not in self._sessions:
            logger.error(f"No active MCP session for category '{category}' (tool: {prefixed_tool_name}).")
            # Optionally, one could attempt an on-demand refresh here, but it might complicate flow.
            # For now, rely on periodic refreshes.
            return {"role": "tool", "tool_call_id": tool_call_id, "name": prefixed_tool_name,
                    "content": f"[TOOL ERROR] MCP category '{category}' is not available or disconnected."}

        session = self._sessions[category]
        tool_args_str = tool_call.function.arguments or "{}"
        tool_args: Dict[str, Any] = {}
        
        try:
            tool_args = json.loads(tool_args_str)
            logger.info(f"Executing tool '{original_tool_name}' on MCP '{category}' with args: {tool_args}")

            # Original code had a commented-out pre-hover logic. If needed, re-implement here.
            # e.g., if category == "browser" and original_tool_name in {...}: await session.call_tool("hover", ...)

            tool_exec_result = await asyncio.wait_for(
                session.call_tool(original_tool_name, tool_args),
                timeout=self.tool_exec_timeout
            )
            
            # Assuming tool_exec_result has a .dict() method or is directly serializable
            if hasattr(tool_exec_result, 'dict') and callable(tool_exec_result.dict):
                tool_output_obj = tool_exec_result.dict()
            else: # Fallback if no .dict() method
                tool_output_obj = tool_exec_result 
            
            tool_output_str = json.dumps(tool_output_obj, ensure_ascii=False, default=str) # default=str for non-serializable objects
            
            # Specific logging for certain tools, e.g., browser snapshot
            if category == 'browser' and original_tool_name == 'snapshot':
                logger.info(f"--- Browser Snapshot Received (length: {len(tool_output_str)}) ---")
                # Potentially truncate very long snapshots for general logging if SNAPSHOT_LOG_MAX_LEN is used
                # from original code, SNAPSHOT_LOG_MAX_LEN was defined but not used in this part.
            
            content = tool_output_str

        except asyncio.TimeoutError:
            logger.error(f"Timeout executing tool '{prefixed_tool_name}' on MCP '{category}'.")
            content = f"[TOOL ERROR] Execution timeout for {prefixed_tool_name} after {self.tool_exec_timeout}s"
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON argument parsing error for tool '{prefixed_tool_name}': {json_err}. Args provided: '{tool_args_str}'")
            content = f"[TOOL ERROR] Invalid JSON arguments for {prefixed_tool_name}: {json_err}. Arguments: {tool_args_str}"
        except AttributeError as attr_err: # E.g. if tool_exec_result.dict() fails or result is unexpected
            logger.error(f"Attribute error processing result for tool '{prefixed_tool_name}': {attr_err}", exc_info=True)
            content = f"[TOOL ERROR] Error processing result for {prefixed_tool_name}: {attr_err}"
        except Exception as tool_exc: # Catch-all for other execution errors
            logger.error(f"Unexpected error executing tool '{prefixed_tool_name}': {tool_exc}", exc_info=True)
            # Consider if certain errors (e.g., connection closed) should lead to session cleanup
            content = f"[TOOL ERROR] Execution failed for {prefixed_tool_name}: {tool_exc}"
        
        return {"role": "tool", "tool_call_id": tool_call_id, "name": prefixed_tool_name, "content": content}

    @property
    def active_categories(self) -> List[str]:
        """Returns a list of categories for currently active MCP sessions."""
        return list(self._sessions.keys())

    @property
    def openai_tools(self) -> List[Dict]:
        """Returns a copy of the current list of OpenAI-formatted tools from all active MCPs."""
        return self._all_openai_tools.copy()

    @property
    def is_ready(self) -> bool:
        """True if the manager is active, has configured endpoints, at least one session, and at least one tool."""
        return self._is_active and bool(self.mcp_endpoints_config) and bool(self._sessions) and bool(self._all_openai_tools)

    @property
    def has_any_connection(self) -> bool:
        """True if the manager is active and has at least one MCP session."""
        return self._is_active and bool(self._sessions)
        
    @property
    def has_endpoints_configured(self) -> bool:
        """True if MCP endpoints are configured for this manager instance."""
        return bool(self.mcp_endpoints_config)