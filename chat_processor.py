"""
V1.0

this is the working versoin of the testing file. This works well.
22 aug 25
    - Handeled reconnecting to MCP servers by calling initialize_mcp_manager(). First time when called it cleans up old instance of the same MCP server and next time it connects
"""

import os
import httpx # For ElevenLabs API call
import json
import asyncio
import openai # Ensure this is the new openai v1+ client
import logging
from typing import Any, Dict, List, AsyncGenerator, Optional
import uuid # For generating unique IDs for synthetic tool calls
from dotenv import load_dotenv
# New imports for MCPManager and OpenAI types
from mcp_client import MCPManager
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import Function
# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()


logger = logging.getLogger(__name__)

# --- Environment Variables & Constants ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") # Default if not set

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Already used by OpenAI client
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o4-mini")  # o4-mini  gpt-4.1-miniOr your preferred model for chat
BASE_URL = os.getenv("BASE_URL")
# TTS Configuration
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai").lower() #openai or elevenlabs
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1") # e.g., tts-1, tts-1-hd, or user-provided "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy") # e.g., alloy, echo, fable, onyx, nova, shimmer

MAX_TURNS = 10

# --- Constants previously in mcp_setup.py, now part of chat_processor configuration ---
# MCP_ENDPOINTS_CONFIG = {
#     # "browser": "http://localhost:8127/sse", # Example, add if browser MCP is used
#     # "danamojoDB": "http://localhost:7121/sse",
#     # "charts": "http://localhost:8129/sse",
# }

mcp_endpoints_json = os.getenv("MCP_ENDPOINTS_JSON")
MCP_ENDPOINTS_CONFIG: Dict[str, str] = {}
if mcp_endpoints_json:
    try:
        MCP_ENDPOINTS_CONFIG = json.loads(mcp_endpoints_json)
        logger.info(f"Loaded MCP_ENDPOINTS_CONFIG from .env: {MCP_ENDPOINTS_CONFIG}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse MCP_ENDPOINTS_JSON: {e}. MCP features will be unavailable.")
        MCP_ENDPOINTS_CONFIG = {} # Ensure it's an empty dict on failure
else:
    logger.warning("MCP_ENDPOINTS_JSON not found in .env. MCP features will be unavailable if MCPs are expected.")
    MCP_ENDPOINTS_CONFIG = {}



SYSTEM_PROMPT = """
You are an advanced application testing agent capable of controlling actions on browser, have access to memory and some other tools
Your goal is to fulfill perform the action by interacting with the appropriate tools.
Tools are namespaced by their category. For example:
- Browser tools start with `browser_` (e.g., `browser_snapshot`, `browser_click`).
- Memory tools (if available) might start with `memory_` (e.g., `memory_get_usage`).
Follow these steps:
1.  **Analyze the Request:** Understand what the user wants to achieve and which category of tools / Agent is most relevant.
2.  **Assess the Current State:** Assess the current state by looking at the current broser snapshot and based on that decide on next action.
3. This is running in autorun mode, so use your best judgement to proceed to next step, do not ask user for inputs.
4. Stick to the exact task that is assigned and act accordingly. Do not assume actions. 
5. Also use browser navigate only when explicitly asked to do so. 
6. Never navigate to example.com is similar generic domains.
"""

# SNAPSHOT_LOG_MAX_LEN = 2000
# ACTIONS_REQUIRING_ELEMENT_INTERACTION = {
#     "browser_click", "browser_hover", "browser_type", "browser_select_option",
# }

# --- Global MCPManager Instance ---
mcp_manager_instance: Optional[MCPManager] = None
# --- Global OpenAI Client Instance ---
# Re-use client for multiple calls (chat, TTS)
openai_client_instance: Optional[openai.AsyncOpenAI] = None

def get_openai_client() -> openai.AsyncOpenAI:
    global openai_client_instance
    if openai_client_instance is None:
        if not OPENAI_API_KEY:
            # This should ideally be checked earlier or handled more gracefully
            # For now, let it raise an error if used without a key.
            logger.error("OPENAI_API_KEY not found. OpenAI client cannot be initialized.")
            raise ValueError("OPENAI_API_KEY is not set.")
        openai_client_instance = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
    return openai_client_instance


async def initialize_mcp_manager():
    """Initializes the MCPManager. Call this at application startup."""
    global mcp_manager_instance
    
    if mcp_manager_instance is not None:
        logger.info("MCPManager already initialized. Attempting to refresh connections and tools.")
        try:
            await mcp_manager_instance.refresh_connections_and_tools()
            logger.info("MCPManager connections and tools refreshed successfully.")
        except Exception as e:
            logger.error(f"ERROR during MCPManager refresh: {e}", exc_info=True)
            # Decide if the manager should be considered unusable after a failed refresh
            # For now, it remains, but might be in a bad state.
        return

    # mcp_manager_instance is None, so initialize
    logger.info(f"Attempting to initialize MCPManager. Endpoints from .env: {MCP_ENDPOINTS_CONFIG}")
    
    # Ensure mcp_manager_instance is None before attempting to set it
    # This is more for clarity as the 'if' block handles this.
    mcp_manager_instance = None 
    
    try:
        # Create the manager object
        manager_obj = MCPManager(
            mcp_endpoints=MCP_ENDPOINTS_CONFIG
            # You can also pass custom timeouts here if needed, e.g.:
            # tool_fetch_timeout=float(os.getenv("MCP_TOOL_FETCH_TIMEOUT", 5.0)),
            # tool_exec_timeout=float(os.getenv("MCP_TOOL_EXEC_TIMEOUT", 30.0)),
            # connection_timeout=float(os.getenv("MCP_CONNECTION_TIMEOUT", 5.0))
        )
        logger.info("MCPManager object created. Attempting to enter async context (__aenter__)...")
        
        # Enter the async context. __aenter__ returns `self` (the manager instance).
        # This is where connections are established and initial tools are fetched.
        initialized_manager = await manager_obj.__aenter__() 
        
        if initialized_manager:
            mcp_manager_instance = initialized_manager # Assign to global variable
            logger.info("MCPManager context entered successfully. Instance assigned.")
            
            # Original readiness checks from your main.py startup_event
            if mcp_manager_instance.is_ready:
                logger.info(f"MCP Manager is ready. Active MCP categories: {mcp_manager_instance.active_categories}")
                logger.info(f"Total OpenAI tools available: {len(mcp_manager_instance.openai_tools)}")
            else:
                logger.warning("MCP Manager initialized but is not fully ready. Check detailed logs from MCPManager component.")
                if not mcp_manager_instance.has_endpoints_configured:
                     logger.warning("-> MCPManager Reason: No MCP endpoints were configured (MCP_ENDPOINTS_CONFIG was empty).")
                elif not mcp_manager_instance.has_any_connection:
                     logger.warning("-> MCPManager Reason: Failed to connect to any configured MCP instances.")
                elif not mcp_manager_instance.openai_tools: # Implies connection but no tools fetched
                     logger.warning("-> MCPManager Reason: Connected to MCP instances but found no tools from them.")
        else:
            # This case should ideally not happen if __aenter__ is correctly implemented
            # (it should return `self` or raise an exception).
            logger.error("MCPManager.__aenter__ returned None or a falsy value, which is unexpected. MCPManager instance will remain None.")
            mcp_manager_instance = None # Explicitly ensure it's None

    except Exception as e:
        logger.error(f"CRITICAL ERROR during MCPManager initialization or its __aenter__ call: {e}", exc_info=True)
        # mcp_manager_instance will remain None (or be re-set to None) due to the exception
        mcp_manager_instance = None 
    
    if mcp_manager_instance is None:
        logger.error("initialize_mcp_manager finished, but mcp_manager_instance is STILL None. Review critical errors above.")


async def shutdown_mcp_manager():
    """Shuts down the MCPManager. Call this at application shutdown."""
    global mcp_manager_instance
    if mcp_manager_instance:
        logger.info("Shutting down MCPManager...")
        try:
            await mcp_manager_instance.__aexit__(None, None, None) # Manually exit async context
            logger.info("MCPManager shut down and context exited successfully.")
        except Exception as e:
            logger.error(f"Error during MCPManager.__aexit__: {e}", exc_info=True)
        finally:
            mcp_manager_instance = None # Clear the instance
    else:
        logger.info("MCPManager instance was already None. No shutdown action needed.")


async def _text_to_speech_elevenlabs_internal(text: str) -> bytes | None:
    """Converts text to speech using ElevenLabs API and returns audio bytes."""
    if not ELEVENLABS_API_KEY:
        logger.error("TTS (ElevenLabs): API key not configured.")
        return None
    if not ELEVENLABS_VOICE_ID: # Check if voice ID is set
        logger.error("TTS (ElevenLabs): Voice ID not configured.")
        return None
    if not text:
        logger.warning("TTS (ElevenLabs): Empty text provided.")
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2", # Or your preferred model
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    logger.info(f"TTS (ElevenLabs): Requesting speech for text: \"{text[:50]}...\"")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=60.0)
            
            if response.status_code == 200:
                logger.info("TTS (ElevenLabs): Successfully received audio stream.")
                audio_bytes = await response.aread()
                return audio_bytes
            else:
                error_content = await response.atext()
                logger.error(f"TTS (ElevenLabs): API error. Status: {response.status_code}, Response: {error_content}")
                return None
    except httpx.RequestError as e:
        logger.error(f"TTS (ElevenLabs): HTTP request failed: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"TTS (ElevenLabs): Unexpected error: {e}", exc_info=True)
        return None

async def _text_to_speech_openai(text: str) -> bytes | None:
    """Converts text to speech using OpenAI API and returns audio bytes."""
    if not OPENAI_API_KEY: # Should be caught by get_openai_client, but good to check
        logger.error("TTS (OpenAI): API key not configured.")
        return None
    if not text:
        logger.warning("TTS (OpenAI): Empty text provided.")
        return None

    logger.info(f"TTS (OpenAI): Requesting speech using model '{OPENAI_TTS_MODEL}' and voice '{OPENAI_TTS_VOICE}' for text: \"{text[:50]}...\"")
    try:
        client = get_openai_client()
        response = await client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format="mp3" # Keep consistent with ElevenLabs mime type
        )
        audio_bytes = response.read() # Reads the binary content
        logger.info("TTS (OpenAI): Successfully received audio data.")
        return audio_bytes
    except openai.APIError as e:
        logger.error(f"TTS (OpenAI): API error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"TTS (OpenAI): Unexpected error: {e}", exc_info=True)
        return None

async def generate_text_to_speech(text: str) -> bytes | None:
    """
    Generates text to speech using the configured provider.
    """
    logger.info(f"Generating speech using TTS_PROVIDER: {TTS_PROVIDER}")
    if TTS_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logger.error("OpenAI TTS provider selected, but OPENAI_API_KEY is not set.")
            return None
        return await _text_to_speech_openai(text)
    elif TTS_PROVIDER == "elevenlabs":
        if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
            logger.error("ElevenLabs TTS provider selected, but ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID is not set.")
            return None
        return await _text_to_speech_elevenlabs_internal(text)
    else:
        logger.warning(f"Unsupported TTS_PROVIDER: '{TTS_PROVIDER}'. No TTS will be generated. Supported providers: 'openai', 'elevenlabs'.")
        return None


async def stream_chat_updates(type: str, data: Any, client_queue: asyncio.Queue):
    """Helper to send structured updates to the client's queue."""
    await client_queue.put({"type": type, "data": data})

# Add a getter function (optional but good practice for controlled access)
def get_mcp_manager() -> Optional[MCPManager]:
    return mcp_manager_instance


async def execute_tool_call(tool_call: ChatCompletionMessageToolCall, client_queue: asyncio.Queue) -> Dict[str, Any]:
    prefixed_tool_name = tool_call.function.name
    tool_call_id = tool_call.id # Original tool_call_id for the main response
    tool_args_str = tool_call.function.arguments or "{}"
    
    await stream_chat_updates("tool_attempt", {
        "name": prefixed_tool_name,
        "arguments": tool_args_str 
    }, client_queue)
    logger.info(f"Attempting Tool Call: {prefixed_tool_name} (ID: {tool_call_id}) with args: {tool_args_str}")

    if not mcp_manager_instance:
        logger.error("MCPManager not initialized. Cannot execute tool.")
        tool_output = "[TOOL ERROR] MCPManager service is not available."
        await stream_chat_updates("tool_result", {
            "id": tool_call_id, "name": prefixed_tool_name, "content": tool_output, "success": False
        }, client_queue)
        return {"role": "tool", "tool_call_id": tool_call_id, "name": prefixed_tool_name, "content": tool_output}

    tool_result_dict: Dict[str, Any]

    # Pre-hover logic for specific browser actions
    try:
        category, original_tool_name = prefixed_tool_name.split("_", 1)
        tool_args = json.loads(tool_args_str) # Parse once for potential pre-hover and main call
        mcp_id = tool_args.get("mcp_id")

        # if category == "browser" and prefixed_tool_name in ACTIONS_REQUIRING_ELEMENT_INTERACTION and \
        #    prefixed_tool_name != "browser_hover" and mcp_id:
            
        #     await stream_chat_updates("info", f"Pre-hovering for action '{prefixed_tool_name}' on mcp_id {mcp_id}...", client_queue)
            
        #     hover_tool_name = "browser_hover"
        #     hover_args_obj = {"mcp_id": mcp_id}
        #     hover_args_str_for_call = json.dumps(hover_args_obj)
            
        #     hover_tool_call_id = f"hover-subcall-{uuid.uuid4()}" 
        #     hover_function = Function(name=hover_tool_name, arguments=hover_args_str_for_call)
        #     synthetic_hover_tool_call = ChatCompletionMessageToolCall(
        #         id=hover_tool_call_id,
        #         function=hover_function,
        #         type="function"
        #     )
            
        #     logger.info(f"Executing synthetic pre-hover: {hover_tool_name} with args {hover_args_obj}")
        #     hover_result = await mcp_manager_instance.execute_tool(synthetic_hover_tool_call)
            
        #     if "[TOOL ERROR]" in hover_result.get("content", ""):
        #         logger.warning(f"Pre-hover for {mcp_id} (action: {prefixed_tool_name}) failed or had error: {hover_result.get('content')}.")
        #         await stream_chat_updates("warning", f"Pre-hover for {mcp_id} failed: {hover_result.get('content')}", client_queue)
        #     else:
        #         logger.info(f"Pre-hover for {mcp_id} (action: {prefixed_tool_name}) successful.")
        #         # Not sending a "tool_result" for this sub-call to avoid confusing client, 
        #         # but an "info" or "status" update could be sent if desired.
    
    except json.JSONDecodeError as json_err: # Error parsing arguments for pre-hover logic
        logger.error(f"Failed to parse JSON arguments for pre-hover logic related to {prefixed_tool_name}: {json_err}. Raw args: '{tool_args_str}'")
        # Proceed to main tool call, pre-hover skipped.
    except Exception as e: # Catch any other errors during pre-hover setup/execution
        logger.error(f"Error during pre-hover phase for {prefixed_tool_name}: {e}", exc_info=True)
        await stream_chat_updates("warning", f"Could not perform pre-hover for {prefixed_tool_name} due to: {str(e)}", client_queue)
        # Proceed to main tool call, pre-hover skipped or failed.

    # Execute the original tool call using MCPManager
    logger.info(f"Executing main tool call via MCPManager: {prefixed_tool_name} (ID: {tool_call_id})")
    tool_result_dict = await mcp_manager_instance.execute_tool(tool_call) # tool_call is the original one from OpenAI

    tool_output = tool_result_dict.get("content", "[TOOL ERROR] No content in MCP response")
    # MCPManager's execute_tool already formats errors with "[TOOL ERROR]"
    success = not tool_output.startswith("[TOOL ERROR]")

    if not success: # Log the full error content if it's an error
         logger.error(f"Tool '{prefixed_tool_name}' (ID: {tool_call_id}) execution resulted in error: {tool_output}")
    else:
         logger.info(f"Tool '{prefixed_tool_name}' (ID: {tool_call_id}) execution successful.")

    await stream_chat_updates("tool_result", {
        "id": tool_call_id, 
        "name": prefixed_tool_name, 
        "content": tool_output, 
        "success": success
    }, client_queue)

    # # Specific UI update for browser_snapshot content
    # if prefixed_tool_name == 'browser_snapshot' and success:
    #     log_content = tool_output[:SNAPSHOT_LOG_MAX_LEN] + ('...' if len(tool_output) > SNAPSHOT_LOG_MAX_LEN else '')
    #     await stream_chat_updates("info", f"Browser Snapshot (truncated for UI): {log_content}", client_queue)
    
    return tool_result_dict


async def process_user_request_for_web(user_input: str, client_queue: asyncio.Queue, original_input_was_voice: bool):
    logger.info(f"PROCESS_USER_REQUEST: Started for input='{user_input}', client_queue: {id(client_queue)}, original_input_was_voice: {original_input_was_voice}")
    # Refresh MCP connections check to see if new MSP servers have come up then reconnect 
    await initialize_mcp_manager()
    if not mcp_manager_instance or not mcp_manager_instance.is_ready:
        error_msg = "MCP services are currently unavailable or not fully configured."
        if not mcp_manager_instance:
            error_msg = "MCPManager is not initialized. Please check server logs."
        elif not mcp_manager_instance.has_endpoints_configured:
            error_msg = "MCPManager has no endpoints configured. Please check server configuration."
        elif not mcp_manager_instance.has_any_connection:
            error_msg = "MCPManager could not connect to any MCP instance. Please check MCP server statuses and logs."
        elif not mcp_manager_instance.openai_tools: # Implies connection but no tools fetched
            error_msg = "MCPManager connected but found no tools. Please check MCP tool definitions and logs."
        
        logger.error(f"MCP readiness check failed: {error_msg}")
        await stream_chat_updates("error", error_msg, client_queue)
        return

    await stream_chat_updates("user_message", user_input, client_queue)
    logger.info(f"Processing user request for web: '{user_input}'")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    turn_count = 0

    while turn_count < MAX_TURNS:
        turn_count += 1
        await stream_chat_updates("info", f"--- Agent Turn {turn_count}/{MAX_TURNS} ---", client_queue)
        logger.info(f"--- Turn {turn_count}/{MAX_TURNS} ---")
        
        serializable_messages_for_log = [msg.model_dump(exclude_unset=True) if hasattr(msg, 'model_dump') else msg for msg in messages]
        logger.debug(f"Messages to OpenAI:\n{json.dumps(serializable_messages_for_log, indent=2)}")

        try:
            await stream_chat_updates("thinking", "Contacting LLM...", client_queue)
            openai_client = get_openai_client() # Use shared client instance
            
            current_openai_tools = mcp_manager_instance.openai_tools # Fetched from MCPManager
            if not current_openai_tools:
                logger.warning("No OpenAI tools available from MCPManager for this turn. LLM will operate without tools.")
            print ("-----------------------------MESSAGES---------------------------")
            print (messages)
            print ("--------------------------------------------------------")
            response = await openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages, #type: ignore
                tools=current_openai_tools if current_openai_tools else None, 
                tool_choice="auto" if current_openai_tools else None, 
            )
            response_message = response.choices[0].message
            # Ensure messages list stores the Pydantic model directly if it's one
            messages.append(response_message) 

            if response_message.tool_calls:
                await stream_chat_updates("info", f"LLM requested {len(response_message.tool_calls)} tool(s).", client_queue)
                
                tool_results_messages = []
                for tool_call_from_openai in response_message.tool_calls:
                    tool_result_message = await execute_tool_call(tool_call_from_openai, client_queue)
                    tool_results_messages.append(tool_result_message)
                
                messages.extend(tool_results_messages) # Append tool results for next turn
                await stream_chat_updates("info", "Completed tool calls for this turn.", client_queue)
            else:
                final_content = response_message.content or "[No text content in response]"
                await stream_chat_updates("final_answer", final_content, client_queue)
                logger.info(f"LLM provided final answer: {final_content}")

                if original_input_was_voice and final_content and final_content.strip() != "[No text content in response]":
                    logger.info("Original input was voice, attempting TTS for the final answer.")
                    await stream_chat_updates("status", f"Generating speech output via {TTS_PROVIDER}...", client_queue)
                    
                    audio_bytes = await generate_text_to_speech(final_content) # Updated call
                    
                    if audio_bytes:
                        import base64
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        await stream_chat_updates("audio_response_data", 
                                                  {"audio_base64": audio_base64, 
                                                   "mime_type": "audio/mpeg", # Both providers output mp3
                                                   "text_for_reference": final_content[:100]+"..."
                                                  }, 
                                                  client_queue)
                        logger.info(f"Sent audio response (base64, via {TTS_PROVIDER}) to client.")
                    else:
                        logger.warning(f"Failed to generate TTS (via {TTS_PROVIDER}), so no audio response will be sent.")
                        await stream_chat_updates("warning", "Could not generate speech output for the response.", client_queue)
                return # Exit after final answer

        except openai.APIError as e:
            error_msg = f"OpenAI API Error in turn {turn_count}: {e}"
            logger.error(error_msg)
            await stream_chat_updates("error", error_msg, client_queue)
            return
        except Exception as e:
            error_msg = f"Unexpected error in processing loop turn {turn_count}: {e}"
            logger.error(error_msg, exc_info=True)
            await stream_chat_updates("error", error_msg, client_queue)
            return

    warning_msg = f"Reached maximum turns ({MAX_TURNS}) without a final answer."
    logger.warning(warning_msg)
    last_assistant_content = "N/A"
    # Iterate backwards to find the last assistant text message (not tool call)
    for msg_idx in range(len(messages) - 1, -1, -1):
        msg = messages[msg_idx]
        # Check if msg is a Pydantic model or dict
        role = getattr(msg, 'role', None) if not isinstance(msg, dict) else msg.get('role')
        content = getattr(msg, 'content', None) if not isinstance(msg, dict) else msg.get('content')
        
        if role == 'assistant' and content: 
            # Ensure it's not a message that *only* contains tool_calls
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            if not has_tool_calls:
                 last_assistant_content = content
                 break
    await stream_chat_updates("warning", f"{warning_msg} Last assistant thought: {last_assistant_content}", client_queue)
