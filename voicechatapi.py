import os
import uuid
import json
import asyncio
import logging
from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import openai
import aiofiles
from pathlib import Path
from shared_state import client_queues
from chat_processor import process_user_request_for_web
from dotenv import load_dotenv

load_dotenv()

# Logger namespacing is good practice
logger = logging.getLogger("api_routes")

router = APIRouter()

# REMOVED: The get_index route is no longer needed here.
# The main app will serve the unified index.html.

# --- SSE Stream for Chat Updates ---
async def chat_event_generator(client_id: str, request: Request):
    queue = asyncio.Queue()
    client_queues[client_id] = queue
    logger.info(f"SSE Generator: Created queue for client {client_id}")
    try:
        while True:
            if await request.is_disconnected():
                logger.info(f"SSE Generator: Client {client_id} disconnected.")
                break
            message = await queue.get()
            if message is None:
                break
            yield f"data: {json.dumps(message)}\n\n"
            queue.task_done()
    except asyncio.CancelledError:
        logger.warning(f"SSE stream for client {client_id} was cancelled.")
    finally:
        if client_id in client_queues:
            del client_queues[client_id]
            logger.info(f"SSE Generator: Cleaned up queue for client {client_id}")

@router.get("/chat_stream/{client_id}")
async def chat_stream(client_id: str, request: Request):
    logger.info(f"Client {client_id} connected to SSE stream.")
    return StreamingResponse(chat_event_generator(client_id, request), media_type="text/event-stream")

# --- API Endpoint for Processing Chat Input ---
@router.post("/process_chat/{client_id}")
async def handle_chat_message(
    client_id: str,
    text_input: str = Form(None),
    audio_file: UploadFile = File(None)
):
    if client_id not in client_queues:
        logger.error(f"CRITICAL: Client queue for {client_id} not found!")
        raise HTTPException(status_code=404, detail="No active chat session found.")

    client_queue = client_queues[client_id]
    user_message = ""
    input_was_voice = False

    if audio_file and audio_file.filename:
        input_was_voice = True
        temp_audio_path = f"temp_audio_{uuid.uuid4()}.webm"
        try:
            async with aiofiles.open(temp_audio_path, 'wb') as out_file:
                content = await audio_file.read()
                await out_file.write(content)

            if os.path.getsize(temp_audio_path) > 0:
                openai_client = openai.AsyncOpenAI()
                transcript_response = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=Path(temp_audio_path)
                )
                user_message = transcript_response.text
                await client_queue.put({"type": "transcription_result", "data": user_message})
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await client_queue.put({"type": "error", "data": f"Audio processing error: {e}"})
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    if not user_message and text_input:
        user_message = text_input

    if user_message:
        logger.info(f"Final user message for client {client_id}: '{user_message}'")
        asyncio.create_task(process_user_request_for_web(user_message, client_queue, input_was_voice))
        return {"status": "processing_started"}
    
    logger.error(f"No user message could be processed for client {client_id}.")
    await client_queue.put({"type": "error", "data": "Input could not be processed."})
    return {"status": "error", "message": "No valid input received."}