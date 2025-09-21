# uvicorn main_web_app:app --reload

# ==============================================================================
# main_web_app.py: The single, unified server for the AI Testing Suite
# ==============================================================================

import os
import shutil
import uuid
import json
import io
import asyncio
import logging
from datetime import datetime

# --- FastAPI & Web Server Imports ---
from fastapi import FastAPI, Request, File, UploadFile, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTasks
from fastapi.middleware.wsgi import WSGIMiddleware
import aiofiles

# --- Flask & Test Generation Imports ---
from flask import Flask, request, jsonify, send_file
from openpyxl import Workbook

# --- Custom Application Logic Imports ---
from testingAgent import run_test_suite_from_web
from chat_processor import initialize_mcp_manager, shutdown_mcp_manager, OPENAI_MODEL
from voicechatapi import router as chat_api_router # Import the chat router

# --- Environment & OpenAI Imports ---
from dotenv import load_dotenv
from openai import OpenAI

# ==============================================================================
# 1. INITIAL SETUP & CONFIGURATION
# ==============================================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy library logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp.client.session").setLevel(logging.INFO)
logging.getLogger("api_routes").setLevel(logging.INFO)

# --- FastAPI App Initialization ---
app = FastAPI(title="CodeFire AI Testing Suite")

# ==============================================================================
# 2. STARTUP & SHUTDOWN EVENTS (for MCP Manager)
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing MCP Manager...")
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("FATAL: OPENAI_API_KEY is not set. The application cannot function.")
    await initialize_mcp_manager()
    logger.info("MCP Manager initialization process completed.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown: Cleaning up MCP Manager...")
    await shutdown_mcp_manager()

# ==============================================================================
# 3. FLASK APP SETUP (for Test Case Generation)
# ==============================================================================
flask_app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url= os.getenv("BASE_URL"))
DATA_FILE = 'test_cases.json'
INSTRUCTIONS_FILE = 'instructions.json'

# --- Flask Helper Functions ---
def load_test_cases():
    if not os.path.exists(DATA_FILE): return []
    try:
        with open(DATA_FILE, 'r') as f: return json.load(f)
    except json.JSONDecodeError: return []

def save_test_cases(test_cases):
    with open(DATA_FILE, 'w') as f: json.dump(test_cases, f, indent=4)

def load_instructions():
    if not os.path.exists(INSTRUCTIONS_FILE): return {"text": ""}
    try:
        with open(INSTRUCTIONS_FILE, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, dict) and "text" in data else {"text": ""}
    except json.JSONDecodeError: return {"text": ""}

def save_instructions(instructions_text):
    with open(INSTRUCTIONS_FILE, 'w') as f:
        json.dump({"text": instructions_text, "last_modified": datetime.utcnow().isoformat() + 'Z'}, f, indent=4)


# --- Flask Endpoints ---
@flask_app.route('/test_cases', methods=['GET', 'POST'])
def manage_test_cases():
    test_cases = load_test_cases()
    if request.method == 'GET': return jsonify(test_cases)
    if request.method == 'POST':
        data = request.get_json()
        test_case_id = data.get('id')
        if test_case_id:
            for tc in test_cases:
                if tc['id'] == test_case_id:
                    tc.update(data); tc['last_modified'] = datetime.utcnow().isoformat() + 'Z'; break
        else:
            new_tc = { "id": str(uuid.uuid4()), "created_at": datetime.utcnow().isoformat() + 'Z', **data }
            test_cases.append(new_tc)
        save_test_cases(test_cases)
        return jsonify({"message": "Test case saved", "test_cases": test_cases}), 201

@flask_app.route('/generate_test_cases', methods=['POST'])
def generate_test_cases_from_text():
    data = request.get_json()
    user_text = data.get('text', '')
    additional_instructions = data.get('additional_instructions', '')
    prompt = f"""You are an expert QA engineer. Generate a test case in JSON format with keys "heading", "steps", and "verification". The steps should be one per line. Use the provided instructions and primary requirements. IMPORTANT INSTRUCTIONS: \"\"\"{additional_instructions}\"\"\" PRIMARY REQUIREMENT: \"\"\"{user_text}\"\"\" JSON Output:"""
    try:
        completion = client.chat.completions.create(model="gpt-4-turbo-preview", messages=[{"role": "system", "content": "You are an AI assistant that generates test cases in a strict JSON format with keys 'heading', 'steps', and 'verification'."}, {"role": "user", "content": prompt}], temperature=0.5, response_format={"type": "json_object"})
        test_case_data = json.loads(completion.choices[0].message.content)
        if not all(k in test_case_data for k in ["heading", "steps", "verification"]): raise ValueError("Generated JSON is missing required keys.")
        return jsonify(test_case_data)
    except Exception as e:
        flask_app.logger.error(f"Error generating test case: {e}")
        return jsonify({"error": str(e)}), 500

@flask_app.route('/test_cases/<string:test_case_id>', methods=['DELETE'])
def delete_test_case(test_case_id):
    test_cases = load_test_cases()
    test_cases = [tc for tc in test_cases if tc.get('id') != test_case_id]
    save_test_cases(test_cases)
    return jsonify({"message": "Test case deleted"}), 200

@flask_app.route('/export_xlsx', methods=['POST'])
def export_xlsx():
    selected_ids = request.get_json().get('test_case_ids', [])
    test_cases_to_export = [tc for tc in load_test_cases() if tc['id'] in selected_ids]
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Cases"
    headers = ["S.No", "Test case heading", "Steps", "Verification", "Pass Status", "Status"]
    ws.append(headers)
    for i, tc in enumerate(test_cases_to_export, 1): ws.append([i, tc.get('heading'), tc.get('steps'), tc.get('verification'), tc.get('pass_status', ''), ""])
    file_stream = io.BytesIO()
    wb.save(file_stream)
    file_stream.seek(0)
    return send_file(file_stream, as_attachment=True, download_name='test_cases.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@flask_app.route('/instructions', methods=['GET', 'POST'])
def manage_instructions():
    if request.method == 'GET': return jsonify(load_instructions())
    if request.method == 'POST':
        save_instructions(request.get_json().get('text', ''))
        return jsonify({"message": "Instructions saved"}), 200

# ==============================================================================
# 4. FASTAPI ROUTERS (for Test Execution and Chat)
# ==============================================================================

# --- Router for Test Execution ---
execution_router = APIRouter()
class ConnectionManager:
    def __init__(self): self.active_connections: dict[str, WebSocket] = {}
    async def connect(self, websocket: WebSocket, run_id: str): await websocket.accept(); self.active_connections[run_id] = websocket
    def disconnect(self, run_id: str):
        if run_id in self.active_connections: del self.active_connections[run_id]
    async def get_connection(self, run_id: str) -> WebSocket | None: return self.active_connections.get(run_id)
manager = ConnectionManager()

@execution_router.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await manager.connect(websocket, run_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(run_id)

@execution_router.post("/upload-and-run-test/")
async def create_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.path.splitext(file.filename)[0]}"
    run_dir = os.path.join("results", run_id); os.makedirs(run_dir, exist_ok=True)
    file_path = os.path.join(run_dir, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file: await out_file.write(await file.read())
    background_tasks.add_task(start_test_run, file_path, run_id)
    return {"message": "File uploaded. Test run starting.", "run_id": run_id}

async def start_test_run(file_path: str, run_id: str):
    await asyncio.sleep(1)
    websocket = await manager.get_connection(run_id)
    if websocket:
        await run_test_suite_from_web(file_path, websocket, run_id)
    else:
        logger.error(f"Execution Error: Could not find WebSocket for run_id {run_id}")

# ==============================================================================
# 5. MOUNTING & FINAL APP SETUP
# ==============================================================================

# Mount the static files directory (for CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# **** FIX IS HERE: Mount the results directory to serve screenshots and reports ****
app.mount("/results", StaticFiles(directory="results"), name="results")

# Mount the Flask app for test case generation
app.mount("/generate", WSGIMiddleware(flask_app))

# Include the API router for interactive chat
app.include_router(chat_api_router, prefix="/chat")

# Include the API router for test execution
app.include_router(execution_router, prefix="/execution")

# --- Root Endpoint to serve the main HTML page ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    templates = Jinja2Templates(directory="templates")
    return templates.TemplateResponse("index.html", {"request": request})
# uvicorn main_web_app:app --reload

if __name__ == "__main__":
    import uvicorn
    logger.info("--- Starting Web testing agent---")
    
    
    uvicorn.run("main_web_app:app", host="0.0.0.0", port=8000, timeout_graceful_shutdown=1,reload=True)

# --- END OF MERGED main.py --- 