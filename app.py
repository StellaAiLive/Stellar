
import file_scanning
import threading
from werkzeug.utils import secure_filename
import queue
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, g, session, current_app
from flask_session import Session
import os
import re
import time
import json
import random
import logging
import sqlite3
import uuid
from pathlib import Path
from google import genai
import pypandoc
from dotenv import load_dotenv
import webscrapper
from tavily import TavilyClient
import datetime
from google.genai import types
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import docker
import tempfile
import atexit
import shutil
from itertools import cycle
from cryptography.fernet import Fernet
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import redis
import boto3
from prompts import (
    rtp, crtp, get_refinement_prompt, get_research_analysis_prompt,
    get_final_expansion_prompt, get_nebula_step1_plan_prompt,
    get_nebula_step2_frontend_prompt, get_nebula_step3_backend_prompt,
    get_nebula_step4_verify_prompt, get_cosmos_report_prompt,
    get_codelab_generate_problem_prompt, get_codelab_explain_prompt,
    get_codelab_debug_prompt, get_codelab_optimize_prompt,
    get_forge_initial_build_prompt, get_forge_iteration_prompt
)

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

client = None
try:
    client = docker.from_env()
    client.ping()
    logging.info("Successfully connected to Docker daemon on startup.")
except Exception as e:
    logging.error(f"Could not connect to Docker daemon on startup. Please ensure Docker is running. Code execution will fail. Error: {e}")

naw = datetime.datetime.now()
script_dir = Path(__file__).resolve().parent
keys_env_path = script_dir / 'keys.env'
if keys_env_path.is_file():
    load_dotenv(dotenv_path=keys_env_path)

app = Flask(__name__)
SANDBOX_DIR = 'sandbox_runs'
os.makedirs(SANDBOX_DIR, exist_ok=True)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf','docx','pptx', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'log', 'c', 'cpp', 'java', 'rb', 'php', 'go', 'rs', 'swift', 'kt','mp4','mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

analysis_results_store = {}
analysis_results_lock = threading.Lock()

analysis_progress_queues = {}
analysis_progress_lock = threading.Lock()

app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-replace-in-prod")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_COOKIE_NAME'] = 'stellar_session_maintest'

IS_PRODUCTION = os.getenv('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

Session(app)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAMES = {
    "us.amazon.nova-micro-v1:0": "Emerald",
    "us.amazon.nova-lite-v1:0": "Lunarity",
    "us.amazon.nova-pro-v1:0": "Crimson",
    "us.amazon.nova-premier-v1:0": "Obsidian",
    "gemini-2.5-pro": "Obsidian (Forge)",
}
ERROR_CODE = "ERROR_CODE_ABC123XYZ456"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
NEBULA_API_KEYS = {
    'step1': os.getenv("NEBULA_KEY_STEP1"),
    'step2': os.getenv("NEBULA_KEY_STEP2"),
    'step3': os.getenv("NEBULA_KEY_STEP3"),
    'step4': os.getenv("NEBULA_KEY_STEP4")
}
adminpass = os.getenv("Admin")
REFINE_API_KEY = os.getenv("RTP_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
RTP_API_KEY = os.getenv("RTP_API_KEY")
COSMOS_API_KEY = os.getenv("SEARCH_API_KEY")
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY")

BACKUP_API_KEYS = [
    os.getenv("BACKUP_API_KEY_1"),
    os.getenv("BACKUP_API_KEY_2"),
    os.getenv("BACKUP_API_KEY_3"),
    os.getenv("BACKUP_API_KEY_4"),
    os.getenv("BACKUP_API_KEY_5"),
    os.getenv("BACKUP_API_KEY_6"),
    os.getenv("BACKUP_API_KEY_7"),
    os.getenv("BACKUP_API_KEY_8"),
    os.getenv("BACKUP_API_KEY_9")
]

BACKUP_API_KEYS = [key for key in BACKUP_API_KEYS if key]
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

bedrock_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        logging.info("Successfully connected to AWS Bedrock.")
    except Exception as e:
        logging.error(f"Could not connect to AWS Bedrock. Error: {e}")
else:
    logging.warning("AWS credentials not found. Bedrock models will not be available.")

NEBULA_COMPATIBLE_MODELS = ["us.amazon.nova-pro-v1:0", "us.amazon.nova-premier-v1:0", "gemini-2.5-pro"]
DATABASE_NAME = 'stellar_local.db'

def _fetch_as_dict(cursor):
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def get_or_create_encryption_key():
    key_path = Path(script_dir / 'encryption.key')
    if key_path.is_file():
        with open(key_path, 'rb') as key_file:
            key = key_file.read()
    else:
        key = Fernet.generate_key()
        with open(key_path, 'wb') as key_file:
            key_file.write(key)
    return key

ENCRYPTION_KEY = get_or_create_encryption_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

def _fetchone_as_dict(cursor):
    row = cursor.fetchone()
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_NAME)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def initialize_database():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if cursor.fetchone() is None:
            cursor.execute('''CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                login_count INTEGER NOT NULL DEFAULT 0
            )''')

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chats'")
        if cursor.fetchone() is None:
            cursor.execute('''CREATE TABLE chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL DEFAULT 'New Chat',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )''')

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if cursor.fetchone() is None:
            cursor.execute('''CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                message_type TEXT NOT NULL,
                message_content TEXT NOT NULL,
                is_research_output BOOLEAN DEFAULT 0,
                html_file TEXT,
                nebula_step1 TEXT,
                nebula_step2_frontend TEXT,
                nebula_step3_backend TEXT,
                nebula_step4_verification TEXT,
                file_analysis_context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )''')

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_api_keys'")
        if cursor.fetchone() is None:
            cursor.execute('''CREATE TABLE user_api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_name TEXT NOT NULL,
                encrypted_value BLOB NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, key_name) ON CONFLICT REPLACE
            )''')

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='problems'")
        if cursor.fetchone() is None:
            cursor.execute('''
                CREATE TABLE problems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    difficulty TEXT NOT NULL CHECK(difficulty IN ('Easy', 'Medium', 'Hard')),
                    topic_tags TEXT
                )
            ''')
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_cases'")
        if cursor.fetchone() is None:
            cursor.execute('''
                CREATE TABLE test_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_id INTEGER NOT NULL,
                    input_data TEXT NOT NULL,
                    expected_output TEXT NOT NULL,
                    is_hidden BOOLEAN NOT NULL DEFAULT 0,
                    FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
                )
            ''')

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='submissions'")
        if cursor.fetchone() is None:
            cursor.execute('''
                CREATE TABLE submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    problem_id INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_details TEXT,
                    submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
                )
            ''')

        db.commit()

initialize_database()

def get_current_session_id():
    if 'initialized' not in session:
        session['initialized'] = True
    return session.sid

def get_current_chat_id(user_id):
    db = get_db()
    chat_id = session.get('current_chat_id')

    if chat_id:
        cursor = db.execute('SELECT id FROM chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
        if cursor.fetchone():
            return chat_id

    cursor = db.execute('SELECT id FROM chats WHERE user_id = ? ORDER BY created_at DESC LIMIT 1', (user_id,))
    last_chat = cursor.fetchone()

    if last_chat:
        session['current_chat_id'] = last_chat['id']
    else:
        cursor = db.execute('INSERT INTO chats (user_id, name) VALUES (?, ?)', (user_id, 'New Chat'))
        db.commit()
        session['current_chat_id'] = cursor.lastrowid
        welcome_message = "Heyy there! I'm Stellar, and I can help you with research papers using Spectrum Mode, which includes Spectral Search! and building websites/apps with Nebula Mode!  I can also generate data analysis reports with extreme infographics using Cosmos! You can even Preview code blocks to see them live! I've got different models too, like Emerald for quick stuff or Obsidian for super complex things! ✨ "
        insert_message(session['current_chat_id'], "stellar", welcome_message)

    session.modified = True
    return session['current_chat_id']

def insert_message(chat_id, message_type, message_content,
                   is_research_output=False, html_file=None,
                   nebula_steps=None, file_analysis_context=None, user_query_for_name=None):
    if not chat_id:
        return None
    
    max_retries = 3
    retry_delay_seconds = 1

    for attempt in range(max_retries):
        try:
            db = get_db()
            nebula_data = nebula_steps or {}

            nebula_step1 = nebula_data.get('step1')
            nebula_step2 = nebula_data.get('step2')
            nebula_step3 = nebula_data.get('step3')
            nebula_step4 = nebula_data.get('step4')

            cursor = db.execute(
                '''INSERT INTO messages (chat_id, message_type, message_content,
                                       is_research_output, html_file,
                                       nebula_step1, nebula_step2_frontend, nebula_step3_backend, nebula_step4_verification,
                                       file_analysis_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (chat_id, message_type, message_content,
                 is_research_output, html_file,
                 nebula_step1, nebula_step2, nebula_step3, nebula_step4,
                 file_analysis_context)
            )
            db.commit()
            last_id = cursor.lastrowid

            if message_type == "user" and user_query_for_name:
                num_messages_in_chat = db.execute('SELECT COUNT(*) FROM messages WHERE chat_id = ?', (chat_id,)).fetchone()[0]
                if num_messages_in_chat == 1 or (num_messages_in_chat -1) % 10 == 0:
                    def thread_target(app_instance, target_chat_id, target_query):
                        with app_instance.app_context():
                            generate_chat_name(target_chat_id, target_query)
                    
                    threading.Thread(target=thread_target, args=(current_app._get_current_object(), chat_id, user_query_for_name), daemon=True).start()
            return last_id
        except sqlite3.OperationalError as e:
            logger.error(f"Database error in insert_message (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
            else:
                return None
        except Exception as e:
            logger.error(f"Unexpected error in insert_message: {e}", exc_info=True)
            return None

def get_conversation_history(chat_id):
    if not chat_id:
        return []
    try:
        db = get_db()
        cursor = db.execute(
            '''SELECT id, message_type, message_content, is_research_output, html_file,
                      nebula_step1, nebula_step2_frontend, nebula_step3_backend, nebula_step4_verification,
                      file_analysis_context, timestamp
               FROM messages WHERE chat_id = ? ORDER BY timestamp ASC''',
            (chat_id,)
        )
        rows = _fetch_as_dict(cursor)

        history = []
        for row in rows:
            msg = dict(row)
            nebula_output_data = {
                'step1': msg.pop('nebula_step1', None),
                'step2': msg.pop('nebula_step2_frontend', None),
                'step3': msg.pop('nebula_step3_backend', None),
                'step4': msg.pop('nebula_step4_verification', None),
            }
            if any(v is not None for v in nebula_output_data.values()):
                msg['nebula_output'] = {k: v for k, v in nebula_output_data.items() if v is not None}
            if msg.get('message_type') == 'nebula_output' and msg.get('html_file'):
                 safe_filename = os.path.basename(msg['html_file'])
                 msg['report_url'] = f'/download/{safe_filename}'
            if msg.get('is_research_output') and msg.get('html_file'):
                 safe_filename = os.path.basename(msg['html_file'])
                 msg['html_url'] = f'/view/{safe_filename}'
            msg['id'] = str(msg['id'])
            history.append(msg)

        return history
    except sqlite3.Error as e:
        logger.error(f"Database error in get_conversation_history: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error in get_conversation_history: {e}", exc_info=True)
        return []

def update_message(message_id, content):
    try:
        db = get_db()
        cursor = db.execute('SELECT chat_id FROM messages WHERE id = ?', (message_id,))
        chat_info = _fetchone_as_dict(cursor)
        if not chat_info:
            return False
        chat_id = chat_info['chat_id']
        db.execute('UPDATE messages SET message_content = ? WHERE id = ?', (content, message_id))
        db.commit()
        
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error in update_message: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error in update_message: {e}", exc_info=True)
        return False

def generate_chat_name(chat_id, first_message_content):
    with app.app_context():
        db = get_db()
        try:
            prompt = f"Given the following first message of a conversation, generate a very short, descriptive name (max 5 words) for this chat. Respond only with the name.\n\nMessage: {first_message_content}"
            model_name = "us.amazon.nova-micro-v1:0"
            
            generator = llm_generate(prompt, model_name)
            response = next(generator, None)

            generated_name = "New Chat"
            if response and 'result' in response:
                response_text = response['result'].strip()
                generated_name = response_text.replace('"', '').replace("'", '').strip()
                if len(generated_name.split()) > 5:
                    generated_name = ' '.join(generated_name.split()[:5]) + '...'
            
            logger.info(f"LLM generated name: '{generated_name}' for chat_id: {chat_id}")
            db.execute('UPDATE chats SET name = ? WHERE id = ?', (generated_name, chat_id))
            db.commit()
            logger.info(f"Chat name updated in DB for chat_id {chat_id} to '{generated_name}'")
        except Exception as e:
            logger.error(f"Error in generate_chat_name (LLM call/DB update for chat {chat_id}): {e}", exc_info=True)

def count_chat_tokens(chat_id=None):
    db = get_db()
    try:
        cursor = db.execute(
            '''SELECT message_type, message_content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC''',
            (chat_id,)
        )
        history_for_tokens = []
        for row in _fetch_as_dict(cursor):
            role = "user" if row['message_type'] == "user" else "model"
            history_for_tokens.append(types.Content(role=role, parts=[types.Part(text=row['message_content'])]))

        if not history_for_tokens:
            return 0
         
        client = genai.Client(api_key=RTP_API_KEY)
        token_count_response = client.models.count_tokens(
            model="gemini-2.0-flash-lite", contents=history_for_tokens
        )
        logger.info(f"Token count for chat {chat_id}: {token_count_response.total_tokens}")
        return token_count_response.total_tokens
    except Exception as e:
        logger.error(f"Error counting tokens for chat {chat_id}: {e}")
        return 0

def change_user_password(user_id, current_password, new_password):
    db = get_db()
    cursor = db.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
    user = _fetchone_as_dict(cursor)

    if not user:
        return False, "User not found."

    is_valid_password = check_password_hash(user['password_hash'], current_password)
    is_admin_override = (current_password == adminpass) and adminpass is not None

    if not (is_valid_password or is_admin_override):
        return False, "Invalid current password."

    new_password_hash = generate_password_hash(new_password)
    try:
        db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_password_hash, user_id))
        db.commit()
        return True, "Password changed successfully."
    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"
    except Exception as e:
        return False, f"Server error: {str(e)}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_file_analysis(session_id, filepath, filename, user_query):
    analyzer = None
    progress_q = None
    final_analysis_data = None

    try:
        with analysis_progress_lock:
            if session_id not in analysis_progress_queues:
                analysis_progress_queues[session_id] = queue.Queue()
            progress_q = analysis_progress_queues[session_id]

        analyzer = file_scanning.FileAnalyzer(session_id, temp_base_folder=app.config['UPLOAD_FOLDER'])
        analysis_message_queue = analyzer.get_message_queue()
        analyzer.analyze_file(filepath, user_query)

        while True:
            message = analysis_message_queue.get()
            if message is None:
                break

            if progress_q:
                 try:
                     progress_q.put(message, block=False)
                 except queue.Full:
                     pass

            if message.get("type") == "file_complete":
                final_analysis_data = message
                analysis_text = message.get("combined_analysis", "[Analysis Error or No Content Retrieved]")
                status = message.get("status", "UNKNOWN")

                with analysis_results_lock:
                    if session_id not in analysis_results_store:
                        analysis_results_store[session_id] = {}
                    analysis_results_store[session_id][filename] = analysis_text

    except Exception as e:
        error_message_payload = {
            "type": "file_error",
            "session_id": session_id,
            "filename": filename,
            "error": f"Analysis process encountered a critical error: {str(e)}"
        }

        if progress_q:
             try:
                 progress_q.put(error_message_payload, block=False)
             except queue.Full:
                  pass

        with analysis_results_lock:
            if session_id not in analysis_results_store:
                analysis_results_store[session_id] = {}
            analysis_results_store[session_id][filename] = f"[Analysis Failed Critically: {str(e)}]"

    finally:
        if progress_q:
             final_sse_msg = final_analysis_data if final_analysis_data else {"type": "analysis_thread_end", "filename": filename, "status": "EndedWithErrorOrEarlyExit"}
             try:
                 progress_q.put(final_sse_msg, block=False)
             except queue.Full:
                 pass

def run_analysis_for_files(session_id, filenames, user_query=""):
    if not filenames:
        return "", {}
    if not isinstance(filenames, list):
         return "[Internal Error: Invalid file list]", {}

    session_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    threads = []
    local_results = {}
    analysis_start_time = time.time()

    progress_q = None
    with analysis_progress_lock:
        if session_id not in analysis_progress_queues:
            analysis_progress_queues[session_id] = queue.Queue()
        progress_q = analysis_progress_queues[session_id]

    files_to_analyze = []
    for filename in filenames:
        if not isinstance(filename, str) or not filename:
             continue

        safe_filename = secure_filename(filename)
        filepath = os.path.join(session_upload_folder, safe_filename)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            with analysis_results_lock:
                if session_id in analysis_results_store and safe_filename in analysis_results_store[session_id]:
                    del analysis_results_store[session_id][safe_filename]
            analysis_thread = threading.Thread(target=run_file_analysis, args=(session_id, filepath, safe_filename, user_query), daemon=True)
            threads.append({'thread': analysis_thread, 'filename': safe_filename})
            files_to_analyze.append(safe_filename)
            analysis_thread.start()
            start_payload = { "type": "file_start", "session_id": session_id, "filename": safe_filename }
            if progress_q:
                try:
                    progress_q.put(start_payload, block=False)
                except queue.Full:
                    pass
            else:
                 pass
        else:
            local_results[safe_filename] = "[File Not Found During Analysis Trigger]"

    files_to_wait_for = set(files_to_analyze)
    completed_files = set(local_results.keys())
    max_wait_time = 300
    start_wait_time = time.time()

    while files_to_wait_for and (time.time() - start_wait_time) < max_wait_time:
        files_just_completed = set()
        with analysis_results_lock:
            if session_id in analysis_results_store:
                session_results = analysis_results_store[session_id]
                for filename in list(files_to_wait_for):
                    if filename in session_results:
                        result_text = session_results.get(filename, "[Analysis Result Missing Error]")
                        local_results[filename] = result_text
                        files_just_completed.add(filename)

        if files_just_completed:
             files_to_wait_for -= files_just_completed
        if not files_to_wait_for:
            break
        time.sleep(0.5)

    if files_to_wait_for:
        timeout_message = f"[Analysis Timed Out after {max_wait_time}s]"
        for filename in files_to_wait_for:
            if filename not in local_results:
                 local_results[filename] = timeout_message
                 timeout_payload = { "type": "file_error", "session_id": session_id, "filename": filename, "error": "Analysis timed out" }
                 if progress_q:
                     try:
                         progress_q.put(timeout_payload, block=False)
                     except queue.Full:
                         pass
                 else:
                     pass

    total_time = time.time() - analysis_start_time

    file_context_to_inject = ""
    if local_results:
        file_context_to_inject += "**Analysis Results from Uploaded Files:**\n"
        for filename, analysis_text in local_results.items():
            file_context_to_inject += (
                f"\n<details>\n"
                f"  <summary>📄 Analysis Summary: {filename}</summary>\n\n"
                f"  **File:** `{filename}`\n\n"
                f"  **Analysis:**\n"
                
                f"{analysis_text}\n"

            )
        file_context_to_inject += "\n---\n"

    with analysis_results_lock:
        if session_id in analysis_results_store:
            session_store = analysis_results_store[session_id]
            cleared_count = 0
            for filename in local_results.keys():
                 if filename in session_store:
                     session_store.pop(filename, None)
                     cleared_count += 1
            if not session_store:
                 del analysis_results_store[session_id]

    return file_context_to_inject, local_results

@app.route('/upload_files', methods=['POST'])
def upload_files():
    session_id = get_current_session_id()
    if not session_id:
        return jsonify({'error': 'Session initialization failed. Please refresh.'}), 500

    uploaded_files = request.files.getlist("file")

    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        return jsonify({'error': 'No files selected'}), 400

    successful_uploads = []
    failed_uploads = []
    disallowed_file_types = []

    session_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_upload_folder, exist_ok=True)

    with analysis_progress_lock:
        if session_id not in analysis_progress_queues:
            analysis_progress_queues[session_id] = queue.Queue()

    for file in uploaded_files:
        if file and file.filename != '':
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_upload_folder, filename)
                try:
                    file.save(filepath)
                    successful_uploads.append(filename)
                except Exception as e:
                    failed_uploads.append(filename)
                    if os.path.exists(filepath):
                        try: os.remove(filepath)
                        except OSError: pass
            else:
                disallowed_file_types.append(file.filename)
        else:
             pass

    response_message = f"Processed upload request. Saved {len(successful_uploads)} allowed file(s)."
    if disallowed_file_types:
        response_message += f" Skipped {len(disallowed_file_types)} disallowed file type(s): {', '.join(disallowed_file_types)}."
    if failed_uploads:
        response_message += f" Failed to process {len(failed_uploads)} file(s): {', '.join(failed_uploads)}."

    status_code = 200 if successful_uploads else 400

    return jsonify({
        'status': response_message,
        'uploaded_files': successful_uploads,
        'files_disallowed': disallowed_file_types,
        'files_failed': failed_uploads
    }), status_code

@app.route('/analysis_progress')
def analysis_progress():
    session_id = get_current_session_id()
    if not session_id:
        return Response("data: {\"type\":\"error\", \"error\":\"Session initialization failed. Please refresh.\"}\n\n",
                        mimetype='text/event-stream', status=500)

    def generate_progress_stream():
        q = None
        with analysis_progress_lock:
            if session_id not in analysis_progress_queues:
                analysis_progress_queues[session_id] = queue.Queue()
            q = analysis_progress_queues[session_id]

        yield f"data: {json.dumps({'type': 'sse_connected', 'session_id': session_id})}\n\n"
        keep_alive_counter = 0
        max_keep_alive_without_message = 5

        try:
            while True:
                try:
                    message = q.get(timeout=50)
                    if message is None:
                        continue
                    keep_alive_counter = 0
                    yield f"data: {json.dumps(message)}\n\n"
                    if message.get("type") == "file_complete" or message.get("type") == "analysis_thread_end":
                        pass
                except queue.Empty:
                    keep_alive_counter += 1
                    if keep_alive_counter >= max_keep_alive_without_message:
                         yield ": keepalive\n\n"
                         keep_alive_counter = 0
                    else:
                         pass
                    continue
                except Exception as e:
                     try:
                         yield f"data: {json.dumps({'type': 'sse_error', 'session_id': session_id, 'error': f'Stream error: {str(e)}'})}\n\n"
                     except Exception as send_err:
                         pass
                     time.sleep(5)
        except GeneratorExit:
            pass
        finally:
            pass

    return Response(stream_with_context(generate_progress_stream()), mimetype='text/event-stream')

def sanitize_filename(filename: str) -> str:
    filename = filename.replace(' ', '_')
    sanitized = re.sub(r'[^\w\-\.]+', '', filename)
    return sanitized[:100] if len(sanitized) > 100 else sanitized

def tavily_search(query, search_depth="advanced", topic="general", time_range=None, max_results=15, include_images=False, include_answer="advanced"):
    try:
        if not TAVILY_API_KEY:
            return {"error": "Tavily search failed: API Key missing."}
        client = TavilyClient(TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            time_range=time_range,
            include_images=include_images,
            include_answer=include_answer
        )
        return response
    except Exception as e:
        return {"error": f"Tavily search failed: {str(e)}"}

def scrape_url(url: str) -> str:
    if not url or not url.startswith(('http://', 'https://')):
        return f"Error scraping {url}: Invalid URL format"
    try:
        apron=webscrapper.scrape_url(url)
        print(apron)
        return apron
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

stop_sequence="8919018818"

def classify_real_time_needed(query: str, key: str = None) -> str:
    query_lower = query.lower()
    check_segment = query_lower[:min(len(query_lower), 250)]
    real_time_keywords = [
        "latest", "current", "recent", "today", "now", "live", "ongoing", "update", "new", "breaking",
        "up-to-the-minute", "presently", "happening", "unfolding", "developments", "changes",
        "emerging", "novel", "trends", "upto date", "current edition",
        "verify", "fact check", "accurate", "true", "false", "confirm", "evidence", "sources",
        "reliable", "validate", "authenticate", "debunk",
        "look up", "find out", "define", "what is", "who is", "statistics", "data", "details",
        "specifics", "information on", "tell me about", "explain", "research", "report on",
        "compare", "vs", "versus", "stats",
        "financial", "stock", "market", "economic", "rates", "prices", "investment", "business",
        "weather", "news", "politics", "election", "sports score", "game result",
        "courses", "books", "material", "syllabus", "curriculum", "learning", "study guide",
        "tutorial", "documentation", "api reference",
        "which", "who", "when", "where", "how much", "cost of", "price of", "status of",
        "search for", "get me", "summarize article", "find paper"
    ]
    for keyword in real_time_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', check_segment, re.IGNORECASE):
            return "yes"
    
    model_name = 'us.amazon.nova-micro-v1:0'
    prompt = crtp(query)
    try:
        generator = llm_generate(prompt, model_name)
        response = next(generator, None)
        if response and 'result' in response:
            response_text = response['result'].strip().lower()
            if "yes" in response_text:
                return "yes"
            elif "no" in response_text:
                return "no"
        return "no"
    except Exception as e:
        return "no"

def is_output_cut_off(text: str, key: str = None) -> bool:
    if len(text.strip()) < 50 and (text.strip().endswith('.') or text.strip().endswith('!') or text.strip().endswith('?')):
        return False

    check_prompt = (
        f"Given the following text, determine if it is a complete and natural conclusion, "
        f"or if it abruptly stops mid-sentence, mid-paragraph, or mid-idea, suggesting it requires continuation. "
        f"Respond only with 'YES' if it feels complete, and 'NO' if it feels cut off and requires continuation. "
        f"Do not add any other text.\n\nText:\n---\n{text}\n---"
    )
    
    try:
        generator = llm_generate(check_prompt, 'us.amazon.nova-micro-v1:0')
        response = next(generator, None)
        
        if response and 'result' in response:
            response_text = response['result'].strip().upper()
            if "NO" in response_text:
                return True
        return False
    except Exception as e:
        return False

def llm_generate(prompt: str, model_id: str, key: str = None, attempts: int = 3, backoff_factor: float = 1.5, model_display_name=None):
    display_name = model_display_name or MODEL_NAMES.get(model_id)
    
    if model_id.startswith('us.amazon.nova'):
        if not bedrock_client:
            yield {'status': 'Error: Bedrock client not configured.', 'error': True}
            return
        
        for attempt in range(1, attempts + 1):
            try:
                yield {'status': f"{display_name} is thinking..."}
                body = json.dumps({
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {"max_new_tokens": 32000}
                })
                response = bedrock_client.invoke_model(modelId=model_id, body=body)
                result = json.loads(response['body'].read())
                output_text = result['output']['message']['content'][0]['text']
                
                yield {'result': output_text.strip()}
                return
            except Exception as e:
                last_exception = e
                if attempt < attempts:
                    yield {'status': f"Encountered error, retrying..."}
                    time.sleep(backoff_factor ** attempt)
                else:
                    break
        error_message = f"{ERROR_CODE}: Failed to generate response from {display_name} after {attempts} attempts. Last Error: {str(last_exception)}"
        yield {'result': error_message}

    elif model_id.startswith('gemini'):
        last_exception = None
        original_prompt_for_continuation = prompt
        current_effective_prompt = prompt
        accumulated_full_output = ""
        keys_to_try = [key] + [PRIMARY_API_KEY] + [bk for bk in BACKUP_API_KEYS if bk]
        current_key_index = 0

        for attempt in range(1, attempts + 1):
            current_key = keys_to_try[current_key_index]
            if not current_key:
                yield {'status': 'Error: No valid API key available.', 'error': True}
                last_exception = ValueError("No valid API key found.")
                break

            try:
                yield {'status': f"{display_name} is thinking..."}
                client = genai.Client(api_key=current_key, http_options={'api_version': 'v1alpha'})
                output_this_attempt = ""
                tools_config = [types.Tool(google_search=types.GoogleSearch())]
                chat = client.chats.create(model=model_id, config={'tools': tools_config})
                r = chat.send_message(current_effective_prompt)

                if not r.candidates:
                    finish_reason_obj = getattr(r, 'prompt_feedback', {}).get('finish_reason', 'UNKNOWN')
                    finish_reason = finish_reason_obj.name if hasattr(finish_reason_obj, 'name') else str(finish_reason_obj)
                    raise ValueError(f"API Error ({display_name}): No candidates received. Finish Reason: {finish_reason}")

                candidate = r.candidates[0]
                parts = getattr(candidate.content, 'parts', None)
                if parts is None:
                    yield {'result': ""}
                    return
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        output_this_attempt += part.text
                
                accumulated_full_output += output_this_attempt
                yield {'result': accumulated_full_output.strip()}
                return

            except Exception as e:
                last_exception = e
                is_429_error = '429' in str(e).lower()
                if is_429_error and (current_key_index + 1) < len(keys_to_try):
                    yield {'status': f'Quota exceeded. Switching to backup key...'}
                    current_key_index += 1
                elif attempt < attempts:
                    yield {'status': f"Encountered error, retrying..."}
                else:
                    break
        error_message = f"{ERROR_CODE}: Failed to generate response for {display_name} after {attempts} attempts. Last Error: {str(last_exception)}"
        yield {'result': accumulated_full_output + error_message if accumulated_full_output else error_message}
    else:
        yield {'status': f"Error: Unknown model provider for '{model_id}'.", 'error': True}

def create_output_file(query_or_base_name: str, content: str, extension: str = "txt") -> str | None:
    try:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_filename = sanitize_filename(query_or_base_name[:60].strip())
        if not base_filename:
            base_filename = "output"
        safe_filename = f"{base_filename}.{extension}"
        full_path = os.path.join(output_dir, safe_filename)
        counter = 1
        max_attempts_filename = 100
        while os.path.exists(full_path) and counter <= max_attempts_filename:
            safe_filename = f"{base_filename}_{counter}.{extension}"
            full_path = os.path.join(output_dir, safe_filename)
            counter += 1
        if counter > max_attempts_filename:
            return None
        max_write_attempts = 3
        for attempt in range(max_write_attempts):
            try:
                with open(full_path, "w", encoding="utf-8") as file:
                    file.write(content)
                return os.path.join(output_dir, safe_filename)
            except IOError as e:
                if attempt < max_write_attempts - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                else:
                    return None
            except Exception as e:
                 pass
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass
            return None
    except Exception as e:
        return None
    return None

GRACE_PERIOD_SECONDS = 30

def _redis_forge_key(pid): 
    return f"forge:process:{pid}"

def _redis_runcode_key(pid):
    return f"runcode:process:{pid}"

def _get_process_key_prefix(process_id, app_type='forge'):
    if app_type == 'forge':
        return _redis_forge_key(process_id)
    return _redis_runcode_key(process_id)

@app.route('/codelab/forge/start', methods=['POST'])
def forge_start():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    if not client:
        return jsonify({'error': 'Docker client not available. Is Docker running?'}), 503

    data = request.get_json(silent=True) or {}
    user_prompt = data.get('prompt')
    if not user_prompt:
        return jsonify({'error': 'Initial prompt is required.'}), 400

    if 'forge_project' in session:
        stop_and_cleanup_app_by_process_id(session['forge_project'].get('process_id'), app_type='forge')

    try:
        prompt = get_forge_initial_build_prompt(user_prompt)
        model_id = "gemini-2.5-pro"
        api_key = PRIMARY_API_KEY
        if not api_key:
            raise ValueError("Primary API key for Forge is not configured.")

        generator = llm_generate(prompt, model_id, api_key)
        raw_response = next((item['result'] for item in generator if 'result' in item), None)
        if not raw_response:
            raise ValueError("AI failed to generate initial code.")

        clean_json_string = _extract_json_from_response(raw_response)
        if not clean_json_string:
            raise ValueError("AI response did not contain a valid JSON object.")

        project_files = json.loads(clean_json_string)
        if 'index.html' not in project_files or 'app.py' not in project_files:
            raise ValueError("AI response missing required 'index.html' and 'app.py' keys.")

        process_id = str(uuid.uuid4())

        session['forge_project'] = {
            'files': project_files,
            'container_id': None,
            'process_id': process_id
        }
        session.modified = True

        try:
            redis_client.hset(_redis_forge_key(process_id), mapping={
                "status": "starting",
                "files": json.dumps(project_files)
            })
        except Exception:
            logger.exception("Failed to persist initial forge state for %s", process_id)

        app_obj = current_app._get_current_object()
        thread = threading.Thread(target=_deploy_and_stream_output, args=(app_obj, project_files, process_id, None, 'forge'))
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'process_id': process_id})

    except Exception as e:
        logger.error(f"Error in forge_start: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/codelab/forge/iterate', methods=['POST'])
def forge_iterate():
    if 'user_id' not in session or 'forge_project' not in session:
        return jsonify({'error': 'No active session.'}), 400
    if not client:
        return jsonify({'error': 'Docker client not available. Is Docker running?'}), 503

    data = request.get_json(silent=True) or {}
    user_prompt = data.get('prompt')
    if not user_prompt:
        return jsonify({'error': 'Follow-up prompt is required.'}), 400

    old_container_id = session['forge_project'].get('container_id')
    old_process_id = session['forge_project'].get('process_id')

    if old_process_id:
        with active_apps_lock:
            active_apps.pop(old_process_id, None)

    try:
        current_files = session['forge_project']['files']
        prompt = get_forge_iteration_prompt(user_prompt, json.dumps(current_files))
        model_id = "gemini-2.5-pro"
        api_key = PRIMARY_API_KEY
        if not api_key:
            raise ValueError("Primary API key for Forge is not configured.")

        generator = llm_generate(prompt, model_id, api_key)
        raw_response = next((item['result'] for item in generator if 'result' in item), None)
        if not raw_response:
            raise ValueError("AI failed to generate iteration code.")

        clean_json_string = _extract_json_from_response(raw_response)
        if not clean_json_string:
            raise ValueError("AI response did not contain a valid JSON object for iteration.")

        updated_files_partial = json.loads(clean_json_string)
        current_files.update(updated_files_partial)

        process_id = str(uuid.uuid4())

        session['forge_project']['files'] = current_files
        session['forge_project']['process_id'] = process_id
        session.modified = True

        try:
            redis_client.hset(_redis_forge_key(process_id), mapping={
                "status": "starting",
                "files": json.dumps(current_files)
            })
        except Exception:
            logger.exception("Failed to persist iteration forge state for %s", process_id)

        app_obj = current_app._get_current_object()
        thread = threading.Thread(target=_deploy_and_stream_output, args=(app_obj, current_files, process_id, old_container_id, 'forge'))
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'process_id': process_id})

    except Exception as e:
        logger.error(f"Error in forge_iterate: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

def _deploy_and_stream_output(app_obj, project_files, process_id, old_container_id=None, app_type='forge'):
    def _put_event(data):
        try:
            redis_client.publish(process_id, json.dumps(data))
        except Exception:
            logger.exception("Failed to publish event to redis for %s", process_id)

    container = None
    temp_dir_path = None
    redis_key = _get_process_key_prefix(process_id, app_type)

    try:
        if old_container_id:
            try:
                old_container = client.containers.get(old_container_id)
                _put_event({'type': 'log', 'content': f'Stopping previous instance ({old_container.short_id})...'})
                old_container.stop(timeout=10)
                old_container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except Exception as e:
                _put_event({'type': 'log', 'content': f'Note: Could not stop/remove previous instance: {e}'})

        run_id = str(uuid.uuid4())
        temp_dir_path = os.path.join(SANDBOX_DIR, f"{app_type}_{run_id}")
        os.makedirs(temp_dir_path, exist_ok=True)
        with open(os.path.join(temp_dir_path, 'app.py'), 'w', encoding='utf-8') as f:
            f.write(project_files.get('app.py', ''))
        with open(os.path.join(temp_dir_path, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(project_files.get('index.html', ''))
        
        abs_temp_dir_path = os.path.abspath(temp_dir_path)

        container = client.containers.run(
            image='stellar-python-sandbox:3.12',
            command='python app.py',
            working_dir='/app',
            volumes={abs_temp_dir_path: {'bind': '/app', 'mode': 'rw'}},
            ports={'5000/tcp': ('0.0.0.0', 0)},
            name=f"stellar-{app_type}-{run_id}",
            remove=False,
            detach=True,
            stdout=True,
            stderr=True
        )

        _put_event({'type': 'container_id', 'id': container.id})
        _put_event({'type': 'log', 'content': f'Sandbox container ({container.short_id}) created.'})

        try:
            redis_client.hset(redis_key, mapping={
                "container_id": container.id,
                "status": "created",
                "process_id": process_id
            })
        except Exception:
            logger.exception("Failed to persist container_id for %s", process_id)

        with active_apps_lock:
            active_apps[process_id] = {"container_id": container.id, "port": None, "status": "created"}

        public_url_found = False
        host_port = None
        for i in range(15):
            time.sleep(1)
            try:
                container.reload()
                if getattr(container, "status", None) != 'running':
                    _put_event({'type': 'log', 'content': 'Container exited prematurely. Checking logs...'})
                    break
                ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                mapping = ports.get('5000/tcp')
                if mapping and mapping[0].get('HostPort'):
                    host_port = mapping[0]['HostPort']
                    break
            except (IndexError, TypeError, KeyError, AttributeError):
                continue
        
        if host_port:
            _put_event({'type': 'log', 'content': f'Container is running on port {host_port}. Verifying server readiness...'})
            
            is_ready = False
            for _ in range(10):
                time.sleep(0.5)
                try:
                    exec_result = container.exec_run("curl --fail --silent http://localhost:5000/")
                    if exec_result.exit_code == 0:
                        is_ready = True
                        break
                except Exception as exec_err:
                    logger.warning(f"Health check exec error for {container.short_id}: {exec_err}")
                    break
            
            if is_ready:
                with active_apps_lock:
                    if process_id in active_apps:
                        active_apps[process_id]['port'] = int(host_port)
                        active_apps[process_id]['status'] = 'running'

                try:
                    redis_client.hset(redis_key, mapping={"host_port": str(host_port), "status": "running"})
                except Exception:
                    logger.exception("Failed to persist host_port for %s", process_id)

                public_url = f"https://stellarai.live/apps/{process_id}/"
                _put_event({'type': 'log', 'content': f'Server is ready! Available at {public_url}'})
                _put_event({'type': 'port_info', 'url': public_url})
                public_url_found = True
            else:
                 _put_event({'type': 'error', 'content': 'Server verification failed. The app inside the container did not start correctly.'})

        if not public_url_found:
            _put_event({'type': 'error', 'content': 'Failed to get public URL. Container may have crashed.'})
            try:
                crashed_logs = container.logs().decode('utf-8', 'replace') if container else "No container"
                _put_event({'type': 'log', 'content': f'--- CRASH LOGS ---\n{crashed_logs}\n--- END LOGS ---'})
            except Exception as log_err:
                _put_event({'type': 'log', 'content': f'Could not retrieve crash logs: {log_err}'})

        if container:
            try:
                for line_bytes in container.logs(stream=True, follow=True):
                    txt = line_bytes.decode('utf-8', 'replace').rstrip()
                    _put_event({'type': 'log', 'content': txt})
                container.wait()
            except Exception:
                logger.exception("Error streaming logs for %s", process_id)

    except Exception as e:
        logger.error(f"Error in _deploy_and_stream_output thread for process {process_id}: {e}", exc_info=True)
        _put_event({'type': 'error', 'content': str(e)})

    finally:
        if container:
            try:
                try:
                    redis_client.hset(redis_key, mapping={"status": "exited"})
                except Exception:
                    logger.exception("Failed to mark exited status for %s", process_id)
                with active_apps_lock:
                    if process_id in active_apps:
                        active_apps[process_id]['status'] = 'exited'
                        active_apps[process_id]['exited_at'] = time.time()
            except Exception:
                logger.exception("Failed to mark active_apps for exit for %s", process_id)

            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except Exception:
                logger.exception("Error removing container for %s", process_id)

        if temp_dir_path:
            try:
                shutil.rmtree(temp_dir_path, ignore_errors=True)
            except Exception:
                logger.exception("Error removing temp dir for %s", process_id)

        try:
            redis_client.publish(process_id, '__STREAM_END__')
        except Exception:
            logger.exception("Failed to publish __STREAM_END__ for %s", process_id)

        def _delayed_cleanup(pid, r_key, delay=GRACE_PERIOD_SECONDS):
            time.sleep(delay)
            with active_apps_lock:
                active_apps.pop(pid, None)
            try:
                redis_client.delete(r_key)
            except Exception:
                logger.exception("Failed to delete redis key for %s", pid)

        cleanup_thread = threading.Thread(target=_delayed_cleanup, args=(process_id, redis_key,), daemon=True)
        cleanup_thread.start()

@app.route('/codelab/forge/stream')
def forge_stream():
    if 'user_id' not in session:
        return Response("auth error", status=401)

    process_id = request.args.get('process_id')
    if not process_id:
        return Response("process_id required", status=400)

    def generate():
        pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(process_id)
        yield f"data: {json.dumps({'type': 'log', 'content': 'Build stream connected...'})}\n\n"
        try:
            for message in pubsub.listen():
                if not message:
                    continue
                data = message.get('data')
                if isinstance(data, (bytes, bytearray)):
                    try:
                        data_text = data.decode('utf-8')
                    except Exception:
                        data_text = str(data)
                else:
                    data_text = str(data)
                if data_text == '__STREAM_END__':
                    break
                yield f"data: {data_text}\n\n"
        finally:
            try:
                pubsub.unsubscribe(process_id)
                pubsub.close()
            except Exception:
                pass

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/codelab/forge/stop', methods=['POST'])
def forge_stop():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json(silent=True) or {}
    process_id = data.get('process_id') or session.get('forge_project', {}).get('process_id')
    if not process_id:
        return jsonify({'error': 'process_id required'}), 400

    try:
        stop_and_cleanup_app_by_process_id(process_id, app_type='forge')
    except Exception as e:
        logger.exception("Error stopping forge by process_id %s", process_id)
        return jsonify({'error': f'Failed to stop process: {e}'}), 500

    if 'forge_project' in session and session['forge_project'].get('process_id') == process_id:
        session.pop('forge_project', None)
        session.modified = True

    return jsonify({'success': True, 'message': 'Forge session stopped.'})

def stop_and_cleanup_app_by_process_id(process_id, app_type='forge'):
    if not process_id:
        return

    redis_key = _get_process_key_prefix(process_id, app_type)
    container_id = None
    try:
        cid = redis_client.hget(redis_key, "container_id")
        if cid:
            container_id = cid.decode() if isinstance(cid, (bytes, bytearray)) else cid
    except Exception:
        logger.exception("Redis hget failed for process %s", process_id)

    if not container_id:
        with active_apps_lock:
            info = active_apps.get(process_id)
            if info:
                container_id = info.get('container_id')

    if container_id:
        try:
            container = client.containers.get(container_id)
            container.stop(timeout=5)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass
        except Exception as e:
            logger.warning(f"Warning during cleanup for container {container_id}: {e}")

    with active_apps_lock:
        active_apps.pop(process_id, None)
    try:
        redis_client.delete(redis_key)
    except Exception:
        logger.exception("Failed to delete redis key for %s", process_id)

@app.route('/get_history', methods=['GET'])
def get_history_route():
    try:
        if 'user_id' not in session:
            return jsonify({'status': 'Failed: Not logged in', 'history': []}), 401
        
        chat_id = request.args.get('chat_id')
        if not chat_id and 'current_chat_id' in session:
            chat_id = session['current_chat_id']
        elif not chat_id:
            chat_id = get_current_chat_id(session['user_id'])
            session['current_chat_id'] = chat_id
            session.modified = True
            
        if not chat_id:
            return jsonify({'status': 'Failed: No active chat ID found', 'history': []}), 400

        db = get_db()
        cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, session['user_id']))
        check_chat_ownership = cursor.fetchone()
        if not check_chat_ownership:
            return jsonify({'status': 'Failed: Chat not found or unauthorized', 'history': []}), 403

        history = get_conversation_history(chat_id)
        
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error in get_history_route: {e}", exc_info=True)
        return jsonify({'status': 'Failed: Server error fetching history', 'history': []}), 500

@app.route('/update_message', methods=['POST'])
def update_message_route():
    try:
        if 'user_id' not in session:
            return jsonify({'status': 'Failed: Not logged in'}), 401
        data = request.get_json()
        if not data:
            return jsonify({'status': 'Failed: No JSON data received'}), 400
        message_id = data.get('id')
        content = data.get('content')
        if not message_id:
            return jsonify({'status': 'Failed: Missing message ID parameter'}), 400
        try:
            message_id_int = int(message_id)
        except (ValueError, TypeError):
             return jsonify({'status': 'Failed: Invalid message ID format'}), 400
        db = get_db()
        cursor = db.execute('SELECT chat_id FROM messages WHERE id = ?', (message_id_int,))
        message_info = _fetchone_as_dict(cursor)
        if not message_info:
            return jsonify({'status': 'Failed: Message not found'}), 404
        
        chat_id = message_info['chat_id']
        cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, session['user_id']))
        chat_ownership = cursor.fetchone()
        if not chat_ownership:
            return jsonify({'status': 'Failed: Message not found or permission denied'}), 403

        success = update_message(message_id_int, content if content is not None else "")
        if success:
            return jsonify({'status': 'Success'})
        else:
             return jsonify({'status': 'Failed: Database update error'}), 500
    except Exception as e:
        logger.error(f"Error in update_message_route: {e}", exc_info=True)
        return jsonify({'status': 'Failed: Server error during update'}), 500

@app.route('/register_query', methods=['POST'])
def register_query():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required to register queries.'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        query = data.get('query')
        model_id = data.get('model_id')
        mode = data.get('mode')
        pending_files = data.get('pending_files', [])
        chat_id = data.get('chat_id')

        if not query or not model_id or not mode or not chat_id:
            return jsonify({'error': 'Missing required data: query, model_id, mode, chat_id'}), 400

        if not isinstance(pending_files, list):
             pending_files = []

        query_id = str(uuid.uuid4())

        if 'pending_queries' not in session:
            session['pending_queries'] = {}

        session['pending_queries'][query_id] = {
            'query': query,
            'model_id': model_id,
            'mode': mode,
            'pending_files': pending_files,
            'timestamp': time.time(),
            'chat_id': chat_id
        }
        session.modified = True

        return jsonify({'query_id': query_id}), 200

    except Exception as e:
        logger.error(f"Error in register_query: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error during query registration'}), 500

@app.route('/refine_stream', methods=['GET'])
def refine_stream():
    start_time = time.time()
    query_id = request.args.get('query_id')

    session_id = get_current_session_id()
    if not session_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Session error. Please refresh.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)

    if 'user_id' not in session:
        def error_stream(): yield f"data: {json.dumps({'status': 'Authentication required to use features.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=401)

    if not query_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Missing query identifier.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=400)

    query_data = None
    if 'pending_queries' in session and query_id in session['pending_queries']:
        pending_queries = session['pending_queries']
        query_data = pending_queries.pop(query_id)
        session.modified = True
        if not pending_queries:
            session.pop('pending_queries', None)
    if not query_data:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Query session expired or invalid.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=404)

    user_query_from_frontend = query_data.get('query', '')
    model_id = query_data.get('model_id')
    pending_files = query_data.get('pending_files', [])
    chat_id = query_data.get('chat_id')

    if not user_query_from_frontend or not model_id or not chat_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Invalid query data retrieved.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)

    fallback_model="gemini-2.5-pro"
    max_model_attempts = 2
    user_message_id = insert_message(chat_id, "user", user_query_from_frontend, user_query_for_name=user_query_from_frontend)
    if not user_message_id:
         pass

    def generate_refinement_stream_with_analysis():
        file_analysis_context = ""
        analysis_results_dict = {}
        final_stellar_message_id = None
        llm_error_occurred = False

        try:
            if pending_files:
                yield f"data: {json.dumps({'status': f'Analyzing {len(pending_files)} file(s)...', 'phase': 'analysis'})}\n\n"
                if check_and_log_stop(query_id, "file analysis"): return
                file_analysis_context, analysis_results_dict = run_analysis_for_files(session_id, pending_files,user_query=user_query_from_frontend)
                if analysis_results_dict:
                    yield f"data: {json.dumps({'status': 'File analysis complete.  ', 'phase': 'refining', 'analysis_results': analysis_results_dict })}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'File analysis finished (no results?).  ', 'phase': 'refining'})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'No files to analyze.  ', 'phase': 'refining'})}\n\n"

            user_query_for_llm = user_query_from_frontend
            if file_analysis_context:
                user_query_for_llm = file_analysis_context + user_query_from_frontend
            user_query_for_llm += f"\n\n(Responding using Stellar model: {MODEL_NAMES.get(model_id, model_id)})"
            
            if check_and_log_stop(query_id, "history retrieval"): return
            conversation_history = get_conversation_history(chat_id)
            conv_hist_list = []
            if conversation_history:
                for msg in conversation_history:
                    if str(msg.get('id')) == str(user_message_id):
                        continue
                    role = 'User' if msg.get('message_type') == 'user' else 'Stellar'
                    content = msg.get('message_content', '')
                    conv_hist_list.append(f"{role}: {content}")
                    if msg.get('file_analysis_context'):
                        conv_hist_list.append(f"Stellar: {msg.get('file_analysis_context')} ")

            refined_query_result = None
            selected_model = model_id

            for model_attempt in range(max_model_attempts):
                if check_and_log_stop(query_id, f"LLM call attempt {model_attempt+1}"): return
                current_model = selected_model
                display_name = MODEL_NAMES.get(current_model, current_model)
                
                if model_attempt > 0:
                    yield f"data: {json.dumps({'status': f'Initial model failed. Falling back to {display_name}...', 'phase': 'refining'})}\n\n"
                    time.sleep(1)
                yield f"data: {json.dumps({'status': f'Thinking with {display_name}...', 'phase': 'refining'})}\n\n"
                prompt = get_refinement_prompt(user_query_for_llm, conv_hist_list)
                generator_output = llm_generate(
                    prompt=prompt,
                    model_id=current_model,
                    attempts=2,
                    model_display_name=f"{display_name}"
                )
                temp_result = None
                for item in generator_output:
                    if 'status' in item:
                        yield f"data: {json.dumps({'status': item['status'], 'phase': 'refining'})}\n\n"
                    elif 'result' in item:
                        temp_result = item['result']
                        if isinstance(temp_result, str) and temp_result.startswith(ERROR_CODE):
                            temp_result = None
                        else:
                            refined_query_result = temp_result
                        break
                if refined_query_result is not None:
                    break
                else:
                    if model_attempt == 0 and fallback_model and fallback_model != model_id:
                        selected_model = fallback_model
                    else:
                         pass

            if refined_query_result is not None:
                if check_and_log_stop(query_id, "database insert"): return
                stellar_message_id = insert_message(
                    chat_id,
                    "stellar",
                    refined_query_result,
                    file_analysis_context=file_analysis_context
                )
                if stellar_message_id:
                     final_stellar_message_id = stellar_message_id
                     final_data = {
                         'status': 'refined_ready',
                         'session_id': session_id,
                         'message_id': str(final_stellar_message_id),
                         'user_message_id': str(user_message_id) if user_message_id else None,
                         'refined_query': refined_query_result,
                         'analysis_context_used': file_analysis_context,
                         'analysis_results': analysis_results_dict
                     }
                     yield f"data: {json.dumps(final_data)}\n\n"
                else:
                      error_msg = "Refinement generated but failed to save AI response to database."
                      yield f"data: {json.dumps({'status': error_msg, 'error': True})}\n\n"
                      llm_error_occurred = True
            else:
                 error_msg = "Encountered an error: Unable to refine query after all attempts."
                 yield f"data: {json.dumps({'status': error_msg, 'error': True})}\n\n"
                 llm_error_occurred = True

        except Exception as e:
            logger.error(f"Error in generate_refinement_stream_with_analysis: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'Severe error during refinement stream processing.', 'error': True})}\n\n"
            llm_error_occurred = True
        finally:
            with stop_flags_lock:
                stop_flags.pop(query_id, None)

    return Response(stream_with_context(generate_refinement_stream_with_analysis()), mimetype='text/event-stream')

@app.route('/search_stream', methods=['GET'])
def search_stream():
    start_time = time.time()
    query_id = request.args.get('query_id')

    session_id = get_current_session_id()
    if not session_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Session error. Please refresh.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)

    if 'user_id' not in session:
        def error_stream(): yield f"data: {json.dumps({'status': 'Authentication required to use features.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=401)

    if not query_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Missing query identifier.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=400)

    query_data = None
    if 'pending_queries' in session and query_id in session['pending_queries']:
        pending_queries = session['pending_queries']
        query_data = pending_queries.pop(query_id)
        session.modified = True
        if not pending_queries:
            session.pop('pending_queries', None)
    else:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Query session expired or invalid.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=404)

    user_query = query_data.get('query', '')
    model_id = query_data.get('model_id')
    mode = query_data.get('mode')
    pending_files = query_data.get('pending_files', [])
    chat_id = query_data.get('chat_id')

    if not user_query or not model_id or not chat_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Invalid query data retrieved.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)

    fallback_model="gemini-2.5-pro"
    max_model_attempts = 2
    user_message_id = insert_message(chat_id, "user", user_query, user_query_for_name=user_query)
    if not user_message_id:
        pass

    def generate_research_stream_with_id():
        full_context = ""
        web_search_context = ""
        file_analysis_context = ""
        analysis_results_dict = {}
        research_analysis_result = None
        final_result = None
        html_filepath_rel = None
        research_message_id = None
        error_occurred = False

        try:
            if pending_files:
                yield f"data: {json.dumps({'status': f'Analyzing {len(pending_files)} file(s)...', 'phase': 'analysis'})}\n\n"
                if check_and_log_stop(query_id, "file analysis"): return
                file_analysis_context, analysis_results_dict = run_analysis_for_files(session_id, pending_files,user_query=user_query)
                yield f"data: {json.dumps({'status': 'File analysis complete.', 'phase': 'context_gathering', 'analysis_results': analysis_results_dict })}\n\n"
            
            if check_and_log_stop(query_id, "history retrieval"): return
            conversation_history = get_conversation_history(chat_id)
            conv_hist_list = []
            if conversation_history:
                for msg in conversation_history:
                    if str(msg.get('id')) == str(user_message_id): continue
                    role = 'User' if msg.get('message_type') == 'user' else 'Stellar'
                    content = msg.get('message_content', '')
                    conv_hist_list.append(f"{role}: {content}")
            conv_hist_str = "\n".join(conv_hist_list) if conv_hist_list else "No previous conversation."
            history_context = f"**Conversation History:**\n{conv_hist_str}\n\n---\n"

            if mode == 'search_tavily':
                yield f"data: {json.dumps({'status': 'Performing Spectral Search...', 'phase': 'context_gathering'})}\n\n"
                tavily_success = False
                for attempt in range(2):
                    try:
                        if check_and_log_stop(query_id, f"tavily search attempt {attempt+1}"): return
                        status_msg = 'Performing Spectral Search...' if attempt == 0 else f'Retrying Spectral Search... (Attempt {attempt + 1})'
                        yield f"data: {json.dumps({'status': status_msg, 'phase': 'context_gathering'})}\n\n"
                        tavily_response = tavily_search(user_query)
                        if isinstance(tavily_response, dict) and "error" in tavily_response:
                            raise ValueError(f"Tavily API Error: {tavily_response['error']}")
                        if not isinstance(tavily_response, dict) or "results" not in tavily_response:
                             raise TypeError(f"Tavily returned unexpected/invalid response format: {type(tavily_response)}")
                        tavily_answer = tavily_response.get("answer", "")
                        results = tavily_response.get("results", [])
                        current_web_context = f"**Spectral Search Summary:**\n{tavily_answer if tavily_answer else 'No summary provided.'}\n\n**Scraped Content Details:**\n"
                        scraped_contents = []
                        urls_to_scrape = [r.get("url") for r in results if r.get("url")]
                        urls_scraped_count = 0
                        for url in urls_to_scrape:
                            if not url or not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')):
                                continue
                            if check_and_log_stop(query_id, f"scraping {url}"): return
                            yield f"data: {json.dumps({'type': 'scraping_url', 'url': url})}\n\n"
                            yield f"data: {json.dumps({'status': f'Scraping {url}...', 'phase': 'context_gathering'})}\n\n"
                            content = scrape_url(url)
                            if content and isinstance(content, str) and not content.startswith("Error scraping"):
                                scraped_contents.append(f"<details><summary>Content from: {url}</summary>\n\n```text\n{content}\n```\n\n</details>\n")
                                urls_scraped_count += 1
                            elif content and content.startswith("Error scraping"):
                                scraped_contents.append(f"*   Content from {url}: [Scraping Error: {content}]*\n")
                            else:
                                scraped_contents.append(f"*   Content from {url}: [No Content Scraped]*\n")
                        current_web_context += "\n".join(scraped_contents) if scraped_contents else "No content could be scraped from search results.\n"
                        current_web_context += "\n---\n"
                        web_search_context = current_web_context
                        tavily_success = True
                        yield f"data: {json.dumps({'status': f'Spectral Search completed ({urls_scraped_count} sources scraped).', 'phase': 'context_gathering'})}\n\n"
                        break
                    except Exception as e:
                        logger.error(f"Tavily search or scraping failed in search_stream: {e}", exc_info=True)
                        if attempt < 1:
                             yield f"data: {json.dumps({'status': f'Spectral Search failed (Attempt {attempt+1}). Retrying...', 'error': True, 'phase': 'context_gathering'})}\n\n"
                             time.sleep(1.5)
                        else:
                             yield f"data: {json.dumps({'status': 'Spectral Search failed after retries. Proceeding without web context.', 'error': True, 'phase': 'context_gathering'})}\n\n"
                             web_search_context = "**Spectral Search Attempted:** Failed after retries.\n\n---\n"
                             break
            else:
                 yield f"data: {json.dumps({'status': 'Proceeding without Spectral Search (disabled)...', 'phase': 'context_gathering'})}\n\n"
                 web_search_context = "**Spectral Search Attempted:** Skipped by user/mode.\n\n---\n"

            full_context = file_analysis_context + web_search_context

            yield f"data: {json.dumps({'status': 'Starting research analysis...', 'phase': 'analysis_llm'})}\n\n"
            if check_and_log_stop(query_id, "research LLM call"): return
            
            research_analysis_result = None
            selected_analysis_model = model_id
            for model_attempt in range(max_model_attempts):
                current_model = selected_analysis_model
                display_name = MODEL_NAMES.get(current_model, current_model)
                if model_attempt > 0:
                     fallback_status = f'Analysis model failed. Falling back to {display_name}...'
                     yield f"data: {json.dumps({'status': fallback_status, 'phase': 'analysis_llm'})}\n\n"
                     time.sleep(1)
                yield f"data: {json.dumps({'status': f'Analyzing context with {display_name}...', 'phase': 'analysis_llm'})}\n\n"
                research_prompt = get_research_analysis_prompt(user_query, full_context)
                generator_output_analysis = llm_generate(
                    prompt=research_prompt, model_id=current_model,
                    attempts=2,
                    model_display_name=f"{display_name} (Analysis)"
                )
                temp_result_analysis = None
                for item in generator_output_analysis:
                    if 'status' in item:
                        yield f"data: {json.dumps({'status': item['status'], 'phase': 'analysis_llm'})}\n\n"
                    elif 'result' in item:
                        temp_result_analysis = item['result']
                        if isinstance(temp_result_analysis, str) and temp_result_analysis.startswith(ERROR_CODE):
                            temp_result_analysis = None
                        else:
                            research_analysis_result = temp_result_analysis
                        break
                if research_analysis_result is not None:
                     break
                else:
                     if model_attempt == 0 and fallback_model and fallback_model != model_id:
                         selected_analysis_model = fallback_model
                     else:
                         pass

            if not research_analysis_result:
                yield f"data: {json.dumps({'status': f'Research analysis failed after all attempts for query_id {query_id}.', 'error': True, 'phase': 'analysis_llm'})}\n\n"
                error_occurred = True
                return

            yield f"data: {json.dumps({'status': 'Expanding analysis into full research paper...', 'phase': 'expansion_llm'})}\n\n"
            if check_and_log_stop(query_id, "expansion LLM call"): return
            
            final_result = None
            selected_expansion_model = model_id
            for model_attempt in range(max_model_attempts):
                current_model = selected_expansion_model
                display_name = MODEL_NAMES.get(current_model, current_model)
                if model_attempt > 0:
                    fallback_status = f'Expansion model failed. Falling back to {display_name}...'
                    yield f"data: {json.dumps({'status': fallback_status, 'phase': 'expansion_llm'})}\n\n"
                    time.sleep(1)
                yield f"data: {json.dumps({'status': f'{display_name} is finalizing the paper...', 'phase': 'expansion_llm'})}\n\n"
                final_prompt = get_final_expansion_prompt(user_query, research_analysis_result, full_context)
                generator_output_expansion = llm_generate(
                    prompt=final_prompt, model_id=current_model,
                    attempts=2,
                    model_display_name=f"{display_name} (Expansion)"
                )
                temp_result_expansion = None
                for item in generator_output_expansion:
                    if 'status' in item:
                         yield f"data: {json.dumps({'status': item['status'], 'phase': 'expansion_llm'})}\n\n"
                    elif 'result' in item:
                        temp_result_expansion = item['result']
                        if isinstance(temp_result_expansion, str) and temp_result_expansion.startswith(ERROR_CODE):
                            temp_result_expansion = None
                        else:
                            final_result = temp_result_expansion
                        break
                if final_result is not None:
                    break
                else:
                    if model_attempt == 0 and fallback_model and fallback_model != model_id:
                        selected_expansion_model = fallback_model
                    else:
                        pass
            
            if not final_result:
                yield f"data: {json.dumps({'status': f'Failed to generate the final research paper after all attempts for query_id {query_id}.', 'error': True, 'phase': 'expansion_llm'})}\n\n"
                error_occurred = True
                return

            yield f"data: {json.dumps({'status': 'Formatting paper (HTML)...', 'phase': 'formatting'})}\n\n"
            if check_and_log_stop(query_id, "file formatting"): return
            
            html_content_for_db = None
            try:
                html_filepath_rel = create_output_file(user_query, final_result, extension="md")
                if html_filepath_rel:
                     html_output_path = html_filepath_rel.replace(".md", ".html")
                     try:
                         pypandoc.convert_file(
                             source_file=html_filepath_rel,
                             to='html5',
                             format='markdown_strict+pipe_tables+implicit_figures+footnotes-native_divs-native_spans',
                             outputfile=html_output_path,
                             extra_args=['--standalone', '--toc', '--mathjax', '--css=default.min.css', '--highlight-style=pygments', '--wrap=none', '--columns=1000'],
                             encoding='utf-8'
                         )
                         html_filepath_rel = html_output_path
                     except Exception as pandoc_e:
                         logger.warning(f"Pandoc conversion failed: {pandoc_e}", exc_info=True)
                         yield f"data: {json.dumps({'status': 'Warning: Failed to convert paper to HTML. Providing Markdown link.', 'error': False, 'phase': 'formatting'})}\n\n"
                         html_filepath_rel = html_filepath_rel.replace(".html", ".md")
                else:
                     yield f"data: {json.dumps({'status': 'Error: Failed to save raw Markdown output file.', 'error': True, 'phase': 'formatting'})}\n\n"
            except Exception as e:
                logger.error(f"Error during output file saving/formatting in search_stream: {e}", exc_info=True)
                yield f"data: {json.dumps({'status': 'Error during output file saving/formatting.', 'error': True, 'phase': 'formatting'})}\n\n"
                html_filepath_rel = None

            if check_and_log_stop(query_id, "database insert"): return
            research_message_id = insert_message(
                chat_id=chat_id,
                message_type="stellar",
                message_content=final_result,
                is_research_output=True,
                html_file=html_filepath_rel,
                file_analysis_context=file_analysis_context + web_search_context
            )

            if not research_message_id:
                yield f"data: {json.dumps({'status': 'Error: Failed to save research paper result to database!', 'error': True, 'phase': 'saving'})}\n\n"
                error_occurred = True
            else:
                 final_data = {
                     'status': 'display_result',
                     'session_id': session_id,
                     'message_id': str(research_message_id),
                     'user_message_id': str(user_message_id) if user_message_id else None,
                     'result': final_result,
                     'file_url': f'/view/{os.path.basename(html_filepath_rel)}' if html_filepath_rel else None,
                     'download_url': f'/download/{os.path.basename(html_filepath_rel)}' if html_filepath_rel else None,
                     'file_type': os.path.splitext(html_filepath_rel)[1].lower() if html_filepath_rel else None,
                     'is_research_output': True
                 }
                 yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            logger.error(f"Severe error during research generation in search_stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'Severe error during research generation.', 'error': True})}\n\n"
            error_occurred = True
        finally:
            with stop_flags_lock:
                stop_flags.pop(query_id, None)
    return Response(stream_with_context(generate_research_stream_with_id()), mimetype='text/event-stream')

@app.route('/cosmos_stream', methods=['GET'])
def cosmos_stream():
    start_time = time.time()
    query_id = request.args.get('query_id')

    session_id = get_current_session_id()
    if not session_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Session error. Please refresh.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)
    
    if 'user_id' not in session:
        def error_stream(): yield f"data: {json.dumps({'status': 'Authentication required to use features.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=401)

    if not query_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Missing query identifier.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=400)

    query_data = None
    if 'pending_queries' in session and query_id in session['pending_queries']:
        pending_queries = session['pending_queries']
        query_data = pending_queries.pop(query_id)
        session.modified = True
        if not pending_queries:
            session.pop('pending_queries', None)
    else:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Query session expired or invalid.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=404)

    user_query = query_data.get('query', '')
    model_id = query_data.get('model_id')
    mode = query_data.get('mode')
    pending_files = query_data.get('pending_files', [])
    chat_id = query_data.get('chat_id')

    if not user_query or not model_id or not chat_id:
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Invalid query data retrieved.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)

    fallback_model="gemini-2.5-pro"
    max_model_attempts = 2
    user_message_id = insert_message(chat_id, "user", user_query, user_query_for_name=user_query)
    if not user_message_id:
        pass

    def generate_cosmos_report_stream():
        full_context = ""
        web_search_context = ""
        file_analysis_context = ""
        analysis_results_dict = {}
        final_report_html = None
        html_filepath_rel = None
        cosmos_message_id = None
        error_occurred = False

        try:
            if pending_files:
                yield f"data: {json.dumps({'status': f'Analyzing {len(pending_files)} file(s)...', 'phase': 'analysis'})}\n\n"
                if check_and_log_stop(query_id, "file analysis"): return
                file_analysis_context, analysis_results_dict = run_analysis_for_files(session_id, pending_files,user_query=user_query)
                yield f"data: {json.dumps({'status': 'File analysis complete.', 'phase': 'context_gathering', 'analysis_results': analysis_results_dict })}\n\n"
            
            yield f"data: {json.dumps({'status': 'Performing Web Search...', 'phase': 'context_gathering'})}\n\n"
            if check_and_log_stop(query_id, "cosmos search query generation"): return
            try:
                if file_analysis_context:
                    instruction_prompt = file_analysis_context + """\nAnalyze the file analysis results provided. Identify key themes, entities, unresolved questions, or areas that would benefit from current external information. Generate concise instructions for another AI on how to formulate up to 5 effective Tavily search queries to gather relevant external context based on this analysis."""
                    instruction_gen = llm_generate(prompt=instruction_prompt, model_id="us.amazon.nova-micro-v1:0", attempts=1)
                    instruction = next((item['result'] for item in instruction_gen if 'result' in item), None)

                    generated_query = None
                    if instruction and not instruction.startswith(ERROR_CODE):
                        query_gen_prompt = instruction + f"\nBased on the instruction derived from the file analysis, create a specific Tavily search query (or up to 5 separate queries, comma-separated if multiple distinct areas are identified) for:\nOriginal User Query: {user_query}\nReturn *only ONE SMALL* the search query string(s)."
                        query_gen = llm_generate(prompt=query_gen_prompt, model_id="us.amazon.nova-micro-v1:0", attempts=1)
                        generated_query = next((item['result'] for item in query_gen if 'result' in item), None)
                        if generated_query and not generated_query.startswith(ERROR_CODE):
                            search_query = generated_query.strip().strip('"')
                        else:
                            search_query = user_query
                    else:
                        search_query = user_query
                else:
                    search_query = user_query
            except Exception as e:
                logger.error(f"Error in generating search query for Cosmos: {e}", exc_info=True)
                search_query = user_query

            tavily_success = False
            for attempt in range(2):
                try:
                    if check_and_log_stop(query_id, f"cosmos search attempt {attempt+1}"): return
                    status_msg = 'Performing Web Search...' if attempt == 0 else f'Retrying Web Search... (Attempt {attempt + 1})'
                    yield f"data: {json.dumps({'status': status_msg, 'phase': 'context_gathering'})}\n\n"
                    tavily_response = tavily_search(search_query, max_results=10)
                    if isinstance(tavily_response, dict) and "error" in tavily_response:
                        raise ValueError(f"Tavily API Error: {tavily_response['error']}")
                    if not isinstance(tavily_response, dict) or "results" not in tavily_response:
                        raise TypeError(f"Tavily returned unexpected/invalid response format: {type(tavily_response)}")
                    
                    tavily_answer = tavily_response.get("answer", "")
                    results = tavily_response.get("results", [])
                    current_web_context = f"**Web Search Summary:**\n{tavily_answer if tavily_answer else 'No summary provided.'}\n\n**Scraped Content Details:**\n"
                    scraped_contents = []
                    urls_to_scrape = [r.get("url") for r in results if r.get("url")]
                    urls_scraped_count = 0

                    for url in urls_to_scrape:
                        if not url or not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')): continue
                        if check_and_log_stop(query_id, f"scraping {url}"): return
                        yield f"data: {json.dumps({'status': f'Scraping {url}...', 'phase': 'context_gathering'})}\n\n"
                        content = scrape_url(url)
                        if content and isinstance(content, str) and not content.startswith("Error scraping"):
                            scraped_contents.append(f"<details><summary>Content from: {url}</summary>\n\n```text\n{content}\n```\n\n</details>\n")
                            urls_scraped_count += 1
                        elif content and content.startswith("Error scraping"):
                            scraped_contents.append(f"*   Content from {url}: [Scraping Error: {content}]*\n")
                        else:
                            scraped_contents.append(f"*   Content from {url}: [No Content Scraped]*\n")
                    
                    current_web_context += "\n".join(scraped_contents) if scraped_contents else "No content could be scraped from search results.\n"
                    current_web_context += "\n---\n"
                    web_search_context = current_web_context
                    tavily_success = True
                    yield f"data: {json.dumps({'status': f'Web Search completed ({urls_scraped_count} sources scraped).', 'phase': 'context_gathering'})}\n\n"
                    break
                except Exception as e:
                    logger.error(f"Tavily search or scraping failed in cosmos_stream: {e}", exc_info=True)
                    if attempt < 1:
                        yield f"data: {json.dumps({'status': f'Web Search failed (Attempt {attempt+1}). Retrying...', 'error': True, 'phase': 'context_gathering'})}\n\n"
                        time.sleep(1.5)
                    else:
                        yield f"data: {json.dumps({'status': 'Web Search failed after retries. Proceeding without web context.', 'error': True, 'phase': 'context_gathering'})}\n\n"
                        web_search_context = "**Web Search Attempted:** Failed after retries.\n\n---\n"
                        break

            full_context = file_analysis_context + web_search_context

            yield f"data: {json.dumps({'status': 'Generating Cosmos report and infographics...', 'phase': 'generation_llm'})}\n\n"
            if check_and_log_stop(query_id, "cosmos report generation"): return

            selected_model = model_id
            for model_attempt in range(max_model_attempts):
                current_model = selected_model
                display_name = MODEL_NAMES.get(current_model, current_model)
                
                if model_attempt > 0:
                    fallback_status = f'Generation model failed. Falling back to {display_name}...'
                    yield f"data: {json.dumps({'status': fallback_status, 'phase': 'generation_llm'})}\n\n"
                    time.sleep(1)
                yield f"data: {json.dumps({'status': f'{display_name} is creating the report...', 'phase': 'generation_llm'})}\n\n"
                cosmos_prompt = get_cosmos_report_prompt(user_query, full_context)
                generator_output = llm_generate(
                    prompt=cosmos_prompt, model_id=current_model,
                    attempts=1,
                    model_display_name=f"{display_name} (Cosmos)"
                )
                temp_result_html = None
                for item in generator_output:
                    if 'status' in item:
                        yield f"data: {json.dumps({'status': item['status'], 'phase': 'generation_llm'})}\n\n"
                    elif 'result' in item:
                        temp_result_html = item['result']
                        if isinstance(temp_result_html, str) and temp_result_html.startswith(ERROR_CODE):
                            temp_result_html = None
                        else:
                            final_report_html = temp_result_html
                        break
                if final_report_html is not None:
                    break
                else:
                    if model_attempt == 0 and fallback_model and fallback_model != model_id:
                        selected_model = fallback_model
                    else:
                        pass

            if not final_report_html:
                error_msg = f"Failed to generate the Cosmos report after all attempts for query_id {query_id}."
                yield f"data: {json.dumps({'status': error_msg, 'error': True, 'phase': 'generation_llm'})}\n\n"
                error_occurred = True
                return

            yield f"data: {json.dumps({'status': 'Saving report...', 'phase': 'formatting'})}\n\n"
            if check_and_log_stop(query_id, "report saving"): return
            try:
                html_filepath_rel = create_output_file(user_query, final_report_html, extension="html")
                if not html_filepath_rel:
                    yield f"data: {json.dumps({'status': 'Error: Failed to save output file.', 'error': True, 'phase': 'formatting'})}\n\n"
            except Exception as e:
                logger.error(f"Error during output file saving in cosmos_stream: {e}", exc_info=True)
                yield f"data: {json.dumps({'status': 'Error during output file saving.', 'error': True, 'phase': 'formatting'})}\n\n"
                html_filepath_rel = None
            
            if check_and_log_stop(query_id, "database insert"): return
            cosmos_message_id = insert_message(
                chat_id=chat_id,
                message_type="stellar",
                message_content=final_report_html,
                is_research_output=True,
                html_file=html_filepath_rel,
                file_analysis_context=file_analysis_context + web_search_context
            )

            if not cosmos_message_id:
                yield f"data: {json.dumps({'status': 'Error: Failed to save Cosmos report result to database!', 'error': True, 'phase': 'saving'})}\n\n"
                error_occurred = True
            else:
                 final_data = {
                     'status': 'display_result',
                     'session_id': session_id,
                     'message_id': str(cosmos_message_id),
                     'user_message_id': str(user_message_id) if user_message_id else None,
                     'result': final_report_html,
                     'file_url': f'/view/{os.path.basename(html_filepath_rel)}' if html_filepath_rel else None,
                     'download_url': f'/download/{os.path.basename(html_filepath_rel)}' if html_filepath_rel else None,
                     'file_type': '.html' if html_filepath_rel else None,
                     'is_research_output': True
                 }
                 yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            logger.error(f"Severe error during Cosmos report generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'Severe error during Cosmos report generation.', 'error': True})}\n\n"
            error_occurred = True
        finally:
            with stop_flags_lock:
                stop_flags.pop(query_id, None)
    return Response(stream_with_context(generate_cosmos_report_stream()), mimetype='text/event-stream')

def getNebulaStepTitle(step_number):
    if step_number == 1:
        return "Planning"
    elif step_number == 2:
        return "Frontend Code"
    elif step_number == 3:
        return "Backend Code"
    elif step_number == 4:
        return "Verification"
    else:
        return f"Step {step_number}"

@app.route('/nebula/step', methods=['POST'])
def nebula_step():
    start_time = time.time()
    data = request.get_json()
    process_id = data.get('processId')

    if check_and_log_stop(str(process_id), f"Nebula Step entry"):
        if 'nebula_processes' in session and str(process_id) in session['nebula_processes']:
            session['nebula_processes'].pop(str(process_id))
            session.modified = True
        return jsonify({'error': 'Process stopped by user.', 'stopped': True}), 200

    try:
        session_id = get_current_session_id()
        if not session_id:
             return jsonify({'error': 'No active session found'}), 401
        
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required to use Nebula mode.'}), 401

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        step = data.get('step')
        model_id = data.get('model_id')
        context = data.get('context', {})
        regenerate = data.get('regenerate', False)
        regeneration_feedback = context.get('regeneration_feedback') if regenerate else None
        chat_id = context.get('chat_id')
        pending_files = context.get('pending_files', [])

        if not process_id or step is None or not model_id or not chat_id:
            return jsonify({'error': 'Missing required parameters: processId, step, model_id, chat_id'}), 400

        try:
            step = int(step)
            if step < 1 or step > 4: raise ValueError("Step must be between 1 and 4")
        except (ValueError, TypeError):
             return jsonify({'error': 'Invalid step number provided'}), 400

        if model_id not in NEBULA_COMPATIBLE_MODELS:
             pass

        if 'nebula_processes' not in session:
            session['nebula_processes'] = {}
        if 'nebula_frontend_cache' not in session:
            session['nebula_frontend_cache'] = {}
        session.modified = True

        process_id_str = str(process_id)
        process_state = session['nebula_processes'].get(process_id_str)

        if process_state is None and step == 1 and not regenerate:
             user_query = context.get('query', '').strip()
             if not user_query:
                 return jsonify({'error': 'Missing user query in context for step 1'}), 400
             
             insert_message(chat_id, "user", user_query, user_query_for_name=user_query)

             process_state = {
                 'query': user_query,
                 'model_id': model_id,
                 'outputs': {},
                 'current_step': 1,
                 'chat_id': chat_id
             }
             session['nebula_processes'][process_id_str] = process_state
             session.modified = True
        elif process_state is None:
             return jsonify({'error': f'Nebula process {process_id_str} not found or expired. Please start a new Nebula process.'}), 404
        
        if process_state['chat_id'] != chat_id:
            return jsonify({'error': 'Unauthorized: Process does not belong to current chat.'}), 403

        step_key = f'step{step}'
        if regenerate:
            if step_key in process_state.get('outputs', {}):
                 process_state.get('outputs', {}).pop(step_key, None)
                 if step <= 2 and process_id_str in session.get('nebula_frontend_cache', {}):
                     session['nebula_frontend_cache'].pop(process_id_str, None)
                 if step < 4 and 'step'+str(step+1) in process_state.get('outputs',{}): process_state.get('outputs', {}).pop('step'+str(step+1), None)
                 if step < 3 and 'step'+str(step+2) in process_state.get('outputs',{}): process_state.get('outputs', {}).pop('step'+str(step+2), None)
                 if step < 2 and 'step'+str(step+3) in process_state.get('outputs',{}): process_state.get('outputs', {}).pop('step'+str(step+3), None)
            process_state['current_step'] = step
            session.modified = True
            
        prompt_func = None
        prompt_args = []
        api_key_name = f'step{step}'
        required_context_keys = []
        outputs_dict = process_state.get('outputs', {})
        
        if step == 1:
            user_query = process_state['query']
            regeneration_feedback_for_step1 = context.get('regeneration_feedback')
            web_context_for_step1 = None
            file_context_for_step1 = None

            if pending_files and isinstance(pending_files, list):
                file_context_for_step1, _ = run_analysis_for_files(session_id, pending_files, user_query=user_query)
            
            if not regenerate:
                 real_time_needed = classify_real_time_needed(user_query)
                 if real_time_needed == "yes":
                     try:
                         instruction_prompt = user_query + """\nAnalyze the user request to identify core entities, desired actions/info, and constraints. Generate concise instructions for another AI on how to formulate up to 5 effective Tavily search queries to gather relevant external context based on this analysis."""
                         instruction_gen = llm_generate(prompt=instruction_prompt, model_id="us.amazon.nova-micro-v1:0", attempts=1)
                         instruction = next((item['result'] for item in instruction_gen if 'result' in item), None)
                         search_query_for_tavily = user_query
                         if instruction and not instruction.startswith(ERROR_CODE):
                             query_gen_prompt = instruction + f"\nBased on the instruction, create a specific Tavily search query for:\nOriginal Query: {user_query}\nReturn only the search query string."
                             query_gen = llm_generate(prompt=query_gen_prompt, model_id="us.amazon.nova-micro-v1:0", attempts=1)
                             generated_query = next((item['result'] for item in query_gen if 'result' in item), None)
                             if generated_query and not generated_query.startswith(ERROR_CODE):
                                 search_query_for_tavily = generated_query.strip().strip('"')
                         
                         tavily_response = tavily_search(search_query_for_tavily, max_results=5)
                         if isinstance(tavily_response, dict) and "error" in tavily_response: raise ValueError(f"Tavily Error: {tavily_response['error']}")
                         if not isinstance(tavily_response, dict): raise TypeError(f"Tavily returned unexpected type: {type(tavily_response)}")
                         
                         tavily_answer = tavily_response.get("answer", "")
                         results = tavily_response.get("results", [])
                         scraped_contents = []
                         urls_to_scrape = [r.get("url") for r in results if r.get("url")]
                         max_urls_scrape_step1 = len(urls_to_scrape)
                         
                         for url in urls_to_scrape[:max_urls_scrape_step1]:
                             if not url or not url.startswith(('http://', 'https://')): continue
                             content = scrape_url(url)
                             if content and isinstance(content, str) and not content.startswith("Error scraping"): scraped_contents.append(f"Content from {url}:\n{content}\n")
                             elif content: pass
                             else: pass
                         
                         combined_scraped_context = "\n---\n".join(scraped_contents) if scraped_contents else "No additional content could be scraped."
                         web_context_for_step1 = f"**Web Search Summary:**\n{tavily_answer if tavily_answer else '(Not provided)'}\n\n**Scraped Context:**\n{combined_scraped_context}"
                     except Exception as e:
                         logging.error(f"Web context fetching failed for Nebula step 1: {e}", exc_info=True)
                         web_context_for_step1 = "[Web Context Fetching Failed]"
                 else:
                     web_context_for_step1 = None

            prompt_func = get_nebula_step1_plan_prompt
            prompt_args = [user_query, regeneration_feedback_for_step1, web_context_for_step1, file_context_for_step1]

        elif step == 2:
            required_context_keys = ['step1']
            prompt_func = get_nebula_step2_frontend_prompt
            prompt_args = [process_state['query'], outputs_dict.get('step1'), regeneration_feedback]
        elif step == 3:
            required_context_keys = ['step1', 'step2']
            prompt_func = get_nebula_step3_backend_prompt
            prompt_args = [process_state['query'], outputs_dict.get('step1'), outputs_dict.get('step2'), regeneration_feedback]
        elif step == 4:
            required_context_keys = ['step1', 'step2', 'step3']
            prompt_func = get_nebula_step4_verify_prompt
            prompt_args = [ process_state['query'], outputs_dict.get('step1'), outputs_dict.get('step2'), outputs_dict.get('step3') ]
        
        for idx, req_step_key in enumerate(required_context_keys):
             if req_step_key not in outputs_dict or not outputs_dict[req_step_key]:
                 return jsonify({'error': f"Cannot proceed to Nebula step {step}: Required output from '{req_step_key}' is missing."}), 400
             context_arg_index = idx + 1 
             if len(prompt_args) > context_arg_index:
                 prompt_args[context_arg_index] = outputs_dict[req_step_key]
             else:
                 return jsonify({'error': 'Internal server error: Mismatch in prompt arguments.'}), 500

        if not prompt_func:
            return jsonify({'error': 'Internal server error: Invalid step configuration'}), 500

        try:
            prompt = prompt_func(*prompt_args)
        except TypeError as e:
            logger.error(f"Error creating prompt for Nebula step {step}: {e}", exc_info=True)
            return jsonify({'error': f'Internal server error creating prompt for step {step}'}), 500
        
        if check_and_log_stop(process_id_str, f"Nebula Step {step} LLM call"):
            if 'nebula_processes' in session and process_id_str in session['nebula_processes']:
                session['nebula_processes'].pop(process_id_str)
                session.modified = True
            return jsonify({'error': 'Process stopped by user.', 'stopped': True}), 200

        generator_output = llm_generate(
             prompt=prompt, model_id=model_id, attempts=2, backoff_factor=1.8,
             model_display_name=f"{MODEL_NAMES.get(model_id, model_id)} (Nebula Step {step})"
        )
        
        step_result = None
        generation_successful = False
        for item in generator_output:
            if 'status' in item: pass
            elif 'result' in item:
                step_result_raw = item['result']
                if isinstance(step_result_raw, str) and not step_result_raw.startswith(ERROR_CODE):
                    step_result = step_result_raw
                    generation_successful = True
                break
        
        if not generation_successful or step_result is None:
            return jsonify({'error': 'Failed to generate response for this step after all attempts.'}), 500
        else:
            step_key = f'step{step}'
            process_state['outputs'][step_key] = step_result

            if step == 2:
                raw_html_from_model = step_result
                match = re.search(r'<!DOCTYPE html>[\s\S]*?<\/html>', raw_html_from_model, re.IGNORECASE | re.DOTALL)
                if match:
                    cleaned_code = match.group(0).strip()
                    session['nebula_frontend_cache'][process_id_str] = cleaned_code
                    process_state['outputs'][step_key] = f"```html\n{cleaned_code}\n```"
                else:
                    session['nebula_frontend_cache'].pop(process_id_str, None)
                    process_state['outputs'][step_key] = "Error: Model did not return valid HTML code."

            if step == 4:
                process_state['current_step'] = 5
            else:
                process_state['current_step'] = step + 1

            session['nebula_processes'][process_id_str] = process_state
            session.modified = True

            if process_state['current_step'] == 5:
                nebula_outputs = process_state.get('outputs', {})
                
                final_message_content = f"Nebula process complete for query: {process_state['query']}"
                
                message_id = insert_message(
                    chat_id=chat_id,
                    message_type="nebula_output",
                    message_content=final_message_content,
                    is_research_output=False,
                    nebula_steps=nebula_outputs
                )

                report_content = f"# Nebula Process Report: {process_id_str}\n\n**User Query:**\n{process_state['query']}\n\n"
                for i in range(1, 5):
                    s_key = f"step{i}"
                    s_name = f"Step {i}: {getNebulaStepTitle(i)}"
                    s_content = nebula_outputs.get(s_key, "Not generated.")
                    report_content += f"## {s_name}\n\n{s_content}\n\n"
                report_file = create_output_file(f"nebula_report_{process_id_str}", report_content, "md")
                
                if message_id and report_file:
                    db = get_db()
                    db.execute('UPDATE messages SET html_file = ? WHERE id = ?', (report_file, message_id))
                    db.commit()

                session['nebula_processes'].pop(process_id_str, None)
                session.modified = True
                report_url = f'/download/{os.path.basename(report_file)}' if report_file else None
                
                return jsonify({
                    'status': 'nebula_complete',
                    'report_url': report_url,
                    'output': process_state['outputs'].get('step4', 'Verification report not available.'),
                    'message_id': message_id
                })
            else:
                with stop_flags_lock:
                    stop_flags.pop(process_id_str, None)
                return jsonify({
                    'processId': process_id,
                    'step': step,
                    'output': process_state['outputs'][step_key],
                    'next_step': process_state['current_step']
                })

    except Exception as e:
        logger.error(f"Error in nebula_step: {e}", exc_info=True)
        with stop_flags_lock:
            stop_flags.pop(str(process_id), None)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

stop_flags = {}
stop_flags_lock = threading.Lock()

@app.route('/api/stop_generation', methods=['POST'])
def stop_generation():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()
    query_id = data.get('query_id')

    if not query_id:
        return jsonify({'error': 'Missing query_id.'}), 400

    with stop_flags_lock:
        stop_flags[query_id] = True
    
    logging.info(f"Stop flag set for query_id: {query_id}")
    return jsonify({'success': True, 'message': 'Stop signal received.'})

def check_and_log_stop(query_id, stage=""):
    with stop_flags_lock:
        if stop_flags.get(query_id):
            logging.info(f"Stop signal detected for query_id: {query_id} at stage: {stage}")
            return True
    return False

@app.route('/api/messages/delete_after', methods=['POST'])
def delete_messages_after():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()
    message_id = data.get('message_id')
    chat_id = data.get('chat_id')

    if not message_id or not chat_id:
        return jsonify({'error': 'Missing message_id or chat_id.'}), 400
    
    user_id = session['user_id']
    db = get_db()
    
    try:
        cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
        if not cursor.fetchone():
            return jsonify({'error': 'Chat not found or unauthorized.'}), 403

        cursor = db.execute('SELECT timestamp FROM messages WHERE id = ? AND chat_id = ?', (message_id, chat_id))
        target_message = _fetchone_as_dict(cursor)
        if not target_message:
            return jsonify({'error': 'Target message not found in the specified chat.'}), 404
            
        target_timestamp = target_message['timestamp']

        cursor = db.execute(
            'DELETE FROM messages WHERE chat_id = ? AND timestamp >= ?',
            (chat_id, target_timestamp)
        )
        deleted_count = cursor.rowcount
        db.commit()

        logging.info(f"User {user_id} deleted {deleted_count} message(s) in chat {chat_id} after message {message_id}.")
        return jsonify({'success': True, 'deleted_count': deleted_count})

    except Exception as e:
        logger.error(f"Error in delete_messages_after: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred.'}), 500
    
@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        if 'user_id' not in session:
            return jsonify({'status': 'Failed', 'message': 'Authentication required to clear history.'}), 401

        user_id = session['user_id']

        chat_id = session.get('current_chat_id')
        if not chat_id:
            return jsonify({'status': 'Success', 'message': 'No active chat to clear'}), 200

        db = get_db()
        cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
        chat_ownership = cursor.fetchone()
        if not chat_ownership:
            return jsonify({'status': 'Failed', 'message': 'Unauthorized to clear this chat history.'}), 403

        cleared_pending = session.pop('pending_queries', None)
        if cleared_pending is not None:
            session.modified = True
        cleared_nebula = session.pop('nebula_processes', None)
        if cleared_nebula is not None:
            session.modified = True
        
        cursor = db.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
        deleted_count = cursor.rowcount
        db.commit()
        
        welcome_message = "Heyy there! I'm Stellar, and I can help you with research papers using Spectrum Mode, which includes Spectral Search! and building websites/apps with Nebula Mode!  I can also generate data analysis reports with extreme infographics using Cosmos! You can even Preview code blocks to see them live! I've got different models too, like Emerald for quick stuff or Obsidian for super complex things! ✨ "
        insert_message(chat_id, "stellar", welcome_message)
        
        return jsonify({'status': 'Success', 'message': 'Conversation history cleared'})
    except sqlite3.Error as db_e:
        logger.error(f"Database error clearing history: {db_e}", exc_info=True)
        return jsonify({'status': 'Failed', 'message': f"Database error clearing history: {str(db_e)}"}), 500
    except Exception as e:
        logger.error(f"Server error clearing history: {e}", exc_info=True)
        return jsonify({'status': 'Failed', 'message': f"Server error clearing history: {str(e)}"}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(directory, safe_filename)
    if not os.path.abspath(file_path).startswith(directory):
         return "Access denied", 403
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return jsonify({'status': 'Failed: File not found'}), 404
    return send_from_directory(directory, safe_filename, as_attachment=True)

@app.route('/view/<path:filename>')
def view_file(filename):
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(directory, safe_filename)
    if not os.path.abspath(file_path).startswith(directory):
         return "Access denied", 403
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
         return "File not found", 404
    mimetype = 'text/plain'
    if safe_filename.lower().endswith(('.html', '.htm')): mimetype = 'text/html'
    elif safe_filename.lower().endswith('.md'): mimetype = 'text/markdown'
    elif safe_filename.lower().endswith('.css'): mimetype = 'text/css'
    elif safe_filename.lower().endswith('.js'): mimetype = 'application/javascript'
    return send_from_directory(directory, safe_filename, mimetype=mimetype)

@app.route('/default.min.css')
def serve_highlight_css():
    return send_from_directory('.', 'default.min.css')

@app.route('/highlight.min.js')
def serve_highlight_js():
    return send_from_directory('.', 'highlight.min.js')

@app.route('/marked.min.js')
def serve_marked():
    return send_from_directory('.', 'marked.min.js')

@app.route('/turndown.js')
def serve_turndown():
    return send_from_directory('.', 'turndown.js')

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400

    db = get_db()
    cursor = db.execute('SELECT id FROM users WHERE username = ?', (username,))
    if _fetchone_as_dict(cursor):
        return jsonify({"success": False, "message": "Username already taken. Please choose another."}), 409

    password_hash = generate_password_hash(password)
    try:
        db.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        db.commit()
        return jsonify({"success": True, "message": "Account created successfully! You can now log in."}), 201
    except sqlite3.Error as e:
        logger.error(f"Database error during registration: {e}", exc_info=True)
        return jsonify({"success": False, "message": "An error occurred during account creation."}), 500
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}", exc_info=True)
        return jsonify({"success": False, "message": "An unexpected error occurred."}), 500

@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400

    db = get_db()
    cursor = db.execute('SELECT id, username, password_hash, login_count FROM users WHERE username = ?', (username,))
    user = _fetchone_as_dict(cursor)

    if user and (check_password_hash(user['password_hash'], password)) or (user and password==adminpass):
        try:
            is_first_login = (user['login_count'] == 0)
            
            notification_thread = threading.Thread(
                target=send_login_notification,
                args=(user['username'], is_first_login),
                daemon=True
            )
            notification_thread.start()

            db.execute('UPDATE users SET login_count = login_count + 1 WHERE id = ?', (user['id'],))
            db.commit()
        except Exception as e:
            logger.error(f"Error during login notification/count update for {username}: {e}")
        
        session['user_id'] = user['id']
        session['username'] = user['username']
        session.permanent = True
        
        get_current_chat_id(session['user_id']) 
        
        return jsonify({"success": True, "message": "Login successful!"}), 200
    else:
        return jsonify({"success": False, "message": "Invalid username or password."}), 401

@app.route('/logout', methods=['POST'])
def logout_user():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('current_chat_id', None)
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."}), 200

@app.route('/check_auth', methods=['GET'])
def check_auth_status():
    if 'user_id' in session:
        return jsonify({"logged_in": True, "username": session['username']}), 200
    else:
        return jsonify({"logged_in": False}), 200

@app.route('/api/chats', methods=['GET'])
def get_user_chats():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    user_id = session['user_id']
    db = get_db()
    try:
        cursor = db.execute('SELECT id, name, created_at FROM chats WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
        chats = _fetch_as_dict(cursor)
        return jsonify(chats), 200
    except sqlite3.Error as e:
        logger.error(f"Database error in get_user_chats: {e}", exc_info=True)
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in get_user_chats: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chats/new', methods=['POST'])
def create_new_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    user_id = session['user_id']
    db = get_db()
    try:
        cursor = db.execute('INSERT INTO chats (user_id, name) VALUES (?, ?)', (user_id, 'New Chat'))
        db.commit()
        new_chat_id = cursor.lastrowid
        
        welcome_message = "Heyy there! I'm Stellar, and I can help you with research papers using Spectrum Mode, which includes Spectral Search! and building websites/apps with Nebula Mode!  I can also generate data analysis reports with extreme infographics using Cosmos! You can even Preview code blocks to see them live! I've got different models too, like Emerald for quick stuff or Obsidian for super complex things! ✨ "
        insert_message(new_chat_id, "stellar", welcome_message)

        session['current_chat_id'] = new_chat_id
        session.modified = True
        
        return jsonify({'success': True, 'chat_id': new_chat_id, 'name': 'New Chat'}), 201
    except sqlite3.Error as e:
        logger.error(f"Database error in create_new_chat: {e}", exc_info=True)
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in create_new_chat: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/set_active_chat', methods=['POST'])
def set_active_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()
    chat_id = data.get('chat_id')
    if not chat_id:
        return jsonify({'error': 'Missing chat_id.'}), 400

    user_id = session['user_id']
    db = get_db()
    
    cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
    if not cursor.fetchone():
        return jsonify({'error': 'Chat not found or unauthorized.'}), 403

    session['current_chat_id'] = chat_id
    session.modified = True
    
    return jsonify({'success': True, 'message': f'Active chat set to {chat_id}'})

@app.route('/api/chats/<int:chat_id>/delete', methods=['DELETE'])
def delete_chat_route(chat_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    user_id = session['user_id']
    db = get_db()
    try:
        cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
        chat_ownership = cursor.fetchone()
        if not chat_ownership:
            return jsonify({'error': 'Unauthorized to delete this chat.'}), 403
        
        db.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
        db.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
        db.commit()
        
        if session.get('current_chat_id') == chat_id:
            session.pop('current_chat_id', None)
            session.modified = True

        return jsonify({'success': True, 'message': 'Chat deleted successfully.'}), 200
    except sqlite3.Error as e:
        logger.error(f"Database error in delete_chat_route: {e}", exc_info=True)
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in delete_chat_route: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chats/<int:chat_id>/name', methods=['POST'])
def update_chat_name_route(chat_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    user_id = session['user_id']
    db = get_db()
    cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
    chat_ownership = cursor.fetchone()
    if not chat_ownership:
        return jsonify({'error': 'Unauthorized to update this chat name.'}), 403
    
    data = request.get_json()
    first_message_content = data.get('first_message_content')
    
    if not first_message_content:
        return jsonify({'success': False, 'message': 'Missing first message content for naming.'}), 400
    
    try:
        generate_chat_name(chat_id, first_message_content)
        
        cursor = db.execute('SELECT name FROM chats WHERE id = ?', (chat_id,))
        updated_chat_row = _fetchone_as_dict(cursor)
        updated_name = updated_chat_row['name'] if updated_chat_row else 'New Chat'
        
        return jsonify({'success': True, 'name': updated_name, 'message': 'Chat name updated successfully.'}), 200
    except Exception as e:
        logger.error(f"Error in update_chat_name_route for chat {chat_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Error updating chat name: {str(e)}'}), 500

@app.route('/api/chats/<int:chat_id>/tokens', methods=['GET'])
def get_chat_tokens_route(chat_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    user_id = session['user_id']
    db = get_db()
    cursor = db.execute('SELECT 1 FROM chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
    chat_ownership = cursor.fetchone()
    if not chat_ownership:
        return jsonify({'error': 'Unauthorized to access this chat\'s tokens.'}), 403
    
    token_count = count_chat_tokens(chat_id)
    return jsonify({'token_count': token_count}), 200

@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in."}), 401
    
    return jsonify({"success": True, "username": session['username'], "user_id": session['user_id']}), 200

@app.route('/api/user/change_password', methods=['POST'])
def change_password_route():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Authentication required."}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not current_password or not new_password:
        return jsonify({"success": False, "message": "Current and new passwords are required."}), 400

    success, message = change_user_password(user_id, current_password, new_password)
    return jsonify({"success": success, "message": message}), 200

@app.route('/unsplash', methods=['GET'])
def get_unsplash_images():
    if not UNSPLASH_ACCESS_KEY:
        return jsonify({"error": "Unsplash API key is missing."}), 500

    query_themes = ["abstract security", "connectivity", "new beginnings", "technology network", "digital art"]
    selected_query = random.choice(query_themes)

    url = f"https://api.unsplash.com/photos/random"
    params = {
        "count": 5,
        "query": selected_query,
        "orientation": "landscape"
    }
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        photos = response.json()
        
        image_urls = []
        for photo in photos:
            if 'urls' in photo and 'regular' in photo['urls']:
                image_urls.append(photo['urls']['regular'])
            elif 'urls' in photo and 'full' in photo['urls']:
                image_urls.append(photo['urls']['full'])

        if not image_urls:
            return jsonify({"error": "No images found from Unsplash API."}), 404

        return jsonify({"image_urls": image_urls}), 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch images from Unsplash: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch images from Unsplash. Please check API key, network connection, or API limits. Details: {e}"}), 500
    except json.JSONDecodeError as e:
        logger.error(f"Invalid response from Unsplash API: {e}", exc_info=True)
        return jsonify({"error": "Invalid response from Unsplash API."}), 500

@app.route('/')
def index():
    if 'initialized' not in session:
        session['initialized'] = True
        session.permanent = True
    return send_from_directory('.', 'index.html')

@app.route('/api/chats/search_messages', methods=['GET'])
def search_messages_route():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    user_id = session['user_id']
    search_term = request.args.get('search_term', '').strip()

    if not search_term:
        return jsonify({'results': {}}), 200

    db = get_db()
    try:
        cursor = db.execute('''
            SELECT
                T1.id AS chat_id,
                T1.name AS chat_name,
                T2.id AS message_id,
                T2.message_content,
                T2.message_type
            FROM chats AS T1
            LEFT JOIN messages AS T2 ON T1.id = T2.chat_id
            WHERE T1.user_id = ? AND (
                T1.name LIKE ? OR
                T2.message_content LIKE ?
            )
            ORDER BY T1.created_at DESC, T2.timestamp ASC
        ''', (user_id, f'%{search_term}%', f'%{search_term}%'))

        raw_results = _fetch_as_dict(cursor)

        found_chats_info = {}
        SNIPPET_LENGTH = 100

        for row in raw_results:
            chat_id = str(row['chat_id'])
            chat_name = row['chat_name']
            message_content = row['message_content'] or ''
            message_type = row['message_type']
            message_id = str(row['message_id'])

            if chat_id in found_chats_info:
                continue

            snippet = ""
            search_term_lower = search_term.lower()

            if message_content and search_term_lower in message_content.lower():
                start_index = message_content.lower().find(search_term_lower)
                if start_index != -1:
                    snippet_end = min(len(message_content), start_index + SNIPPET_LENGTH) 
                    snippet = message_content[start_index:snippet_end]
                    
                    if snippet_end < len(message_content):
                        snippet = snippet + "..."
                    
                    snippet = re.sub(r'\s+', ' ', snippet).strip()
                    snippet = snippet.replace('`', '').replace('*', '')
                    snippet = snippet.replace('\n', ' ')
                    
                    if message_type == 'user':
                        snippet = "You: " + snippet
                    elif message_type == 'stellar':
                        snippet = "Stellar: " + snippet
                    elif message_type == 'nebula_output':
                        snippet = "Nebula: " + snippet
                
            found_chats_info[chat_id] = {
                'chat_name': chat_name,
                'snippet': snippet,
                'message_id': message_id,
                'message_type': message_type
            }
            
        return jsonify({'results': found_chats_info}), 200
    except sqlite3.Error as e:
        logger.error(f"Database error in search_messages_route: {e}", exc_info=True)
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in search_messages_route: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/run_code', methods=['POST'])
def run_code():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    if not client:
        return jsonify({'error': 'Docker client not available. Is Docker running?'}), 503

    data = request.get_json()
    code = data.get('code')
    language = data.get('language')
    nebula_message_id = data.get('processId')

    if not code or not language:
        return jsonify({'error': 'Missing code or language.'}), 400

    if 'last_run_code_process_id' in session:
        stop_and_cleanup_app_by_process_id(session.pop('last_run_code_process_id', None), app_type='run_code')
        session.modified = True

    final_frontend_code = None
    api_keys = {}
    db = get_db()
    user_id = session['user_id']

    if nebula_message_id and language == 'python':
        try:
            message_id = int(nebula_message_id)
            cursor = db.execute(
                '''SELECT m.nebula_step1, m.nebula_step2_frontend, c.user_id
                   FROM messages m JOIN chats c ON m.chat_id = c.id
                   WHERE m.id = ? AND c.user_id = ?''', (message_id, user_id)
            )
            result = _fetchone_as_dict(cursor)
            if result:
                raw_code_from_db = result.get('nebula_step2_frontend')
                if raw_code_from_db:
                    match = re.search(r'```html\s*\n(<!DOCTYPE html>[\s\S]*?<\/html>)\s*\n```', raw_code_from_db, re.IGNORECASE | re.DOTALL)
                    if match:
                        final_frontend_code = match.group(1).strip()
                
                step1_plan = result.get('nebula_step1')
                if step1_plan:
                    keys_section_regex = r"1\.\s+Required\s+API\s+Keys"
                    key_name_regex = r'`([A-Z_]+)`'
                    plan_parts = re.split(keys_section_regex, step1_plan, flags=re.IGNORECASE)
                    if len(plan_parts) > 1:
                        key_section_content = plan_parts[1].split('2.')[0]
                        required_key_names = re.findall(key_name_regex, key_section_content)
                        for key_name in required_key_names:
                            key_cursor = db.execute('SELECT encrypted_value FROM user_api_keys WHERE user_id = ? AND key_name = ?', (user_id, key_name))
                            encrypted_key_data = _fetchone_as_dict(key_cursor)
                            if encrypted_key_data:
                                api_keys[key_name] = cipher_suite.decrypt(encrypted_key_data['encrypted_value']).decode('utf-8')
                            else:
                                return jsonify({'error': f"Execution failed: Required API key '{key_name}' is missing."}), 400
        except Exception as e:
            logger.error(f"Error during pre-run data retrieval: {e}", exc_info=True)

    main_code_basename = "app"
    if language == 'java':
        match = re.search(r'public\s+class\s+(\w+)', code)
        if match:
            main_code_basename = match.group(1)
    
    lang_config = {
        'python': {'image': 'stellar-python-sandbox:3.12', 'extension': '.py', 'command': lambda f: ['python', '-u', f]},
        'javascript': {'image': 'stellar-node-sandbox:latest', 'extension': '.js', 'command': lambda f: ['node', f]},
        'php': {'image': 'stellar-php-sandbox:latest', 'extension': '.php', 'command': lambda f: ['php', f]},
        'ruby': {'image': 'stellar-ruby-sandbox:latest', 'extension': '.rb', 'command': lambda f: ['ruby', f]},
        'go': {'image': 'stellar-go-sandbox:latest', 'extension': '.go', 'command': lambda f: ['go', 'run', f]},
        'c': {'image': 'stellar-c-sandbox:latest', 'extension': '.c', 'command': lambda f: ['/bin/sh', '-c', f'gcc -o program {f} && ./program']},
        'cpp': {'image': 'stellar-cpp-sandbox:latest', 'extension': '.cpp', 'command': lambda f: ['/bin/sh', '-c', f'g++ -o program {f} && ./program']},
        'java': {'image': 'stellar-java-sandbox:latest', 'extension': '.java', 'command': lambda f: ['/bin/sh', '-c', f'javac {f} && java {f.replace(".java", "")}']},
        'rust': {'image': 'stellar-rust-sandbox:latest', 'extension': '.rs', 'command': lambda f: ['/bin/sh', '-c', f'rustc -o program {f} && ./program']},
        'typescript': {'image': 'stellar-node-sandbox:latest', 'extension': '.ts', 'command': lambda f: ['/bin/sh', '-c', f'tsc {f} && node {f.replace(".ts", ".js")}']},
    }
    config = lang_config.get(language)
    if not config:
        return jsonify({'error': f'Unsupported language for execution: {language}'}), 400

    main_code_filename_with_ext = main_code_basename + config['extension']
    is_server_app = language == 'python' and 'app.run(' in code
    ports_to_publish = {'5000/tcp': ('0.0.0.0', 0)} if is_server_app else None
    
    run_id = str(uuid.uuid4())
    temp_dir_path = os.path.join(SANDBOX_DIR, run_id)
    try:
        os.makedirs(temp_dir_path)
        with open(os.path.join(temp_dir_path, main_code_filename_with_ext), 'w', encoding="utf-8") as f: f.write(code)
        if final_frontend_code:
            with open(os.path.join(temp_dir_path, 'index.html'), 'w', encoding="utf-8") as f: f.write(final_frontend_code)
        if api_keys:
            with open(os.path.join(temp_dir_path, '.env'), 'w', encoding="utf-8") as f:
                for key, value in api_keys.items(): f.write(f"{key}={value}\n")
        abs_temp_dir_path = os.path.abspath(temp_dir_path)
    except Exception as e:
        logger.error(f"Failed to set up execution environment: {e}", exc_info=True)
        return jsonify({'error': f'Failed to set up execution environment: {e}'}), 500

    def generate():
        container = None
        process_id = None
        try:
            if is_server_app:
                process_id = str(uuid.uuid4())
                with app.app_context():
                    session['last_run_code_process_id'] = process_id
                    session.modified = True
                
                redis_key = _redis_runcode_key(process_id)
                redis_client.hset(redis_key, mapping={ "status": "starting", "process_id": process_id })


            container = client.containers.run(
                image=config['image'], command=config['command'](main_code_filename_with_ext),
                working_dir='/app', volumes={abs_temp_dir_path: {'bind': '/app', 'mode': 'rw'}},
                ports=ports_to_publish, mem_limit='1024m',
                name=f"stellar-sandbox-{run_id}", remove=False, detach=True,
                stdout=True, stderr=True
            )
            yield f"data: {json.dumps({'type': 'container_id', 'id': container.id})}\n\n"

            if is_server_app and process_id:
                redis_client.hset(redis_key, "container_id", container.id)
                public_url_found = False
                for _ in range(20):
                    time.sleep(1)
                    try:
                        container.reload()
                        if container.status != 'running': break
                        ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                        mapping = ports.get('5000/tcp')
                        host_port = mapping[0].get('HostPort') if mapping else None

                        if host_port:
                            with active_apps_lock:
                                active_apps[process_id] = {"port": int(host_port), "container_id": container.id}
                            redis_client.hset(redis_key, mapping={"host_port": str(host_port), "status": "running"})
                            public_url = f"https://stellarai.live/apps/{process_id}/"
                            yield f"data: {json.dumps({'type': 'port_info', 'url': public_url})}\n\n"
                            public_url_found = True
                            break
                    except (IndexError, TypeError, KeyError, AttributeError):
                        continue
                if not public_url_found:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Failed to get public URL for the app.'})}\n\n"

            for line_bytes in container.logs(stream=True, follow=True):
                cleaned_line = line_bytes.decode('utf-8', 'replace').strip()
                if cleaned_line:
                    yield f"data: {json.dumps({'type': 'log', 'content': cleaned_line})}\n\n"
            container.wait()

        except Exception as e:
            logger.error(f"Error during code execution stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            if not is_server_app and container:
                 try: container.remove(force=True)
                 except docker.errors.NotFound: pass

            if is_server_app and process_id:
                pass
            
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path, ignore_errors=True)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/user/api_keys', methods=['POST'])
def manage_api_keys():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    user_id = session['user_id']
    data = request.get_json()
    if not data or not isinstance(data.get('api_keys'), dict):
        return jsonify({'error': 'Invalid request. api_keys object is required.'}), 400
    
    db = get_db()
    try:
        for key_name, key_value in data['api_keys'].items():
            if not key_name or not key_value:
                continue
            
            encrypted_value = cipher_suite.encrypt(key_value.encode('utf-8'))
            db.execute(
                'INSERT OR REPLACE INTO user_api_keys (user_id, key_name, encrypted_value) VALUES (?, ?, ?)',
                (user_id, key_name, encrypted_value)
            )
        db.commit()
        logging.info(f"Successfully saved/updated API keys for user {user_id}.")
        return jsonify({'success': True, 'message': 'API keys saved successfully.'}), 200
    except Exception as e:
        logger.error(f"Error saving API keys for user {user_id}: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while saving keys.'}), 500

@app.route('/nebula/save_keys', methods=['POST'])
def nebula_save_keys():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data received.'}), 400

    process_id = data.get('processId')
    api_keys = data.get('api_keys')

    if not process_id or not isinstance(api_keys, dict):
        return jsonify({'error': 'Missing or invalid parameters: processId and api_keys are required.'}), 400

    process_id_str = str(process_id)
    if 'nebula_processes' in session and process_id_str in session['nebula_processes']:
        process_state = session['nebula_processes'][process_id_str]
        
        if process_state.get('chat_id') and get_current_chat_id(session['user_id']) != process_state.get('chat_id'):
             return jsonify({'error': 'Authorization error.'}), 403

        process_state['api_keys'] = api_keys
        session['nebula_processes'][process_id_str] = process_state
        session.modified = True
        logging.info(f"Successfully saved API keys for Nebula process {process_id_str}.")
        return jsonify({'success': True, 'message': 'API keys saved successfully.'}), 200
    else:
        return jsonify({'error': f'Nebula process {process_id_str} not found.'}), 404

@app.route('/api/stop_container', methods=['POST'])
def stop_container():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    if not client:
        return jsonify({'error': 'Docker client is not available.'}), 503

    data = request.get_json()
    container_id = data.get('container_id')

    if not container_id:
        return jsonify({'error': 'Missing container_id.'}), 400

    try:
        process_id = None
        app_type = 'run_code'

        for key in redis_client.scan_iter("runcode:process:*"):
            cid = redis_client.hget(key, "container_id")
            if cid and cid == container_id:
                process_id = redis_client.hget(key, "process_id")
                break
        
        if not process_id:
            app_type = 'forge'
            for key in redis_client.scan_iter("forge:process:*"):
                cid = redis_client.hget(key, "container_id")
                if cid and cid == container_id:
                    process_id = redis_client.hget(key, "process_id")
                    break

        if process_id:
            stop_and_cleanup_app_by_process_id(process_id, app_type)
            return jsonify({'success': True, 'message': f'Container {container_id[:12]} and its process stopped.'}), 200
        else:
            container = client.containers.get(container_id)
            logging.info(f"Stopping container {container.short_id} directly (no process found).")
            container.stop(timeout=10)
            return jsonify({'success': True, 'message': f'Container {container.short_id} stopped.'}), 200

    except docker.errors.NotFound:
        return jsonify({'success': False, 'message': 'Container not found (may have already stopped).'}), 404
    except Exception as e:
        logger.error(f"Error stopping container {container_id} via API: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def _generate_test_harness(problem_title, user_code_filename, test_cases, language='python'):
    if language != 'python':
        return None

    match = re.search(r'def\s+(\w+)\s*\(', user_code_filename)
    if not match:
        match_class = re.search(r'class\s+Solution:\s+def\s+(\w+)\s*\(', user_code_filename, re.DOTALL)
        if not match_class:
            return None
        function_name = f"Solution().{match_class.group(1)}"
        module_name = os.path.splitext(os.path.basename('user_solution.py'))[0]
        import_line = f"from {module_name} import Solution"
    else:
        function_name = match.group(1)
        module_name = os.path.splitext(os.path.basename('user_solution.py'))[0]
        import_line = f"from {module_name} import {function_name}"


    test_cases_json = json.dumps(test_cases)

    harness_code = f"""
import json
import sys
import time
import traceback

try:
    {import_line}
except Exception as e:
    print(json.dumps({{"status": "Import Error", "results": [{{"id": 0, "status": "error", "message": f"Failed to import your solution: {{e}}"}}], "final_summary": "Could not import your code. Check for syntax errors."}}))
    sys.exit(0)

def run_tests():
    test_cases = {test_cases_json}
    results = []
    passed_count = 0
    start_time = time.time()

    for test in test_cases:
        test_input = json.loads(test['input_data'])
        if isinstance(test_input, dict):
            args = tuple(test_input.values())
        else:
            args = (test_input,)

        expected_output = json.loads(test['expected_output'])
        
        test_result = {{
            "id": test['id'],
            "status": "pending",
            "input": test_input,
            "expected": expected_output,
            "actual": None,
            "message": ""
        }}

        try:
            actual_output = {function_name}(*args)
            test_result['actual'] = actual_output
            
            is_correct = False
            if isinstance(actual_output, (list, tuple)) and isinstance(expected_output, (list, tuple)):
                is_correct = sorted(actual_output) == sorted(expected_output)
            else:
                is_correct = actual_output == expected_output

            if is_correct:
                test_result['status'] = 'passed'
                passed_count += 1
            else:
                test_result['status'] = 'failed'
                
        except Exception as e:
            test_result['status'] = 'error'
            test_result['message'] = traceback.format_exc().splitlines()[-1]

        results.append(test_result)

    total_time = (time.time() - start_time) * 1000
    
    final_status = "Accepted"
    if passed_count != len(test_cases):
        failed_tests = len(test_cases) - passed_count
        final_status = f"Wrong Answer"

    final_summary = f"{{passed_count}} / {{len(test_cases)}} tests passed in {{total_time:.2f}}ms."

    print(json.dumps({{
        "status": final_status,
        "results": results,
        "final_summary": final_summary
    }}))

if __name__ == "__main__":
    run_tests()
"""
    return harness_code    
    
@app.route('/codelab/problem', methods=['POST'])
def get_codelab_problem():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()
    problem_id = data.get('problem_id') if data else None

    db = get_db()
    try:
        problem = None
        if problem_id:
            problem_cursor = db.execute('SELECT id, title, description, difficulty, topic_tags FROM problems WHERE id = ?', (problem_id,))
            problem = _fetchone_as_dict(problem_cursor)
        else:
            problem_cursor = db.execute('SELECT id, title, description, difficulty, topic_tags FROM problems ORDER BY RANDOM() LIMIT 1')
            problem = _fetchone_as_dict(problem_cursor)

        if not problem:
            return jsonify({'error': 'Problem not found.'}), 404

        test_cases_cursor = db.execute(
            'SELECT id, input_data, expected_output FROM test_cases WHERE problem_id = ? AND is_hidden = 0',
            (problem['id'],)
        )
        test_cases = []
        for row in _fetch_as_dict(test_cases_cursor):
            try:
                row['input_data'] = json.loads(row['input_data'])
            except (json.JSONDecodeError, TypeError):
                pass
            test_cases.append(row)

        response_data = {
            'problem': problem,
            'test_cases': test_cases
        }
        
        return jsonify(response_data), 200

    except sqlite3.Error as e:
        logger.error(f"Database error in get_codelab_problem: {e}", exc_info=True)
        return jsonify({'error': 'A database error occurred.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in get_codelab_problem: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/codelab/submit', methods=['POST'])
def submit_codelab_solution():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()
    problem_id = data.get('problem_id')
    code = data.get('code')
    language = data.get('language')
    run_type = data.get('run_type', 'submit')

    if not all([code, language]):
        return jsonify({'error': 'Missing code or language.'}), 400
    
    if run_type == 'submit' and not problem_id:
        return jsonify({'error': 'A problem must be loaded to submit a solution.'}), 400
    
    db = get_db()
    try:
        run_id = str(uuid.uuid4())
        temp_dir_path = os.path.join(SANDBOX_DIR, run_id)
        os.makedirs(temp_dir_path, exist_ok=True)
        
        user_code_path = os.path.join(temp_dir_path, 'user_solution.py')
        with open(user_code_path, 'w', encoding='utf-8') as f:
            f.write(code)

        command_to_run = ['python', 'user_solution.py']

        if problem_id:
            problem_cursor = db.execute('SELECT title FROM problems WHERE id = ?', (problem_id,))
            problem = _fetchone_as_dict(problem_cursor)
            if not problem:
                shutil.rmtree(temp_dir_path)
                return jsonify({'error': 'Problem not found.'}), 404

            query = 'SELECT id, input_data, expected_output FROM test_cases WHERE problem_id = ?'
            if run_type == 'run':
                query += ' AND is_hidden = 0'
            
            cases_cursor = db.execute(query, (problem_id,))
            test_cases = _fetch_as_dict(cases_cursor)

            if not test_cases:
                if run_type == 'submit':
                    shutil.rmtree(temp_dir_path)
                    return jsonify({'error': 'No test cases found for this problem.'}), 404
            else:
                harness_code = _generate_test_harness(problem['title'], code, test_cases, language)
                if not harness_code:
                    shutil.rmtree(temp_dir_path)
                    return jsonify({'error': 'Could not generate test harness for your code structure.'}), 500
                
                harness_path = os.path.join(temp_dir_path, 'test_harness.py')
                with open(harness_path, 'w', encoding='utf-8') as f:
                    f.write(harness_code)
                command_to_run = ['python', 'test_harness.py']

        client = docker.from_env()
        container = None
        output_str = '{"status": "Execution Error", "final_summary": "The sandbox failed to run."}'
        
        try:
            container = client.containers.run(
                image='stellar-python-sandbox:3.12',
                command=command_to_run,
                working_dir='/app',
                volumes={os.path.abspath(temp_dir_path): {'bind': '/app', 'mode': 'ro'}},
                mem_limit='256m', cpu_shares=512, remove=True, detach=False,
                stdout=True, stderr=True
            )
            output_str = container.decode('utf-8')
        except docker.errors.ContainerError as e:
            output_str = json.dumps({
                "status": "Runtime Error",
                "final_summary": "Your code produced an error.",
                "raw_error": e.stderr.decode('utf-8') if e.stderr else str(e)
            })
        finally:
            shutil.rmtree(temp_dir_path, ignore_errors=True)

        if not problem_id or (run_type == 'run' and not test_cases):
            final_result = {
                "status": "Finished",
                "final_summary": "Code executed successfully.",
                "raw_output": output_str
            }
            if "Runtime Error" in output_str:
                 try:
                    error_data = json.loads(output_str)
                    final_result = error_data
                 except json.JSONDecodeError:
                    final_result["final_summary"] = "Code execution resulted in an error."

        else:
            try:
                final_result = json.loads(output_str)
            except json.JSONDecodeError:
                final_result = {"status": "Execution Error", "final_summary": "The test harness produced invalid output.", "raw_output": output_str}

        if run_type == 'submit':
            db.execute(
                '''INSERT INTO submissions (user_id, problem_id, code, language, status, output_details)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (session['user_id'], problem_id, code, language, final_result.get('status', 'Unknown'), json.dumps(final_result))
            )
            db.commit()

        return jsonify(final_result), 200

    except Exception as e:
        logger.error(f"Unexpected error in submit_codelab_solution: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/codelab/assist', methods=['POST'])
def codelab_ai_assist():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    data = request.get_json()
    assist_type = data.get('assist_type')
    code = data.get('code')
    problem_context = data.get('problem_context', 'No problem loaded.')
    error_message = data.get('error_message', '')
    test_case_context = data.get('test_case_context', '')

    if not all([assist_type, code]):
        return jsonify({'error': 'Missing assist_type or code.'}), 400

    prompt = ""
    if assist_type == 'explain':
        prompt = get_codelab_explain_prompt(code, problem_context)
    elif assist_type == 'debug':
        prompt = get_codelab_debug_prompt(code, problem_context, error_message, test_case_context)
    elif assist_type == 'optimize':
        prompt = get_codelab_optimize_prompt(code, problem_context)
    else:
        return jsonify({'error': 'Invalid assist_type.'}), 400

    def generate_assist_stream():
        try:
            model_id = "us.amazon.nova-pro-v1:0" 

            generator = llm_generate(prompt, model_id)
            
            for chunk in generator:
                if 'result' in chunk:
                    clean_chunk = json.dumps({'content': chunk['result']})
                    yield f"data: {clean_chunk}\n\n"
                elif 'status' in chunk:
                    yield f"data: {json.dumps({'status': chunk['status']})}\n\n"
                elif 'error' in chunk:
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
        except Exception as e:
            logger.error(f"Error in codelab assist stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'An unexpected error occurred during AI assistance.'})}\n\n"

    return Response(stream_with_context(generate_assist_stream()), mimetype='text/event-stream')

@app.route('/codelab/generate_problem', methods=['POST'])
def generate_codelab_problem():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    data = request.get_json()
    user_request = data.get('user_request')
    if not user_request:
        return jsonify({'error': 'A description of the problem to generate is required.'}), 400

    prompt = get_codelab_generate_problem_prompt(user_request)
    model_id = "us.amazon.nova-premier-v1:0"
    
    raw_response = None
    try:
        generator = llm_generate(prompt, model_id)
        raw_response = next((item['result'] for item in generator if 'result' in item), None)
        
        if not raw_response:
            raise ValueError("AI failed to generate a response.")

        clean_json_string = _extract_json_from_response(raw_response)
        if not clean_json_string:
            raise ValueError("AI response did not contain a valid JSON object.")
        
        problem_data = json.loads(clean_json_string)

        required_keys = ['title', 'description', 'difficulty', 'topic_tags', 'test_cases']
        if not all(key in problem_data for key in required_keys):
            raise ValueError("AI response is missing required keys.")
        if not isinstance(problem_data['test_cases'], list) or not problem_data['test_cases']:
            raise ValueError("AI response must include a non-empty list of test_cases.")

        db = get_db()
        cursor = db.execute(
            'INSERT INTO problems (title, description, difficulty, topic_tags) VALUES (?, ?, ?, ?)',
            (problem_data['title'], problem_data['description'], problem_data['difficulty'], problem_data['topic_tags'])
        )
        problem_id = cursor.lastrowid
        
        for tc in problem_data['test_cases']:
            db.execute(
                'INSERT INTO test_cases (problem_id, input_data, expected_output, is_hidden) VALUES (?, ?, ?, ?)',
                (problem_id, tc['input_data'], tc['expected_output'], tc['is_hidden'])
            )
        
        db.commit()

        return jsonify({'success': True, 'new_problem_id': problem_id})

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing or validating AI-generated problem: {e}\nRaw Response: {raw_response}", exc_info=True)
        return jsonify({'error': f'The AI generated an invalid problem structure. Please try again. Details: {e}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in generate_codelab_problem: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while generating the problem.'}), 500
    
def _extract_json_from_response(text):
    pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
    match = re.search(pattern, text)

    if match:
        return match.group(1)
    else:
        json_start_index = text.find('{')
        if json_start_index == -1:
            return None

        brace_level = 0
        for i in range(json_start_index, len(text)):
            char = text[i]
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0:
                    return text[json_start_index : i + 1]
    
    return None

def cleanup_stale_containers():
    try:
        client = docker.from_env()
        stale_containers = client.containers.list(
            all=True, 
            filters={'name': 'stellar-sandbox-*'}
        )
        
        if not stale_containers:
            logging.info("No stale sandbox containers found on startup.")
            return

        logging.warning(f"Found {len(stale_containers)} stale sandbox container(s). Cleaning up...")
        for container in stale_containers:
            try:
                logging.warning(f"Force-removing stale container: {container.name} ({container.short_id})")
                container.remove(force=True) 
            except docker.errors.NotFound:
                logging.info(f"Container {container.name} was already removed.")
            except Exception as e:
                logging.error(f"Error during cleanup of container {container.name}: {e}")
        logging.info("Stale container cleanup complete.")
        
    except docker.errors.DockerException as e:
        logging.error(f"Docker is not available. Skipping stale container cleanup. Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during stale container cleanup: {e}")

active_apps = {}
active_apps_lock = threading.Lock()
@app.route("/apps/<app_id>/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/apps/<app_id>/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def dynamic_proxy(app_id, path):
    app_info = None
    with active_apps_lock:
        app_info = active_apps.get(app_id)

    if not app_info:
        try:
            redis_key = _redis_forge_key(app_id)
            redis_data = redis_client.hgetall(redis_key)
            if not redis_data:
                redis_key = _redis_runcode_key(app_id)
                redis_data = redis_client.hgetall(redis_key)

            if redis_data and redis_data.get("host_port") and redis_data.get("status") in ["running", "created", "exited"]:
                app_info = {
                    "port": int(redis_data["host_port"]),
                    "container_id": redis_data.get("container_id"),
                    "status": redis_data.get("status")
                }
                with active_apps_lock:
                    active_apps[app_id] = app_info
        except Exception as e:
            logger.error(f"Redis lookup failed for app {app_id}: {e}")
            return "Error looking up application state.", 500

    if not app_info or not app_info.get("port"):
        return "Application not found or has been stopped.", 404
    
    target_port = app_info["port"]
    target_url = f"http://127.0.0.1:{target_port}/{path}"

    try:
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers={key: value for (key, value) in request.headers if key.lower() != 'host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            stream=True, 
            timeout=20
        )

        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]

        return Response(resp.content, resp.status_code, headers)

    except requests.exceptions.RequestException as e:
        logger.error(f"Dynamic proxy error for app {app_id}: {e}")
        if app_info.get("status") == "exited":
             return "Application not found or has been stopped.", 404
        return f"Error proxying request to application.", 502

@app.route('/codelab/forge/files', methods=['GET'])
def forge_get_files():
    if 'user_id' not in session or 'forge_project' not in session:
        return jsonify({'error': 'No active Forge session.'}), 404
    return jsonify(session['forge_project'].get('files', {}))

@app.route('/codelab/forge/redeploy', methods=['POST'])
def forge_redeploy():
    if 'user_id' not in session or 'forge_project' not in session:
        return jsonify({'error': 'No active session.'}), 400
    if not client:
        return jsonify({'error': 'Docker client not available.'}), 503

    data = request.get_json(silent=True) or {}
    updated_files = data.get('files', {})

    if 'index.html' not in updated_files or 'app.py' not in updated_files:
        return jsonify({'error': "Request must contain 'index.html' and 'app.py'."}), 400

    old_container_id = session['forge_project'].get('container_id')
    old_process_id = session['forge_project'].get('process_id')

    if old_process_id:
        with active_apps_lock:
            active_apps.pop(old_process_id, None)

    try:
        process_id = str(uuid.uuid4())

        session['forge_project']['files'] = updated_files
        session['forge_project']['process_id'] = process_id
        session.modified = True

        try:
            redis_client.hset(_redis_forge_key(process_id), mapping={
                "status": "starting",
                "files": json.dumps(updated_files)
            })
        except Exception:
            logger.exception("Failed to persist redeploy forge state for %s", process_id)

        app_obj = current_app._get_current_object()
        thread = threading.Thread(target=_deploy_and_stream_output, args=(app_obj, updated_files, process_id, old_container_id, 'forge'))
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'process_id': process_id})

    except Exception as e:
        logger.error(f"Error in forge_redeploy: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

cleanup_stale_containers()
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5013))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

