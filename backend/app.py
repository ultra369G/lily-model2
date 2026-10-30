import os
import json
import hashlib
import sqlite3
import traceback
import numpy as np
import networkx as nx
from collections import deque
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket
import re
import threading
from threading import Thread
import time
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_sock import Sock
from werkzeug.utils import secure_filename
import uuid
import PyPDF2
import logging
from dotenv import load_dotenv

# ============================================================================
# LOAD ENV
# ============================================================================

load_dotenv()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

DB_FILE = os.path.join(DATA_DIR, "knowledge_v4.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================================
# FLASK
# ============================================================================

app = Flask(__name__)

app.secret_key = os.urandom(24)

CORS(app)

sock = Sock(app)

# ============================================================================
# CONFIG
# ============================================================================

ADMIN_PASSWORD = hashlib.sha256(b"nexus2023").hexdigest()

ALLOWED_EXTENSIONS = {"txt", "json", "pdf"}

SIMILARITY_THRESHOLD = 0.40

SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

EMAIL_ENABLED = bool(
    SMTP_SERVER and
    SMTP_USERNAME and
    SMTP_PASSWORD
)

# ============================================================================
# GLOBALS
# ============================================================================

knowledge_cache = {}

admin_tokens = {}

ws_clients = set()

# ============================================================================
# SIMPLE EMBEDDER
# ============================================================================

class SimpleEmbedder:

    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 "
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str):

        text = text.lower()

        vec = np.zeros(self.vocab_size, dtype=np.float32)

        for char in text:
            if char in self.vocab:
                vec[self.vocab[char]] += 1

        norm = np.linalg.norm(vec)

        if norm > 0:
            vec = vec / norm

        return vec

    def similarity(self, v1, v2):

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 == 0 or n2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (n1 * n2))


embedder = SimpleEmbedder()

# ============================================================================
# DATABASE
# ============================================================================

def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def init_db():

    logger.info(f"Initializing database at: {DB_FILE}")

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input TEXT UNIQUE,
        output TEXT,
        embedding BLOB,
        category TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        email TEXT UNIQUE,
        phone TEXT,
        details TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS email_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient_name TEXT,
        recipient_email TEXT,
        subject TEXT,
        content TEXT,
        status TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

# ============================================================================
# CONCEPT PROCESSOR
# ============================================================================

class ConceptProcessor:

    def __init__(self):
        self.graph = nx.Graph()

    def extract(self, text):

        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

        stopwords = {
            "this", "that", "with", "from",
            "what", "your", "have", "will",
            "would", "there", "their", "about"
        }

        return [
            w for w in words
            if w not in stopwords
        ]

    def build(self, input_text, output_text):

        concepts = self.extract(input_text) + self.extract(output_text)

        concepts = list(set(concepts))

        for c in concepts:
            self.graph.add_node(c)

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                self.graph.add_edge(concepts[i], concepts[j])

    def debug(self):

        logger.info(f"Concept Graph Nodes: {list(self.graph.nodes())}")
        logger.info(f"Concept Graph Edges: {list(self.graph.edges())}")


concept_processor = ConceptProcessor()

# ============================================================================
# MEMORY
# ============================================================================

class ConversationMemory:

    def __init__(self, max_len=10):
        self.history = deque(maxlen=max_len)

    def add(self, user, ai):

        self.history.append({
            "input": user,
            "output": ai,
            "timestamp": datetime.now().isoformat()
        })

    def clear(self):
        self.history.clear()


memory = ConversationMemory()

# ============================================================================
# ADMIN
# ============================================================================

class AdminConsole:

    def authenticate(self, password):

        hashed = hashlib.sha256(password.encode()).hexdigest()

        return hashed == ADMIN_PASSWORD

    def stats(self):

        return {
            "knowledge": len(knowledge_cache),
            "contacts": len(get_all_contacts()),
            "emails": len(get_email_logs())
        }


admin_console = AdminConsole()

# ============================================================================
# KNOWLEDGE
# ============================================================================

def load_knowledge_cache():

    knowledge_cache.clear()

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, input, output, embedding, category
    FROM knowledge
    """)

    rows = cursor.fetchall()

    for row in rows:

        try:

            if row[3]:
                embedding = np.frombuffer(row[3], dtype=np.float32)
            else:
                embedding = embedder.encode(row[1])

            knowledge_cache[row[1].lower()] = {
                "id": row[0],
                "output": row[2],
                "embedding": embedding,
                "category": row[4]
            }

            concept_processor.build(row[1], row[2])

        except Exception as e:
            logger.error(f"Load cache error: {e}")

    conn.close()

    concept_processor.debug()


def store_knowledge(input_text, output_text, category="trained"):

    try:

        cleaned_input = input_text.strip().lower()

        embedding = embedder.encode(cleaned_input)

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO knowledge
        (input, output, embedding, category)
        VALUES (?, ?, ?, ?)
        """, (
            cleaned_input,
            output_text,
            embedding.tobytes(),
            category
        ))

        conn.commit()

        conn.close()

        knowledge_cache[cleaned_input] = {
            "output": output_text,
            "embedding": embedding,
            "category": category
        }

        concept_processor.build(cleaned_input, output_text)

        logger.info(f"Learned: {cleaned_input}")

        return True

    except Exception as e:

        logger.error(f"Store knowledge error: {e}")

        return False


def search_knowledge(query):

    query = query.lower().strip()

    if query in knowledge_cache:
        return knowledge_cache[query]["output"]

    # partial
    for key, value in knowledge_cache.items():

        if query in key or key in query:
            return value["output"]

    # semantic
    if knowledge_cache:

        q_embedding = embedder.encode(query)

        best = None
        best_score = SIMILARITY_THRESHOLD

        for key, value in knowledge_cache.items():

            score = embedder.similarity(
                q_embedding,
                value["embedding"]
            )

            if score > best_score:
                best_score = score
                best = value["output"]

        if best:
            return best

    return None

# ============================================================================
# CONTACTS
# ============================================================================

def validate_email(email):

    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"

    return re.match(pattern, email) is not None


def save_contact(name, email, phone="", details=""):

    try:

        if not validate_email(email):
            return False, "Invalid email"

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO contacts
        (name, email, phone, details)
        VALUES (?, ?, ?, ?)
        """, (
            name.lower(),
            email.lower(),
            phone,
            details
        ))

        conn.commit()
        conn.close()

        return True, "Contact saved"

    except Exception as e:

        logger.error(f"Save contact error: {e}")

        return False, "Failed to save contact"


def get_contact_by_name(name):

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
        SELECT name, email, phone, details
        FROM contacts
        WHERE LOWER(name)=?
        """, (name.lower(),))

        row = cursor.fetchone()

        conn.close()

        if row:
            return {
                "name": row[0],
                "email": row[1],
                "phone": row[2],
                "details": row[3]
            }

        return None

    except Exception as e:

        logger.error(f"Get contact error: {e}")

        return None


def get_all_contacts():

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
        SELECT name, email, phone, details, timestamp
        FROM contacts
        ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()

        conn.close()

        return [
            {
                "name": r[0],
                "email": r[1],
                "phone_number": r[2] or "",
                "details": r[3] or "",
                "timestamp": r[4]
            }
            for r in rows
        ]

    except Exception as e:

        logger.error(f"Get contacts error: {e}")

        return []

# ============================================================================
# EMAIL
# ============================================================================

def get_email_logs():

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
        SELECT id,
               recipient_name,
               recipient_email,
               subject,
               status,
               timestamp
        FROM email_logs
        ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()

        conn.close()

        return [
            {
                "id": r[0],
                "recipient_name": r[1],
                "recipient_email": r[2],
                "subject": r[3],
                "status": r[4],
                "timestamp": r[5],
                "intent": "manual"
            }
            for r in rows
        ]

    except Exception as e:

        logger.error(f"Get email logs error: {e}")

        return []


def send_email(recipient_email, subject, content):

    if not EMAIL_ENABLED:
        return False, "Email disabled in offline mode"

    try:

        msg = MIMEMultipart()

        msg["From"] = SMTP_USERNAME
        msg["To"] = recipient_email
        msg["Subject"] = subject

        msg.attach(MIMEText(content, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:

            server.starttls()

            server.login(
                SMTP_USERNAME,
                SMTP_PASSWORD
            )

            server.send_message(msg)

        return True, "Email sent"

    except Exception as e:

        logger.error(f"Email send error: {e}")

        return False, str(e)

# ============================================================================
# FILES
# ============================================================================

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def parse_teaching_command(line):

    pattern = r'^teach\s+input="([^"]+)"\s+output="([^"]+)"(?:\s+category="([^"]+)")?$'

    match = re.match(pattern, line.strip(), re.IGNORECASE)

    if not match:
        return None

    return {
        "input": match.group(1),
        "output": match.group(2),
        "category": match.group(3) if match.group(3) else "trained"
    }


def parse_text_file(path):

    entries = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            parsed = parse_teaching_command(line)

            if parsed:
                entries.append(parsed)

    return entries


def parse_json_file(path):

    with open(path, "r", encoding="utf-8") as f:

        data = json.load(f)

    if isinstance(data, list):
        return data

    return []


def parse_pdf_file(path):

    entries = []

    with open(path, "rb") as f:

        reader = PyPDF2.PdfReader(f)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

        for line in text.splitlines():

            parsed = parse_teaching_command(line)

            if parsed:
                entries.append(parsed)

    return entries

# ============================================================================
# HELPERS
# ============================================================================

def verify_admin_token(client_ip, token):

    if client_ip not in admin_tokens:
        return False

    item = admin_tokens[client_ip]

    return (
        item["token"] == token and
        item["expires"] > datetime.now()
    )


def handle_math(message):

    try:

        expr = (
            message.lower()
            .replace("plus", "+")
            .replace("minus", "-")
            .replace("times", "*")
            .replace("divided by", "/")
        )

        if not re.fullmatch(r"[0-9\+\-\*\/\.\(\) ]+", expr):
            return None

        result = eval(expr, {"__builtins__": {}})

        return f"The result is {result}"

    except:
        return None


def handle_time(message):

    lower = message.lower()

    now = datetime.now()

    if "time" in lower:
        return now.strftime("Current time: %H:%M:%S")

    if "date" in lower:
        return now.strftime("Current date: %Y-%m-%d")

    return None

# ============================================================================
# RESPONSE ENGINE
# ============================================================================

def generate_response(message):

    answer = search_knowledge(message)

    if answer:
        return answer

    return (
        "I do not have training data for that yet.\n\n"
        f'Teach me using:\n'
        f'teach input="{message}" output="answer here"'
    )

# ============================================================================
# MESSAGE PROCESSOR
# ============================================================================

def process_message(data, client_ip):

    try:

        message = data.get("message", "").strip()

        admin_token = data.get("admin_token", "")

        is_admin = verify_admin_token(
            client_ip,
            admin_token
        )

        learning_mode = data.get("learning_mode", False)

        if not message:
            return {"error": "Empty message"}

        # HELP
        if message.lower() == "help":

            return {
                "response": """
Available commands:

help
clear
what is the time
what is the date

Admin:
!admin nexus2023
learn:start
learn:stop

Teaching:
teach input="hello" output="hi"

Contacts:
save contact name="john" email="john@gmail.com"
show contacts

Emails:
show emails
""",
                "is_admin": is_admin,
                "admin_token": admin_token
            }

        # CLEAR
        if message.lower() == "clear":

            memory.clear()

            return {
                "response": "Conversation cleared",
                "is_admin": is_admin
            }

        # ADMIN LOGIN
        if message.startswith("!admin "):

            password = message.replace("!admin ", "").strip()

            if admin_console.authenticate(password):

                token = str(uuid.uuid4())

                admin_tokens[client_ip] = {
                    "token": token,
                    "expires": datetime.now() + timedelta(hours=2)
                }

                return {
                    "response": "Admin mode enabled",
                    "is_admin": True,
                    "admin_token": token
                }

            return {
                "response": "Invalid password",
                "is_admin": False
            }

        # ADMIN ONLY
        if is_admin:

            if message.lower() == "learn:start":

                return {
                    "response": "Teaching mode enabled",
                    "is_admin": True,
                    "admin_token": admin_token,
                    "learning_mode": True
                }

            if message.lower() == "learn:stop":

                return {
                    "response": "Teaching mode disabled",
                    "is_admin": True,
                    "admin_token": admin_token,
                    "learning_mode": False
                }

            if (
                message.lower().startswith("teach ")
                and learning_mode
            ):

                parsed = parse_teaching_command(message)

                if not parsed:

                    return {
                        "response": "Invalid teaching format",
                        "is_admin": True,
                        "admin_token": admin_token,
                        "learning_mode": True
                    }

                success = store_knowledge(
                    parsed["input"],
                    parsed["output"],
                    parsed["category"]
                )

                if success:

                    return {
                        "response": f'Learned: {parsed["input"]}',
                        "is_admin": True,
                        "admin_token": admin_token,
                        "learning_mode": True
                    }

                return {
                    "response": "Failed to learn",
                    "is_admin": True,
                    "admin_token": admin_token,
                    "learning_mode": True
                }

            # SAVE CONTACT
            if message.lower().startswith("save contact"):

                pattern = r'^save\s+contact\s+name="([^"]+)"\s+email="([^"]+)"'

                match = re.match(pattern, message, re.IGNORECASE)

                if not match:

                    return {
                        "response": "Invalid format",
                        "is_admin": True
                    }

                success, msg = save_contact(
                    match.group(1),
                    match.group(2)
                )

                return {
                    "response": msg,
                    "is_admin": True
                }

            # STATS
            if message.lower() == "!stats":

                stats = admin_console.stats()

                return {
                    "response":
                        f"Knowledge: {stats['knowledge']}\n"
                        f"Contacts: {stats['contacts']}\n"
                        f"Emails: {stats['emails']}",
                    "is_admin": True
                }

        # SHOW CONTACTS
        if message.lower() == "show contacts":

            contacts = get_all_contacts()

            return {
                "response": "Contacts loaded",
                "contacts": contacts,
                "type": "contact_list",
                "is_admin": is_admin
            }

        # SHOW EMAILS
        if message.lower() == "show emails":

            emails = get_email_logs()

            return {
                "response": "Email logs loaded",
                "emails": emails,
                "type": "email_list",
                "is_admin": is_admin
            }

        # MATH
        math_result = handle_math(message)

        if math_result:

            memory.add(message, math_result)

            return {
                "response": math_result,
                "is_admin": is_admin
            }

        # TIME
        time_result = handle_time(message)

        if time_result:

            memory.add(message, time_result)

            return {
                "response": time_result,
                "is_admin": is_admin
            }

        # AI RESPONSE
        response = generate_response(message)

        memory.add(message, response)

        return {
            "response": response,
            "is_admin": is_admin,
            "admin_token": admin_token,
            "learning_mode": learning_mode
        }

    except Exception as e:

        logger.error(f"Process message error: {e}")

        traceback.print_exc()

        return {
            "error": "Server error"
        }

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/favicon.ico")
def favicon():

    return Response(status=204)


@app.route("/chat", methods=["POST"])
def chat_http():

    try:

        data = request.json

        response = process_message(
            data,
            request.remote_addr
        )

        return jsonify(response)

    except Exception as e:

        logger.error(f"HTTP error: {e}")

        return jsonify({
            "error": "HTTP server error"
        }), 500


@sock.route("/chat")
def websocket_chat(ws):

    client_ip = request.remote_addr

    ws_clients.add(ws)

    logger.info(f"WebSocket connected: {client_ip}")

    while True:

        try:

            raw = ws.receive()

            if raw is None:
                break

            data = json.loads(raw)

            response = process_message(
                data,
                client_ip
            )

            ws.send(json.dumps(response))

        except Exception as e:

            logger.error(f"WebSocket error: {e}")

            break

    ws_clients.discard(ws)

    logger.info(f"WebSocket disconnected: {client_ip}")

# ============================================================================
# AUTOTRAIN
# ============================================================================

@app.route("/admin/autotrain", methods=["POST"])
def autotrain():

    try:

        client_ip = request.remote_addr

        token = request.headers.get(
            "X-Admin-Token",
            ""
        )

        if not verify_admin_token(client_ip, token):

            return jsonify({
                "error": "Unauthorized"
            }), 403

        if "file" not in request.files:

            return jsonify({
                "error": "No file"
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "error": "No selected file"
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "error": "Invalid file type"
            }), 400

        filename = secure_filename(file.filename)

        path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        file.save(path)

        ext = filename.rsplit(".", 1)[1].lower()

        entries = []

        if ext == "txt":
            entries = parse_text_file(path)

        elif ext == "json":
            entries = parse_json_file(path)

        elif ext == "pdf":
            entries = parse_pdf_file(path)

        count = 0

        for item in entries:

            if (
                "input" in item and
                "output" in item
            ):

                if store_knowledge(
                    item["input"],
                    item["output"],
                    item.get("category", "trained")
                ):
                    count += 1

        os.remove(path)

        return jsonify({
            "response": f"Learned {count} entries"
        })

    except Exception as e:

        logger.error(f"Autotrain error: {e}")

        return jsonify({
            "error": "Autotrain failed"
        }), 500

# ============================================================================
# INIT
# ============================================================================

try:

    init_db()

    load_knowledge_cache()

except Exception as e:

    logger.error(f"Initialization error: {e}")

    raise

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("ASTRAVERSE NEXUS")
    logger.info("FULLY OFFLINE LOCAL AI")
    logger.info("=" * 60)

    logger.info(f"Frontend: {FRONTEND_DIR}")
    logger.info(f"Database: {DB_FILE}")

    logger.info(
        f"Loaded {len(knowledge_cache)} knowledge entries"
    )

    logger.info("Server running on:")
    logger.info("http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )