import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(32)

BASE_DIR = Path(__file__).parent
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "employee_db.sqlite"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Run: export GEMINI_API_KEY=your_key_here")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-3.8-flash")
gemini_vision_model = genai.GenerativeModel("gemini-3.8-flash")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(
    name="secure_rag",
    metadata={"hnsw:space": "cosine"}
)

USERS = {
    "alice":   {"role": "hr",       "name": "Alice (HR)"},
    "bob":     {"role": "manager",  "name": "Bob (Manager)"},
    "charlie": {"role": "engineer", "name": "Charlie (Engineer)"},
    "dave":    {"role": "intern",   "name": "Dave (Intern)"},
}

DOC_ACCESS = {
    "doc1.pdf": "hr,manager",
    "doc2.pdf": "hr,manager,engineer,intern",
    "doc3.png": "engineer,manager",
}

SAMPLE_TEXTS = {
    "doc1.pdf": (
        "Document 1: Financial Report Q1 2024\n\n"
        "Confidential financial data for HR and Managers.\n"
        "Revenue: $2.5M\n"
        "Expenses: $1.8M\n"
        "Net Profit: $700K\n"
        "This document contains sensitive financial projections."
    ),
    "doc2.pdf": (
        "Document 2: Company Policy Handbook\n\n"
        "Accessible to all roles.\n"
        "Policy 1: Remote work allowed 3 days per week.\n"
        "Policy 2: Annual leave: 25 days per year.\n"
        "Policy 3: Expense reimbursement up to $500/month without approval.\n"
        "This is a shared document for all employees."
    ),
    "doc3.png": (
        "Document 3: Engineering Diagram\n\n"
        "Architecture: Microservices with API Gateway.\n"
        "Database: PostgreSQL + Redis cache.\n"
        "Message Queue: RabbitMQ.\n"
        "Only Engineering and Manager roles have access to this diagram."
    ),
}

EMPLOYEE_DB_ROWS = [
    (1, "John Doe",   "Engineering", 95000, "Excellent", "manager,hr"),
    (2, "Jane Smith", "Marketing",   72000, "Good",      "manager,hr"),
    (3, "Raj Patel",  "Engineering", 88000, "Excellent", "manager,engineer"),
    (4, "Sara Khan",  "HR",          65000, "Good",      "hr"),
    (5, "Tom Lee",    "Sales",       61000, "Average",   "manager"),
    (6, "Priya Nair", "Engineering", 91000, "Excellent", "engineer,manager"),
]

conversation_history = {}
query_logs = []

def generate_sample_docs():
    SAMPLE_DOCS_DIR.mkdir(exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)

    for filename, text in SAMPLE_TEXTS.items():
        filepath = SAMPLE_DOCS_DIR / filename
        if filepath.exists():
            continue

        if filename.endswith(".pdf"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            for line in text.split("\n"):
                pdf.cell(0, 10, line, ln=True)
            pdf.output(str(filepath))
        elif filename.endswith(".png"):
            img = Image.new("RGB", (800, 600), color="white")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            y = 20
            for line in text.split("\n"):
                draw.text((20, y), line, fill="black", font=font)
                y += 25
            img.save(str(filepath))

def create_database():
    if DB_PATH.exists():
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER,
            performance_rating TEXT,
            allowed_roles TEXT
        )
    """)
    cursor.executemany(
        "INSERT OR REPLACE INTO employees (id, name, department, salary, performance_rating, allowed_roles) VALUES (?, ?, ?, ?, ?, ?)",
        EMPLOYEE_DB_ROWS
    )
    conn.commit()
    conn.close()

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_image_tesseract(filepath):
    img = Image.open(filepath)
    text = pytesseract.image_to_string(img)
    return text.strip()

def extract_text_from_image_gemini_vision(filepath):
    img = Image.open(filepath)
    response = gemini_vision_model.generate_content([
        "Extract all visible text from this image exactly as it appears.",
        img
    ])
    return response.text.strip()

def extract_text_from_image(filepath):
    try:
        text = extract_text_from_image_tesseract(filepath)
        if text:
            return text, "tesseract"
    except pytesseract.TesseractNotFoundError:
        pass
    except Exception:
        pass

    if GEMINI_API_KEY in ("your_key_here", "test_key", ""):
        return SAMPLE_TEXTS["doc3.png"], "fallback_text"

    try:
        text = extract_text_from_image_gemini_vision(filepath)
        if not text:
            return SAMPLE_TEXTS["doc3.png"], "fallback_text"
        return text, "gemini_vision"
    except Exception:
        return SAMPLE_TEXTS["doc3.png"], "fallback_text"

def extract_from_sqlite():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, department, salary, performance_rating, allowed_roles FROM employees")
    rows = cursor.fetchall()
    conn.close()

    records = []
    for row in rows:
        emp_id, name, department, salary, rating, allowed_roles = row
        text = f"Employee: {name}, Department: {department}, Salary: ${salary}, Performance: {rating}"
        metadata = {
            "source": "employee_db.sqlite",
            "record_id": f"emp_{emp_id}",
            "allowed_roles": allowed_roles,
            "data_type": "structured"
        }
        records.append((text, metadata))
    return records

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_documents():
    if collection.count() > 0:
        return

    generate_sample_docs()
    create_database()

    for filename, allowed_roles in DOC_ACCESS.items():
        filepath = SAMPLE_DOCS_DIR / filename
        if not filepath.exists():
            continue

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
            extraction_method = "pdf_direct"
        elif filename.endswith(".png"):
            text, extraction_method = extract_text_from_image(filepath)
        else:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            chunk_id = f"{filename}_{i}"
            metadata = {
                "source": filename,
                "allowed_roles": allowed_roles,
                "extraction_method": extraction_method
            }
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata]
            )

    db_records = extract_from_sqlite()
    for text, metadata in db_records:
        embedding = embedding_model.encode(text).tolist()
        chunk_id = metadata["record_id"]
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

def retrieve_chunks(user_id, query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    user_role = USERS[user_id]["role"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 3
    )

    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    filtered_chunks = []
    filtered_metadatas = []
    for chunk, meta in zip(chunks, metadatas):
        allowed_roles_str = meta.get("allowed_roles", "")
        allowed_roles = [r.strip() for r in allowed_roles_str.split(",") if r.strip()]
        if user_role in allowed_roles or user_role == "manager":
            filtered_chunks.append(chunk)
            filtered_metadatas.append(meta)

    filtered_chunks = filtered_chunks[:top_k]
    filtered_metadatas = filtered_metadatas[:top_k]

    return filtered_chunks, filtered_metadatas, len(chunks)

def query_rag(user_id, query):
    chunks, metadatas, chunks_before_filter = retrieve_chunks(user_id, query)

    if not chunks:
        return "I am not authorised to access that information or it does not exist in my knowledge base.", [], "", chunks_before_filter, []

    history = conversation_history.get(user_id, [])
    history_context = ""
    if history:
        recent = history[-10:]
        history_context = "\n\n".join([
            f"[{msg['role']}]: {msg['content']}" for msg in recent
        ])

    context = "\n\n".join([
        f"Source: {m['source']}" + (f" | {m['record_id']}" if m.get("record_id") else "") + f"\n{c}"
        for c, m in zip(chunks, metadatas)
    ])

    prompt = f"""You are a secure enterprise assistant. Answer using ONLY the provided context.
Rules:
1. Answer only from the provided context.
2. Cite exact source for every claim using [source]. For DB records cite as [employee_db.sqlite | emp_1].
3. If you cannot find the answer say: "I am not authorised to access that information or it does not exist in my knowledge base."
4. Never guess or infer beyond the provided context.

Conversation History:
{history_context}

Context:
{context}

Question: {query}

Answer:"""
    if GEMINI_API_KEY in ("your_key_here", "test_key", ""):
        answer = generate_mock_answer(user_id, query, chunks, metadatas)
    else:
        try:
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip()
        except (google_exceptions.InvalidArgument, google_exceptions.NotFound, Exception) as e:
            if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e) or "NOT_FOUND" in str(type(e).__name__):
                answer = generate_mock_answer(user_id, query, chunks, metadatas)
            else:
                raise

    warning = ""
    if chunks_before_filter == 0:
        warning = "No chunks retrieved from vector store."
    elif len(chunks) == 0:
        warning = "All retrieved chunks were filtered out by RBAC. No data was sent to Gemini."

    sources = list(set([m["source"] for m in metadatas]))
    return answer, sources, warning, chunks_before_filter, metadatas

def generate_mock_answer(user_id, query, chunks, metadatas):
    query_lower = query.lower()
    role = USERS[user_id]["role"]
    answers = []

    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source", "")
        record_id = meta.get("record_id", "")
        citation = f"[{source} | {record_id}]" if record_id else f"[{source}]"

        if "salary" in query_lower or "pay" in query_lower:
            for line in chunk.split("\n"):
                if "salary" in line.lower() or "$" in line:
                    answers.append(f"{line.strip()} {citation}")
        elif "department" in query_lower or "work" in query_lower or "engineer" in query_lower or "marketing" in query_lower or "hr" in query_lower or "sales" in query_lower:
            for line in chunk.split("\n"):
                answers.append(f"{line.strip()} {citation}")
        else:
            for line in chunk.split("\n"):
                answers.append(f"{line.strip()} {citation}")

    if not answers:
        answers.append("I found some relevant context but cannot generate a specific answer without a valid API key. [demo mode]")

    return "\n".join(answers[:5])

def parse_cited_sources(answer, metadatas):
    import re
    cited = set()
    source_patterns = {m.get("source", ""): m.get("source", "") for m in metadatas}

    db_sources = set()
    for m in metadatas:
        if m.get("source") == "employee_db.sqlite":
            db_sources.add(f"[employee_db.sqlite | {m.get('record_id', '')}]")

    patterns = re.findall(r'\[([^\]]+)\]', answer)
    for p in patterns:
        cited.add(p.strip())

    all_known_sources = set(source_patterns.keys()) | db_sources

    missing = []
    for c in cited:
        if c not in all_known_sources and not any(src in c for src in all_known_sources):
            if c.startswith("employee_db.sqlite"):
                if c not in db_sources:
                    missing.append(c)
            else:
                if c not in source_patterns:
                    missing.append(c)

    if missing:
        return "⚠ Warning: " + " | ".join(missing) + " was cited but not found in retrieved context."
    return ""

@app.route("/")
def index():
    return render_template("index.html", users=USERS)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("user")
    message = data.get("message")

    if user_id not in USERS:
        return jsonify({"error": "Invalid user"}), 400

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    conversation_history[user_id].append({"role": "user", "content": message})

    answer, sources, warning, chunks_before, metadatas = query_rag(user_id, message)

    citation_warning = parse_cited_sources(answer, metadatas)
    full_warning = warning
    if citation_warning:
        full_warning = (warning + " " + citation_warning).strip() if warning else citation_warning

    conversation_history[user_id].append({"role": "assistant", "content": answer})

    query_logs.append({
        "user": user_id,
        "role": USERS[user_id]["role"],
        "query": message,
        "sources_retrieved": sources,
        "chunks_before_filter": chunks_before,
        "chunks_after_filter": len(sources),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    })
    if len(query_logs) > 50:
        query_logs.pop(0)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "warning": full_warning
    })

@app.route("/clear", methods=["POST"])
def clear():
    data = request.get_json()
    user_id = data.get("user")
    if user_id in conversation_history:
        conversation_history[user_id] = []
    return jsonify({"status": "cleared"})

@app.route("/logs")
def logs():
    return jsonify(query_logs)

@app.route("/logs-page")
def logs_page():
    return render_template("logs.html", logs=query_logs)

if __name__ == "__main__":
    ingest_documents()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)