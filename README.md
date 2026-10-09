Secure Multi-Modal RAG Chatbot with Access Control
====================================================

Prerequisites:
- Python 3.10+
- Tesseract OCR (for image OCR; optional - falls back to Gemini Vision if unavailable)

Setup:

1. Set your Gemini API key in .env:
   GEMINI_API_KEY=your_actual_key_here

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   python app.py

4. Open in browser:
   http://127.0.0.1:5000

5. Test the logs page:
   http://127.0.0.1:5000/logs-page

API Endpoints:
- GET  /            → Chatbot UI
- POST /chat        → { "user": "alice", "message": "..." }
- POST /clear       → { "user": "alice" } — clear chat history
- GET  /logs        → JSON list of last 50 query logs
- GET  /logs-page   → HTML table of query logs

Users:
- alice   (hr role)
- bob     (manager role)
- charlie (engineer role)
- dave    (intern role)

Data sources:
- sample_docs/doc1.pdf  (access: hr, manager)
- sample_docs/doc2.pdf  (access: all roles)
- sample_docs/doc3.png  (access: engineer, manager)
- database/employee_db.sqlite (per-row access control)