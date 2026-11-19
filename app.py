import os
import json
from flask import Flask,request,jsonify,render_template_string
from semantic_search import load_chunks,build_vectorizer
from agent import answer_question,load_env

# Load environment variables
load_env()

app=Flask(__name__)

# Initialize retrieval components once
CHUNKS_FILE="output/chunks.json"
chunks=load_chunks(CHUNKS_FILE)
vectorizer,matrix=build_vectorizer(chunks)

HTML_TEMPLATE="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PawPals Pet Clinic - Knowledge Agent</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 20px; }
        .container { max-width: 720px; margin: 40px auto; background: white; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        h1 { margin-top: 0; color: #0f172a; font-size: 24px; }
        p.subtitle { color: #64748b; margin-bottom: 24px; font-size: 15px; }
        .input-group { display: flex; gap: 8px; margin-bottom: 16px; }
        input[type="text"] { flex: 1; padding: 12px 16px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 15px; }
        button { background: #2563eb; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .samples { margin-bottom: 24px; }
        .sample-btn { background: #f1f5f9; border: 1px solid #e2e8f0; color: #475569; padding: 6px 12px; border-radius: 16px; font-size: 13px; margin: 4px; cursor: pointer; display: inline-block; }
        .sample-btn:hover { background: #e2e8f0; }
        .result-box { display: none; margin-top: 24px; padding: 20px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 700; margin-bottom: 12px; }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-fallback { background: #fef9c3; color: #854d0e; }
        .badge-error { background: #fee2e2; color: #991b1b; }
        .answer { line-height: 1.6; font-size: 15px; white-space: pre-wrap; }
        .sources { margin-top: 16px; font-size: 13px; color: #64748b; font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐾 PawPals Pet Clinic Knowledge Agent</h1>
        <p class="subtitle">Ask any question based on PawPals clinical care sheets and guidelines.</p>

        <div class="samples">
            <strong>Sample questions:</strong><br>
            <span class="sample-btn" onclick="askSample(this)">When should puppies get their core vaccines?</span>
            <span class="sample-btn" onclick="askSample(this)">What human foods are toxic to cats?</span>
            <span class="sample-btn" onclick="askSample(this)">How do I know if my pet has a medical emergency?</span>
            <span class="sample-btn" onclick="askSample(this)">How often should I brush my dog's teeth?</span>
        </div>

        <form id="askForm" onsubmit="handleAsk(event)">
            <div class="input-group">
                <input type="text" id="queryInput" placeholder="Type your pet care question..." required />
                <button type="submit" id="submitBtn">Ask Assistant</button>
            </div>
        </form>

        <div id="resultBox" class="result-box">
            <span id="statusBadge" class="badge"></span>
            <div id="answerText" class="answer"></div>
            <div id="sourcesText" class="sources"></div>
        </div>
    </div>

    <script>
        function askSample(el) {
            document.getElementById('queryInput').value = el.innerText;
            document.getElementById('askForm').dispatchEvent(new Event('submit'));
        }

        async function handleAsk(e) {
            e.preventDefault();
            const query = document.getElementById('queryInput').value;
            const btn = document.getElementById('submitBtn');
            const box = document.getElementById('resultBox');
            const badge = document.getElementById('statusBadge');
            const ans = document.getElementById('answerText');
            const src = document.getElementById('sourcesText');

            btn.disabled = true;
            btn.innerText = "Searching...";
            box.style.display = "none";

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await response.json();

                box.style.display = "block";
                ans.innerText = data.answer || "No response received.";
                src.innerText = data.sources && data.sources.length ? "Cited Sources: " + data.sources.join(", ") : "";

                badge.className = "badge " + (data.status === 'SUCCESS' ? 'badge-success' : (data.status === 'FALLBACK' ? 'badge-fallback' : 'badge-error'));
                badge.innerText = data.status || 'STATUS';
            } catch (err) {
                box.style.display = "block";
                badge.className = "badge badge-error";
                badge.innerText = "ERROR";
                ans.innerText = "Error connecting to service: " + err.message;
                src.innerText = "";
            } finally {
                btn.disabled = false;
                btn.innerText = "Ask Assistant";
            }
        }
    </script>
</body>
</html>
"""

@app.route("/",methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/ask",methods=["GET","POST"])
def ask():
    if request.method=="POST":
        data=request.get_json(silent=True) or {}
        query=data.get("query","")
    else:
        query=request.args.get("query") or request.args.get("q") or ""

    result=answer_question(query,vectorizer,matrix,chunks)
    return jsonify(result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=True)
