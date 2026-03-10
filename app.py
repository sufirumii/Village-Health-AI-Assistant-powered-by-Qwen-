import os
import re
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__, static_folder="static")
CORS(app)

# ── CONFIG ───────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "Rumiii/Qwen2.5-0.5B-MedReason-SFT")
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── LOAD MODEL ───────────────────────────────────────────────
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
print("Model loaded.")

# ── PUBMED RAG ───────────────────────────────────────────────
def fetch_pubmed_context(query: str, max_results: int = 3) -> str:
    try:
        search_url = f"{PUBMED_BASE}/esearch.fcgi"
        search_params = {
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json"
        }
        search_resp = requests.get(search_url, params=search_params, timeout=8)
        ids = search_resp.json()["esearchresult"]["idlist"]
        if not ids:
            return ""

        fetch_url = f"{PUBMED_BASE}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed", "id": ",".join(ids),
            "rettype": "abstract", "retmode": "text"
        }
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=10)
        raw = fetch_resp.text.strip()
        # Keep only first 1200 chars to avoid context overflow
        return raw[:1200] if raw else ""
    except Exception as e:
        print(f"PubMed fetch error: {e}")
        return ""

# ── INFERENCE ────────────────────────────────────────────────
def generate_answer(question: str, pubmed_context: str) -> str:
    system_prompt = (
        "You are a knowledgeable and compassionate medical assistant for rural communities. "
        "Provide clear, accurate, evidence-based medical information. "
        "Structure your answer with: Diagnosis/Assessment, Recommended Action, Warning Signs to watch for. "
        "Keep language simple and easy to understand."
    )

    if pubmed_context:
        user_content = (
            f"Medical Literature Context:\n{pubmed_context}\n\n"
            f"Patient Question: {question}"
        )
    else:
        user_content = question

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens      = 512,
            temperature         = 0.2,
            do_sample           = True,
            repetition_penalty  = 1.3,
            no_repeat_ngram_size= 4,
            top_p               = 0.85,
            top_k               = 40,
            pad_token_id        = tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    # Strip think blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response

# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    pubmed_context = fetch_pubmed_context(question)
    answer         = generate_answer(question, pubmed_context)

    return jsonify({
        "answer":          answer,
        "pubmed_used":     bool(pubmed_context),
        "pubmed_preview":  pubmed_context[:300] if pubmed_context else "",
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
