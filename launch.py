import gradio as gr
import requests, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Rumiii/Qwen2.5-0.5B-MedReason-SFT"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()
print("Model loaded!")

def fetch_pubmed(query):
    try:
        ids = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query,"retmax":3,"retmode":"json"}, timeout=8
        ).json()["esearchresult"]["idlist"]
        if not ids: return ""
        text = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db":"pubmed","id":",".join(ids),"rettype":"abstract","retmode":"text"}, timeout=10
        ).text.strip()
        return text[:1200]
    except: return ""

def ask(question, history):
    pubmed = fetch_pubmed(question)
    user_content = f"Medical Literature:\n{pubmed}\n\nQuestion: {question}" if pubmed else question
    messages = [
        {"role": "system", "content": "You are a knowledgeable medical assistant for rural communities. Provide clear, accurate, evidence-based answers. Structure with: Assessment, Recommended Action, Warning Signs."},
        {"role": "user", "content": user_content}
    ]
    # Fix: tokenize returns dict, extract input_ids tensor properly
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.3,
            top_p=0.85,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    pubmed_note = "\n\n📚 *Answer enhanced with live PubMed research*" if pubmed else ""
    return response + pubmed_note

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500&display=swap');

body, .gradio-container {
    background: #f8fdf9 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
}
#title-block {
    background: linear-gradient(135deg, #0d3b2e 0%, #1a6b4a 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
#title-block h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 42px !important;
    color: white !important;
    margin-bottom: 10px !important;
    line-height: 1.2 !important;
}
#title-block p {
    color: rgba(255,255,255,0.75) !important;
    font-size: 17px !important;
    font-weight: 300 !important;
}
.pubmed-badge {
    display: inline-block;
    background: rgba(245,166,35,0.2);
    border: 1px solid rgba(245,166,35,0.4);
    border-radius: 8px;
    padding: 6px 14px;
    color: #f5a623;
    font-size: 14px;
    font-weight: 500;
    margin-top: 14px;
}
#chatbot {
    border-radius: 16px !important;
    border: 1px solid #ddeee6 !important;
    background: white !important;
    font-size: 17px !important;
    min-height: 460px !important;
    box-shadow: 0 4px 32px rgba(13,59,46,0.08) !important;
}
#chatbot .message {
    font-size: 20px !important;
    line-height: 1.75 !important;
    padding: 16px 20px !important;
}
#question textarea {
    font-size: 17px !important;
    border-radius: 12px !important;
    border: 1.5px solid #c5dfd0 !important;
    background: #e8f5ee !important;
    padding: 14px 18px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
#question textarea:focus {
    border-color: #1a6b4a !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(26,107,74,0.1) !important;
}
#send-btn {
    background: linear-gradient(135deg, #1a6b4a, #2d9b6f) !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(26,107,74,0.3) !important;
    min-width: 100px !important;
}
#send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(26,107,74,0.4) !important;
}
.disclaimer {
    text-align: center;
    color: #6b8c78;
    font-size: 13px;
    margin-top: 8px;
}
"""

with gr.Blocks(css=CSS, title="Village Health Assistant") as demo:

    gr.HTML("""
    <div id="title-block">
        <h1>Village Health Assistant</h1>
        <p>Ask any health question. Our AI combines medical expertise with live PubMed research to give you clear, trustworthy answers.</p>
        <div class="pubmed-badge">📚 Powered by PubMed Research + Fine-tuned Medical AI</div>
    </div>
    """)

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        placeholder="<div style='text-align:center;padding:40px;color:#6b8c78'><div style='font-size:48px'>🌿</div><div style='font-size:20px;font-family:serif;margin:12px 0'>How can I help you today?</div><div>Describe your symptoms or ask any health question</div></div>",
    )

    with gr.Row():
        question = gr.Textbox(
            placeholder="Describe symptoms or ask a health question...",
            show_label=False,
            elem_id="question",
            scale=5,
            lines=2,
        )
        submit = gr.Button("Send ➤", elem_id="send-btn", scale=1, variant="primary")

    gr.Examples(
        examples=[
            "Child has fever and cough for 3 days, what should I do?",
            "What are the early signs of diabetes?",
            "Severe chest pain radiating to left arm — emergency?",
            "Safe medicines during pregnancy for headache",
            "How to manage high blood pressure naturally?",
        ],
        inputs=question,
        label="Try these examples",
    )

    gr.HTML('<div class="disclaimer">⚕️ For educational purposes only · Not a substitute for professional medical advice · Always consult a doctor for serious conditions</div>')

    def respond(msg, history):
        if not msg.strip():
            return "", history
        reply = ask(msg, history)
        history.append((msg, reply))
        return "", history

    submit.click(respond, [question, chatbot], [question, chatbot])
    question.submit(respond, [question, chatbot], [question, chatbot])

demo.launch(share=True)
