# 🏥 AI-Powered Healthcare Assistant for Village

A deployable AI health assistant combining a fine-tuned medical LLM with live PubMed RAG.

## Project Structure

```
AI Assistant Village/
├── app.py              # Flask backend + model inference + PubMed RAG
├── requirements.txt    # Python dependencies
├── config.env          # Model and server configuration
├── README.md           # This file
└── static/
    └── index.html      # Frontend UI
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure model (optional)
Edit `config.env` to change the model:
```
MODEL_ID=Rumiii/Qwen2.5-0.5B-MedReason-SFT      # HuggingFace
MODEL_ID=./Med-Qwen2.5-0.5B-MedReason-v3         # local path
```

### 3. Launch
```bash
python app.py
```

### 4. Open browser
```
http://localhost:7860
```

## Features
- Fine-tuned medical LLM (Qwen2.5-0.5B-MedReason-SFT)
- Live PubMed API RAG — fetches relevant research for each query
- Clean ChatGPT-style interface with large readable text
- Example question chips for easy use
- Works on CPU and GPU

## Disclaimer
For educational purposes only. Not a substitute for professional medical advice.
