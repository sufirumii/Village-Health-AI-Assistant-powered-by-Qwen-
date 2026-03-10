# Village Health AI

A 0.5B medical AI assistant with live PubMed RAG for low-resource healthcare settings.

## Overview

Village Health AI is designed for environments where large models are not practical — rural clinics, community health programs, and offline deployments on affordable hardware. It combines a compact fine-tuned medical reasoning model with real-time PubMed research to produce structured, evidence-backed clinical responses.

The core insight is simple: a small model grounded in live medical literature consistently outperforms a large model answering from memory alone.

## How It Works

1. A user submits a health question through the chat interface
2. The question is forwarded to the free PubMed API which retrieves relevant research abstracts
3. Those abstracts are passed to the model as context alongside the question
4. The model generates a structured clinical response grounded in real published literature
5. The answer is returned indicating whether PubMed research was used

## Model

The assistant is powered by a fine-tuned version of Qwen2.5-0.5B-Instruct, trained on clinical question-answer pairs with chain-of-thought reasoning traces using QLoRA on a single T4 GPU.

**Fine-tuned model:** https://huggingface.co/Rumiii/Qwen2.5-0.5B-MedReason-SFT

| Stage | Base Model | Dataset | Samples |
|---|---|---|---|
| SFT | Qwen2.5-0.5B-Instruct | OpenMed Medical-Reasoning-SFT | 124,520 |

## Project Structure

```
AI Assistant Village/
├── launch.py           Main application — model inference, PubMed RAG, Gradio UI
├── app.py              Flask backend for local deployment
├── requirements.txt    Python dependencies
├── config.env          Model and server configuration
└── static/
    └── index.html      Frontend UI for Flask deployment
```

## Quick Start

### Google Colab

```python
!git clone https://github.com/sufirumii/Village-Health-AI-Assistant-powered-by-Qwen-
!pip install -r "/content/Village-Health-AI-Assistant-powered-by-Qwen-/requirements.txt"
!python "/content/Village-Health-AI-Assistant-powered-by-Qwen-/launch.py"
```

A public Gradio link will appear in the output. Open it in any browser.

### Local Machine

```bash
git clone https://github.com/sufirumii/Village-Health-AI-Assistant-powered-by-Qwen-
cd Village-Health-AI-Assistant-powered-by-Qwen-
pip install -r requirements.txt
python launch.py
```

## Configuration

Edit `config.env` to point to a different model:

```
MODEL_ID=Rumiii/Qwen2.5-0.5B-MedReason-SFT   # default HuggingFace model
MODEL_ID=./your-local-model-path               # or a local path
```

## Requirements

- Python 3.10+
- 8GB RAM minimum (GPU recommended)
- Free Colab T4 GPU works well
- No PubMed API key required

## Features

- Live PubMed API integration at inference time — no API key, no cost
- Structured clinical responses covering assessment, recommended action, and warning signs
- Clean chat interface with example questions
- Runs on a free Colab GPU or modest local hardware
- Fully open source under Apache 2.0

## Limitations

- At 0.5B parameters, the model has a knowledge ceiling and will occasionally produce inaccurate statements
- PubMed context significantly reduces hallucinations but does not eliminate them
- Not intended for direct clinical decision-making
- All outputs must be reviewed by a qualified medical professional before any clinical consideration

## Disclaimer

This project is for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. All AI-generated responses must be verified by a licensed healthcare professional.

## License

Apache 2.0
