# neura-forge-chat

Web and CLI chat application for interacting with Neura Hungarian language model checkpoints.

## Overview & Purpose
neura-forge-chat provides user-facing web and command-line interfaces for loading, querying, and interacting with local language model weights.

## Key Features
- Web interface for interactive chat sessions.
- Terminal CLI mode for quick inference testing.
- Configurable decoding parameters (Temperature, Top-P, Top-K).

## Tech Stack & Dependencies
- **Language**: Python 3.9+
- **Web UI**: Streamlit / Gradio
- **Inference Engine**: PyTorch, Transformers

## Project Structure
```text
neura-forge-chat/
├── app.py
├── cli.py
├── requirements.txt
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.9+

### Steps
```bash
git clone https://github.com/zsomborturcsanyi7-lang/neura-forge-chat.git
cd neura-forge-chat
pip install -r requirements.txt
python app.py
```

## Usage Examples
```bash
python cli.py --model-path ./weights
```

## Status & License
Status: Functional Interface Prototype.
License: MIT
