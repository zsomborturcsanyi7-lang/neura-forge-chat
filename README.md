# NEURA Forge Chat — Hungarian AI Chat Application

**Status:** ⚠️ Prototype — chat demo tested on CPU, catastrophic forgetting on small dataset


**Version:** 1.0  
**Author:** Zsombi & Hermes Agent (Nous Research)  
**Status:** Working prototype

## ⚠️ THIS PROJECT IS UNFINISHED — FEEL FREE TO CONTINUE IT ⚠️

**Ez a projekt NINCS KÉSZEN. Bárki folytathatja, aki akarja!**
Ezt a projektet Zsombi & Hermes Agent (Nous Research) közösen fejlesztette, de egyik projekt sincs 100%-osan befejezve. Ha tetszik az ötlet és tovább fejlesztenéd, nyugodtan fork-old, folytasd, és csinálj belőle valami nagyszerűt!

---


---

## Description

**NEURA Forge Chat** is a Hungarian-language AI chat application built on the **NEURA 300M** model. It supports a web-based (Flask) interface, a command-line (CLI) interface, and a demo mode for testing without a model. The project also includes Hungarian conversational data generation and model fine-tuning.

---

## File Structure

```
forge_chat/
│
├── run.py                      # Main launcher script (web, cli, both modes)
├── cli.py                      # Command-line chat interface
├── __init__.py                 # Package initialization
├── requirements.txt            # Python dependencies
│
├── engine/                     # Chat engine
│   └── chat_engine.py          # Main chat engine (message handling, response generation)
│
├── models/                     # Model layer
│   ├── forge_model.py          # NEURA model wrapper
│   ├── receiver.py             # Model loader / receiver
│   └── lm300m_v3_step390000.pt # NEURA 300M checkpoint
│
├── data/                       # Data layer
│   └── conversations.db        # Conversation database
│
├── web/                        # Web interface
│   ├── app.py                  # Flask application
│   └── templates/
│       └── chat.html           # Chat HTML interface
│
└── generate_hungarian_data.py  # Hungarian data generator (separate file)
```

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Starting the Web Interface

```bash
# Default: web mode, port 5000, localhost
python run.py

# Custom port and host
python run.py web --port 8080 --host 0.0.0.0

# Demo mode (without model)
python run.py web --demo
```

### Command-Line (CLI) Mode

```bash
python run.py cli
```

### Web + CLI Simultaneously

```bash
python run.py both
```

### Loading a Custom Model

```bash
# Custom checkpoint
python run.py web --model /path/to/checkpoint.pt

# Use assistant checkpoint
python run.py web --assistant
```

### Generating Hungarian Data

```bash
python generate_hungarian_data.py
```

---

## Command-Line Switches

| Switch | Description | Default |
|----------|--------|----------------|
| `mode` | `web`, `cli`, `both` | `web` |
| `--port` | Web server port | `5000` |
| `--host` | Web server host | `127.0.0.1` |
| `--debug` | Flask debug mode | Off |
| `--demo` | Demo mode (no model) | Off |
| `--model` | Custom checkpoint path | — |
| `--assistant` | Assistant checkpoint | Off |

---

## Dependencies

- **Python** 3.10+
- **Flask** ≥ 3.0.0 — web server
- **PyTorch** ≥ 2.0.0 — model inference
- **SentencePiece** ≥ 0.1.99 — tokenizer
- **NumPy** ≥ 1.24.0

---

## Developer

Zsombi & Hermes Agent (Nous Research)
