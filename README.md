# NEURA Forge Chat — Magyar AI Chat Alkalmazás

**Verzió:** 1.0  
**Szerző:** Zsombi & Hermes Agent (Nous Research)  
**Státusz:** Működő prototípus

---

## Leírás

A **NEURA Forge Chat** egy magyar nyelvű AI chat alkalmazás, amely a **NEURA 300M** modellre épül. Támogat webes (Flask) és parancssori (CLI) felületet, valamint demo módot modell nélküli teszteléshez. A projekt tartalmazza a magyar nyelvű konverzációs adatok generálását és a modell finomhangolását is.

---

## Fájlszerkezet

```
forge_chat/
│
├── run.py                      # Fő indító script (web, cli, both módok)
├── cli.py                      # Parancssori chat felület
├── __init__.py                 # Csomag inicializálás
├── requirements.txt            # Python függőségek
│
├── engine/                     # Chat motor
│   └── chat_engine.py          # Fő chat motor (üzenet kezelés, válaszgenerálás)
│
├── models/                     # Modell réteg
│   ├── forge_model.py          # NEURA modell wrapper
│   ├── receiver.py             # Modell betöltő / fogadó
│   └── lm300m_v3_step390000.pt # NEURA 300M checkpoint
│
├── data/                       # Adat réteg
│   └── conversations.db        # Konverzációs adatbázis
│
├── web/                        # Webes felület
│   ├── app.py                  # Flask alkalmazás
│   └── templates/
│       └── chat.html           # Chat HTML felület
│
└── generate_hungarian_data.py  # Magyar adat generáló (külön fájl)
```

---

## Használat

### Telepítés

```bash
pip install -r requirements.txt
```

### Webes felület indítása

```bash
# Alapértelmezett: web mód, 5000-es port, localhost
python run.py

# Egyedi port és host
python run.py web --port 8080 --host 0.0.0.0

# Demo mód (modell nélkül)
python run.py web --demo
```

### Parancssori (CLI) mód

```bash
python run.py cli
```

### Web + CLI egyidejűleg

```bash
python run.py both
```

### Egyedi modell betöltése

```bash
# Saját checkpoint
python run.py web --model /eleresi/ut/checkpoint.pt

# Assistant checkpoint használata
python run.py web --assistant
```

### Magyar adatok generálása

```bash
python generate_hungarian_data.py
```

---

## Parancssori kapcsolók

| Kapcsoló | Leírás | Alapértelmezett |
|----------|--------|----------------|
| `mode` | `web`, `cli`, `both` | `web` |
| `--port` | Web szerver port | `5000` |
| `--host` | Web szerver host | `127.0.0.1` |
| `--debug` | Flask debug mód | Ki |
| `--demo` | Demo mód (modell nélkül) | Ki |
| `--model` | Egyedi checkpoint elérési út | — |
| `--assistant` | Assistant checkpoint | Ki |

---

## Függőségek

- **Python** 3.10+
- **Flask** ≥ 3.0.0 — web szerver
- **PyTorch** ≥ 2.0.0 — modell futtatás
- **SentencePiece** ≥ 0.1.99 — tokenizer
- **NumPy** ≥ 1.24.0

---

## Fejlesztő

Zsombi & Hermes Agent (Nous Research) (AI asszisztens segítségével)
