# NEURA Forge Chat — Web+CLI chat felület a NEURA 300M magyar LM-hez

**Status:** ⚠️ Prototype — chat demo tesztelve CPU-n, katasztrofális felejtés kis adathalmazon

Webes (Flask) és CLI chat felület a 355M paraméteres NEURA magyar nyelvmodellhez. Tartalmaz magyar konverzációs adatgenerálást és modell finetuningot.

## ⚠️ THIS PROJECT IS UNFINISHED — FEEL FREE TO CONTINUE IT ⚠️

**Ez a projekt NINCS KÉSZEN. Bárki folytathatja, aki akarja!**
Ezt a projektet Zsombi & Hermes Agent (Nous Research) közösen fejlesztette, de egyik projekt sincs 100%-osan befejezve.

---

## Struktúra

| Könyvtár/Fájl | Leírás |
|------|--------|
| `web/` | Flask web felület |
| `cli.py` | Parancssori chat |
| `engine/` | Chat engine |
| `models/` | Modell fájlok |
| `data/` | Adatok |
| `run.py` | Fő belépési pont |

## Fejlesztő
Zsombi & Hermes Agent (Nous Research)
