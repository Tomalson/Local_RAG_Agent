# 🚀 QUICK START — Tylko Local RAG Agent (bez scrapera)

## 📋 Wymagania

```bash
# Python 3.10+
python --version

# Ollama
# Pobierz: https://ollama.ai
ollama pull llama3
ollama pull nomic-embed-text   # embedding model
```

Pracujemy wyłącznie w folderze `Local_RAG_Agent/`.

## Krok 1: Instalacja zależności

```bash
cd Local_RAG_Agent
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Krok 2: Dodaj dokumenty

- Markdown: wrzuć `.md` do `docs/`
- PDF: wrzuć `.pdf` do `docs/`

## Krok 3: Zbuduj bazę

```bash
# Dla Markdown
..\.venv\Scripts\python.exe ingest_md.py

# Dla PDF
..\.venv\Scripts\python.exe ingest.py
```

## Krok 4: Zapytaj RAG

```bash
..\.venv\Scripts\python.exe main.py
```

Przykład interfejsu:
```
❓ Twoje pytanie: Opisz proces instalacji
💡 ODPOWIEDŹ: [z kontekstu]
📚 ŹRÓDŁA:
  1. docs_1.md
```

## 🔧 Konfiguracja kluczowa

Plik [Local_RAG_Agent/config.py](Local_RAG_Agent/config.py):
- `EMBEDDING_MODEL = "nomic-embed-text"`
- `LLM_MODEL = "llama3"`
- `RETRIEVER_K = 4`, `LLM_TEMPERATURE = 0.1`
- Katalog bazy: `chroma_db/`

## 🆘 Szybkie naprawy

- Ollama nie odpowiada / connection refused:
  - Zamknij i uruchom ponownie aplikację Ollama, potem `ollama list`.
- Brak embeddingów / CUDA error na llama3:
  - Używaj `nomic-embed-text` (już ustawione w config).
- Chcesz wyczyścić bazę i wgrać od nowa:
  - Usuń zawartość `chroma_db/` i ponownie uruchom `ingest_md.py` lub `ingest.py`.

## 🎯 Minimalny demo

```bash
cd Local_RAG_Agent
echo "# Test\nTo jest testowy dokument" > docs/test.md
..\.venv\Scripts\python.exe ingest_md.py
..\.venv\Scripts\python.exe main.py
```

To wszystko — RAG bez scrapera. 🚀
