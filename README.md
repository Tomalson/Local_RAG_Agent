# 🤖 Local RAG Agent

> **Profesjonalny system Retrieval-Augmented Generation z lokalnymi dokumentami**
> 
> Inteligentny asystent AI oparty na LLaMA 3 z lokalną bazą wektorową ChromaDB.
> Nie wysyłaj danych do chmury — wszystko działa lokalnie! 🔒

---

## ✨ Cechy

- ✅ **Całkowicie lokalny** - bez wysyłania danych do chmury
- ✅ **Szybki** - wyszukiwanie wektorowe z ChromaDB
- ✅ **Inteligentny** - LLaMA 3 z Query Decomposition i Hybrid Search
- ✅ **Elastyczny** - obsługuje PDF i Markdown
- ✅ **Łatwy w konfiguracji** - jeden plik `config.py`
- ✅ **Bez API** - nie potrzebujesz klucza OpenAI

---

## 🚀 Szybki Start (5 minut)

### 1. Wymagania
- **Python 3.10+**
- **Ollama** (https://ollama.ai) z modelami:
  ```bash
  ollama pull llama3              # Model QA/generacyjny
  ollama pull nomic-embed-text    # Model embeddingów
  ```

### 2. Instalacja
```bash
cd Local_RAG_Agent
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 3. Dodaj dokumenty
Umieść pliki w `Local_RAG_Agent/docs/`:
- **PDF**: `*.pdf`
- **Markdown**: `*.md`

### 4. Zbuduj bazę wektorową
```bash
# Markdown
python ingest_md.py

# PDF
python ingest.py
```

### 5. Zadawaj pytania
```bash
python main.py
```

**Przykład:**
```
❓ Pytanie: Opisz główne cechy systemu
💡 ODPOWIEDŹ: [Odpowiedź z dokumentów]
📚 ŹRÓDŁA:
   • document1.md
   • document2.pdf
```

---

## 📋 Pełna Dokumentacja

### Struktura projektu
```
Local_RAG_Agent/
├── docs/              # 📁 Wejściowe dokumenty (PDF, MD)
├── chroma_db/         # 🗄️ Baza wektorowa (utworzana automatycznie)
├── config.py          # ⚙️ Konfiguracja (modele, ścieżki, prompty)
├── main.py            # 🎯 CLI interfejs (main.py)
├── advanced_rag.py    # 🧠 Silnik RAG (Query Decomposition, Hybrid Search)
├── rag_service.py     # 🔧 Niskopoziomowe API
├── ingest_md.py       # 📄 Wczytywanie Markdown
├── ingest.py          # 📋 Wczytywanie PDF
├── requirements.txt   # 📦 Zależności
└── run.bat           # 🚀 Skrypt uruchamiający (Windows)
```

### Konfiguracja

Plik [Local_RAG_Agent/config.py](Local_RAG_Agent/config.py) zawiera:

| Ustawienie | Wartość | Opis |
|-----------|---------|------|
| `LLM_MODEL` | `llama3` | Model generacyjny |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model embeddingów |
| `RETRIEVER_K` | `8` | Liczba fragmentów do retrieve'u |
| `CHUNK_SIZE` | `700` | Rozmiar fragmentu w znakach |
| `LLM_TEMPERATURE` | `0.1` | Temperatura (niżej = bardziej deterministyczne) |

**Zmiana modelu LLaMA:**
```python
LLM_MODEL = "llama2"  # lub inne dostępne modele
```

Dostępne modele: `ollama list`

### Skrypty

#### `main.py` - Interaktywny interfejs
```bash
python main.py
```
- Pytania + odpowiedzi w CLI
- Automatyczne odkrywanie źródeł
- Statystyki bazy (`stats` komenda)

#### `ingest_md.py` - Wczytywanie Markdown
```bash
python ingest_md.py
```
- Skanuje `docs/*.md`
- Tworzy embeddingi
- Dodaje do ChromaDB

#### `ingest.py` - Wczytywanie PDF
```bash
python ingest.py
```
- Skanuje `docs/*.pdf`
- Ekstrakcja tekstu
- Tworzy embeddingi
- Dodaje do ChromaDB

---

## ⚙️ Zaawansowana Konfiguracja

### Query Decomposition
Agent automatycznie rozbija złożone pytania:
```
Oryginalnie pytanie:
  "Jak zainstalować i konfigurować system?"
↓
Rozłożone na:
  1. "Jak zainstalować system?"
  2. "Jak skonfigurować system?"
  3. "Jakie są wymagania systemowe?"
```

### Hybrid Search
Kombinacja:
- **Lexical Search** - dopasowanie słów kluczowych
- **Semantic Search** - dopasowanie znaczenia (wektory)
- **MMR (Maximum Marginal Relevance)** - dywersyfikacja wyników

### Context Expansion
Automatyczne powiększanie kontekstu:
- Wyszukiwanie sąsiednich fragmentów
- Rozszeranie okna kontekstu
- Lepsze źródła dla LLM

---

## 🆘 Troubleshooting

### Ollama Connection Refused
```bash
# Sprawdzenie czy Ollama działa
ollama list

# Jeśli nie działa:
# 1. Zamknij aplikację Ollama
# 2. Uruchom ponownie
# 3. Spróbuj jeszcze raz
```

### Brak modelu
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### CUDA/Memory Error
```python
# config.py - zmień na CPU-friendly model
EMBEDDING_MODEL = "nomic-embed-text"  # Lekki model
LLM_TEMPERATURE = 0.1  # Mniej hallucynacji
```

### Wyczyść bazę i załaduj od nowa
```bash
# 1. Usuń starą bazę
rm -r Local_RAG_Agent/chroma_db

# 2. Załaduj dokumenty ponownie
python ingest_md.py
# lub
python ingest.py
```

### Dokumenty nie są wczytywane
- Sprawdź czy pliki są w `Local_RAG_Agent/docs/`
- Sprawdź czy rozszerzenia to `.md` lub `.pdf`
- Spróbuj: `python ingest_md.py -v` (verbose mode)

---

## 🏗️ Architektura

```
┌─────────────────────────────────┐
│      main.py (CLI)              │
│  Interactive Q&A Interface      │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│    advanced_rag.py              │
│  AdvancedRAGAgent               │
│  ├─ Query Decomposition         │
│  ├─ Hybrid Search               │
│  └─ Context Expansion           │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│    rag_service.py               │
│  RAG Service (Low-level)        │
│  ├─ Vector Store (ChromaDB)     │
│  ├─ Embeddings (Ollama)         │
│  └─ LLM (Ollama)                │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│  Ollama (Local LLM)             │
│  - LLaMA 3 (LLM)                │
│  - Nomic Embed (Embeddings)     │
└─────────────────────────────────┘
```

---

## 📚 Przykłady Użycia

### Przykład 1: Dokumentacja API
```bash
# 1. Umieść dokumentację w docs/
cp project-docs/*.md Local_RAG_Agent/docs/

# 2. Załaduj
python ingest_md.py

# 3. Pytaj
python main.py
❓ Pytanie: Jak się uwierzytelniać w API?
```

### Przykład 2: Dokumenty PDF
```bash
# 1. Umieść PDF
cp reports/*.pdf Local_RAG_Agent/docs/

# 2. Załaduj
python ingest.py

# 3. Pytaj
python main.py
❓ Pytanie: Jaki był przychód w Q4 2024?
```

### Przykład 3: Mieszane źródła
```bash
# Masz zarówno MD jak i PDF
python ingest_md.py  # wczyta *.md
python ingest.py     # wczyta *.pdf
python main.py       # pyta z obu
```

---

## 🔒 Bezpieczeństwo i Prywatność

✅ **Wszystko lokalnie**
- Nie wysyłamy danych do chmury
- Nie ma zaufania do żadnych API
- Pełna kontrola nad danymi

✅ **Bez API Keys**
- Nie potrzebujesz OpenAI, Claude, itp.
- Brak ryzyka wycieków kluczy

✅ **GDPR Compliant**
- Dane nigdy nie opuszczają Twojego komputera
- Idealne dla poufnych dokumentów

---

## 📖 Zasoby

- **Ollama**: https://ollama.ai
- **ChromaDB**: https://docs.trychroma.com
- **LangChain**: https://python.langchain.com
- **LLaMA**: https://llama.meta.com

---

## 📝 Licencja

MIT License - patrz [LICENSE](LICENSE)

---

## 🤝 Wkład

Znaleźliście bug? Macie ideę?
- Otwórzcie Issue 🐛
- Wyślijcie Pull Request 🚀

---

## ❓ FAQ

**P: Czy mogę używać inne modele?**
A: Tak! Zmień `LLM_MODEL` w `config.py`. Lista: `ollama list`

**P: Jaka jest minimalna RAM?**
A: ~4GB dla LLaMA 3. Więcej RAM = szybciej.

**P: Czy działa na GPU?**
A: Tak! Ollama automatycznie użyje GPU jeśli jest dostępne.

**P: Czy mogę użyć więcej niż jednej kolekcji dokumentów?**
A: Tak, w `config.py` zmień `CHROMA_COLLECTION_NAME`.

**P: Jak szybko są odpowiedzi?**
A: 5-15 sekund zależy od GPU, CPU i rozmiaru kontekstu.

---

**Czekamy na Ciebie! 🚀 Jeśli podoba Ci się projekt, daj ⭐**

Ostatnia aktualizacja: 2026-01-15
