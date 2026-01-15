# Local RAG Agent 🤖

**Inteligencja AI** - System Retrieval-Augmented Generation z lokalnych dokumentów.

## 📋 Opis

Profesjonalny agent RAG wykorzystujący:
- **Ollama** - Lokalne modele AI (Llama 3)
- **LangChain** - Framework do aplikacji LLM
- **ChromaDB** - Baza wektorowa
- **PyPDF** - Przetwarzanie dokumentów

## 🛠️ Instalacja

### 1. Zainstaluj Ollama

```bash
# Pobierz ze strony: https://ollama.ai
# Następnie pobierz model:
ollama pull llama3
```

### 2. Zainstaluj zależności Python

```bash
pip install -r requirements.txt
```

## 🚀 Użycie

### Krok 1: Dodaj dokumenty

Umieść pliki w folderze `docs/`:
- **PDF** - dokumenty PDF
- **Markdown** - pliki .md (np. z Data Grinder)

### Krok 2: Przetwórz dokumenty

```bash
python ingest.py
```

**Co robi ingest.py:**
- Wczytuje PDF-y i MD z `docs/`
- Dzieli na fragmenty (1000 znaków)
- Tworzy embeddingi (Llama 3)
- Zapisuje w ChromaDB

### Krok 3: Zadawaj pytania

```bash
python main.py
```

**Interfejs CLI:**
- Wpisz pytanie → Enter
- `stats` - statystyki bazy
- `exit` - wyjście

## 📁 Struktura

```
Local_RAG_Agent/
├── config.py              # Konfiguracja systemu
├── ingest.py             # Ingestia docs → ChromaDB
├── rag_service.py        # Logika RAG
├── main.py               # Interfejs CLI
├── requirements.txt      # Zależności
│
├── docs/                 # 📂 Dokumenty (input)
│   ├── *.pdf            # Pliki PDF
│   └── *.md             # Pliki Markdown
│
└── chroma_db/           # Baza wektorowa (output)
```

## 📥 Importowanie danych z Data Grinder

### Automatyczne kopiowanie

```bash
# Z folderu Data_Grinder
Copy-Item "output\*.md" -Destination "..\Local_RAG_Agent\docs\"

# Następnie przetwórz
cd ..\Local_RAG_Agent
python ingest.py
```

### Workflow

```
Data_Grinder/output/*.md  
         ↓
Local_RAG_Agent/docs/*.md
         ↓
python ingest.py
         ↓
chroma_db/ (baza wektorowa)
         ↓
python main.py (zadawaj pytania)
```

## ⚙️ Konfiguracja

Edytuj [config.py](config.py):

### Model
- `LLM_MODEL` - model Ollama (llama3)
- `EMBEDDING_MODEL` - model embeddingów (llama3)

### Parametry LLM
- `LLM_TEMPERATURE` - losowość (0.1)
- `LLM_MAX_TOKENS` - maks. długość odpowiedzi (2048)

### Retrieval
- `RETRIEVER_K` - liczba fragmentów (4)
- `RETRIEVER_SEARCH_TYPE` - typ wyszukiwania (similarity)

### Chunking
- `CHUNK_SIZE` - rozmiar chunka (1000)
- `CHUNK_OVERLAP` - overlap (200)

## 🎯 Prompt systemowy

RAG używa **rygorystycznego promptu** zapobiegającego halucynacjom:

✅ Odpowiada TYLKO na podstawie kontekstu  
✅ Nie używa wiedzy ogólnej  
✅ Przyznaje się gdy nie ma informacji  
✅ Cytuje źródła  

## 🔧 Rozwiązywanie problemów

### Ollama nie odpowiada

```bash
# Sprawdź czy działa
# Local RAG Agent (bez scrapera)

Pytaj własne dokumenty lokalnie przez Ollama.

## Szybki start

```bash
cd Local_RAG_Agent
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull llama3
ollama pull nomic-embed-text

# Dla Markdown
..\.venv\Scripts\python.exe ingest_md.py

# Dla PDF
..\.venv\Scripts\python.exe ingest.py

..\.venv\Scripts\python.exe main.py
```

## Konfiguracja

Plik `config.py`:
- `LLM_MODEL` - model generujący (domyślnie llama3)
- `EMBEDDING_MODEL` - embeddingi (domyślnie nomic-embed-text)
- `RETRIEVER_K` - liczba fragmentów kontekstu
- `LLM_TEMPERATURE` - temperatura LLM

## Struktura

```
Local_RAG_Agent/
├── docs/          # tu wrzucasz pliki .md / .pdf
├── chroma_db/     # baza wektorowa
├── ingest_md.py   # ingest markdown
├── ingest.py      # ingest pdf
├── main.py        # interfejs Q&A
└── config.py
```
- "Jak skonfigurować system?"
- "Jakie są wymagania systemowe?"
- "Opisz proces instalacji"

### Knowledge base ze stron WWW

```bash
# 1. Użyj Data Grinder do scrapowania
cd ..\Data_Grinder
python orchestrator.py full https://docs.example.com

# 2. Skopiuj do docs/
Copy-Item "output\*.md" -Destination "..\Local_RAG_Agent\docs\"

# 3. Wróć do RAG Agent
cd ..\Local_RAG_Agent
python ingest.py

# 4. Zadawaj pytania
python main.py
```

## 🚀 Zaawansowane

### Dodawanie nowych dokumentów

```bash
# 1. Dodaj nowe pliki do docs/
# 2. Usuń starą bazę
Remove-Item -Recurse -Force chroma_db

# 3. Przebuduj
python ingest.py
```

### Zmiana modelu

```bash
# Pobierz inny model
ollama pull mistral

# Zmień w config.py
LLM_MODEL = "mistral"
```

### Optymalizacja retrieval

```python
# W config.py
RETRIEVER_K = 6  # Więcej fragmentów
CHUNK_SIZE = 500  # Mniejsze chunki
CHUNK_OVERLAP = 100
```

## 📝 Format dokumentów

### Wspierane:
- ✅ PDF (przez PyPDF)
- ✅ Markdown (natywnie)
- ✅ TXT (jako Markdown)

### Rekomendacje:
- PDF: dokumentacja, raporty
- Markdown: artykuły, documentation (z Data Grinder)

---

**🔥 Agent gotowy do pracy z Twoją bazą wiedzy!**

Pytania? Sprawdź [config.py](config.py)
