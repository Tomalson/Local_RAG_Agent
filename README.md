# 🤖 Local RAG Agent

> **Lokalny system RAG do analizy dokumentów PDF**
> 
> Zadawaj pytania swoim dokumentom PDF używając lokalnego AI (Ollama + LLaMA 3).
> Wszystko działa na Twoim komputerze - żadnych danych w chmurze! 🔒

## ⚠️ Ważne

To narzędzie używa **darmowej wersji Ollama z modelem LLaMA 3**, który działa lokalnie na Twoim komputerze. 
**To nie jest ChatGPT ani Gemini** - odpowiedzi są wolniejsze i mniej precyzyjne, ale wszystko pozostaje prywatne.

---

## ✨ Co robi?

- 📄 **Głównie czyta PDF-y** - dokumenty, raporty, książki
- 📝 **Dodatkowo obsługuje Markdown** - pliki .md jako bonus
- 💬 Zadajesz pytanie → dostaniesz odpowiedź z dokumentów
- 🔍 Pokazuje źródła - skąd wzięła się odpowiedź
- 🔒 **100% prywatne** - żadne dane nie wychodzą z komputera

---

## 🚀 Szybki Start

### 1. Zainstaluj Ollama
Pobierz z https://ollama.ai i zainstaluj modele:
```bash
ollama pull llama3              # Główny model AI
ollama pull nomic-embed-text    # Model do embeddingów
```

### 2. Zainstaluj zależności Python
```bash
cd Local_RAG_Agent
pip install -r requirements.txt
```

### 3. Wrzuć swoje PDF-y
Skopiuj pliki do `Local_RAG_Agent/docs/`:
- **PDF** - główny format (dokumenty, raporty, książki)
- **MD** - opcjonalnie pliki markdown

### 4. Załaduj dokumenty do systemu
```bash
python ingest.py         # Dla PDF
python ingest_md.py      # Dla Markdown (opcjonalnie)
```

### 5. Zadawaj pytania!
```bash
python main.py
```

**Przykład:**
```
❓ Pytanie: O czym jest dokument?
💡 ODPOWIEDŹ: [Odpowiedź na podstawie PDF-ów]
📚 ŹRÓDŁA: document.pdf
```

---

## 📁 Struktura
```
Local_RAG_Agent/
├── docs/              # Tu wrzucasz PDF-y i MD
├── chroma_db/         # Baza wektorowa (tworzy się sama)
├── main.py            # Program do zadawania pytań
├── ingest.py          # Wczytywanie PDF
├── ingest_md.py       # Wczytywanie MD (opcja)
├── config.py          # Ustawienia
└── requirements.txt   # Zależności Python
```

---

## ⚙️ Podstawowa Konfiguracja

W [Local_RAG_Agent/config.py](Local_RAG_Agent/config.py) możesz zmienić:

- `LLM_MODEL` - model AI (domyślnie `llama3`)
- `CHUNK_SIZE` - rozmiar fragmentów tekstu (domyślnie 700)
- `RETRIEVER_K` - ile fragmentów wyszukiwać (domyślnie 8)

**Dostępne modele:**
```bash
ollama list  # Zobacz zainstalowane modele
```

## 🆘 Najczęstsze Problemy

### Ollama nie działa
```bash
ollama list  # Sprawdź czy działa
# Jeśli nie - uruchom ponownie aplikację Ollama
```

### Brak modelu
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Reset bazy danych
```bash
rm -r Local_RAG_Agent/chroma_db  # Usuń bazę
python ingest.py                  # Załaduj ponownie
```

---

## ❓ FAQ

**P: Dlaczego odpowiedzi są wolne?**
A: Ollama działa lokalnie na Twoim CPU/GPU. To nie jest ChatGPT w chmurze - wymaga więcej czasu, ale zachowuje prywatność.

**P: Czy mogę używać innych modeli?**
A: Tak! Zobacz dostępne: `ollama list`. Zmień w `config.py`.

**P: Ile RAM potrzebuję?**
A: Minimum 4-8GB. LLaMA 3 jest dość wymagający.

**P: Dlaczego odpowiedzi są mniej dokładne niż ChatGPT?**
A: LLaMA 3 (szczególnie w wersji lokalnej) ma swoje ograniczenia. Jest mniejszy i działa offline, więc nie będzie tak precyzyjny jak duże modele komercyjne.

**P: Czy obsługuje tylko PDF?**
A: Głównie PDF (to priorytet), ale dodatkowo może też czytać pliki Markdown (.md).

---

## 📖 Technologie

- **Ollama** - Lokalne modele AI
- **LLaMA 3** - Model językowy
- **ChromaDB** - Baza wektorowa
- **LangChain** - Framework RAG
- **PDFPlumberLoader** - Czytanie PDF

---

**Made for privacy**
