"""
Konfiguracja projektu Local RAG Agent.

Agent AI do Retrieval-Augmented Generation z dokumentów.
"""

from pathlib import Path
from typing import Final

# ==================== ŚCIEŻKI ====================
PROJECT_ROOT: Final[Path] = Path(__file__).parent
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
CHROMA_DB_DIR: Final[Path] = PROJECT_ROOT / "chroma_db"

# Tworzenie katalogów jeśli nie istnieją
DOCS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# ==================== MODEL OLLAMA ====================
OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
LLM_MODEL: Final[str] = "llama3"
EMBEDDING_MODEL: Final[str] = "nomic-embed-text"

# ==================== PARAMETRY LLM ====================
LLM_TEMPERATURE: Final[float] = 0.1
LLM_TOP_P: Final[float] = 0.9
LLM_MAX_TOKENS: Final[int] = 2048

# ==================== PARAMETRY TEXT SPLITTER ====================
CHUNK_SIZE: Final[int] = 700
CHUNK_OVERLAP: Final[int] = 200

# ==================== PARAMETRY RETRIEVERA ====================
RETRIEVER_K: Final[int] = 8  # Optimal: max tested 20
RETRIEVER_SEARCH_TYPE: Final[str] = "mmr"  # Maximum Marginal Relevance - więcej diversity
RETRIEVER_FETCH_K: Final[int] = 16  # Zwiększone dla lepszego MMR diversity

# ==================== KOLEKCJA CHROMADB ====================
CHROMA_COLLECTION_NAME: Final[str] = "local_rag_documents"

# ==================== PROMPT SYSTEMOWY ====================
SYSTEM_PROMPT: Final[str] = """Jesteś asystentem odpowiadającym WYŁĄCZNIE na podstawie dostarczonego kontekstu.

ZASADY:
1) Używaj wyłącznie informacji z kontekstu; jeśli czegoś brakuje, napisz: "Nie znalazłem tej informacji w dostępnych dokumentach.".
2) Tekst odpowiedzi powinien być w tym samym języku co zadane pytanie.
3) Kluczowe liczby, nazwy i terminy cytuj dokładnie jak w tekście (w cudzysłowie, jeśli to dosłowny cytat).
4) Nie parafrazuj liczb ani jednostek; zachowaj oryginalne brzmienie tam, gdzie to ważne dla precyzji.
5) Jeśli kontekst zawiera sprzeczne informacje, wskaż to w odpowiedzi.
6) Odpowiadaj zwięźle, po polsku.
7) Na końcu podaj krótką listę źródeł (nazwy plików z metadanych), jeśli są dostępne.
KONTEKST:
{context}

PYTANIE:
{question}

ODPOWIEDŹ:"""
