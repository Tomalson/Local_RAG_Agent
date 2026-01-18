"""
Skrypt do szybkiego załadowania plików Markdown do ChromaDB.
"""

import sys
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

print("\n[+] Ingestion Markdown dokumentow...")

# Loader dla .md files
loader = DirectoryLoader(
    str(config.DOCS_DIR),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

docs = loader.load()
if not docs:
    print(f"[X] Brak plikow .md w {config.DOCS_DIR}")
    sys.exit(1)

print(f"[OK] Zaladowano {len(docs)} dokumentow")

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)
chunks = splitter.split_documents(docs)
print(f"[OK] Podzielono na {len(chunks)} fragmentow")

# Embeddings
embeddings = OllamaEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.OLLAMA_BASE_URL
)
print(f"[OK] Polaczono z Ollama Embeddings ({config.EMBEDDING_MODEL})")

# ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=config.CHROMA_COLLECTION_NAME,
    persist_directory=str(config.CHROMA_DB_DIR)
)
print(f"[OK] Zapisano do ChromaDB ({config.CHROMA_DB_DIR})")
print(f"[SUCCESS] Sukces! Zaindeksowano {len(chunks)} fragmentow\n")
