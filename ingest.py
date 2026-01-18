"""
Skrypt do ingestii dokumentów PDF do bazy wektorowej ChromaDB.

Wczytuje pliki PDF z folderu /docs, dzieli je na fragmenty
i zapisuje embeddingi w /chroma_db.
Professional Local RAG Agent - Initial Release"""

import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from colorama import Fore, Style, init

import config

# Inicjalizacja kolorowego outputu
init(autoreset=True)


class DocumentIngestor:
    """
    Klasa odpowiedzialna za wczytywanie, przetwarzanie i zapisywanie dokumentów PDF.
    """

    def __init__(self) -> None:
        """Inicjalizacja ingestora dokumentów."""
        self.docs_dir: Path = config.DOCS_DIR
        self.chroma_dir: Path = config.CHROMA_DB_DIR
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
            print(f"{Fore.GREEN}✓ Połączono z Ollama Embeddings ({config.EMBEDDING_MODEL})")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd połączenia z Ollama: {e}")
            print(f"{Fore.YELLOW}Upewnij się, że Ollama jest uruchomiona i model {config.EMBEDDING_MODEL} jest pobrany.")
            sys.exit(1)

    def load_pdf_documents(self) -> List:
        """
        Wczytuje wszystkie dokumenty PDF z folderu docs.

        Returns:
            List: Lista załadowanych dokumentów.

        Raises:
            FileNotFoundError: Gdy folder docs jest pusty lub nie zawiera PDF-ów.
        """
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(
                f"Brak plików PDF w folderze {self.docs_dir}. "
                "Dodaj pliki PDF do przetworzenia."
            )

        print(f"{Fore.CYAN}Znaleziono {len(pdf_files)} plików PDF:")
        for pdf in pdf_files:
            print(f"  • {pdf.name}")

        all_documents = []
        
        for pdf_path in pdf_files:
            try:
                # PDFPlumberLoader lepiej radzi sobie z tabelami i layoutem
                loader = PDFPlumberLoader(str(pdf_path))
                documents = loader.load()
                all_documents.extend(documents)
                print(f"{Fore.GREEN}✓ Załadowano: {pdf_path.name} ({len(documents)} stron)")
            except Exception as e:
                print(f"{Fore.RED}✗ Błąd ładowania {pdf_path.name}: {e}")
                continue

        return all_documents

    def split_documents(self, documents: List) -> List:
        """
        Dzieli dokumenty na mniejsze fragmenty.

        Args:
            documents: Lista dokumentów do podzielenia.

        Returns:
            List: Lista podzielonych fragmentów.
        """
        print(f"\n{Fore.CYAN}Dzielenie dokumentów na fragmenty...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"{Fore.GREEN}✓ Utworzono {len(chunks)} fragmentów tekstu")
        return chunks

    def create_vector_store(self, chunks: List) -> None:
        """
        Tworzy bazę wektorową ChromaDB z fragmentów dokumentów.

        Args:
            chunks: Lista fragmentów tekstu do zapisania.
        """
        try:
            print(f"\n{Fore.CYAN}Tworzenie bazy wektorowej ChromaDB...")
            
            # Usuń istniejącą kolekcję jeśli istnieje
            if self.chroma_dir.exists():
                import shutil
                import time
                
                print(f"{Fore.YELLOW}⚠ Próba usunięcia istniejącej bazy danych...")
                
                # Spróbuj 3 razy z opóźnieniem
                for attempt in range(3):
                    try:
                        shutil.rmtree(self.chroma_dir)
                        print(f"{Fore.GREEN}✓ Usunięto istniejącą bazę danych")
                        break
                    except PermissionError as e:
                        if attempt < 2:
                            print(f"{Fore.YELLOW}⚠ Próba {attempt + 1}/3: Baza jest używana, czekam 2 sekundy...")
                            time.sleep(2)
                        else:
                            print(f"{Fore.RED}✗ BŁĄD: Nie można usunąć bazy - jest używana przez inny proces!")
                            print(f"{Fore.YELLOW}💡 ROZWIĄZANIE:")
                            print(f"{Fore.YELLOW}  1. Zamknij wszystkie uruchomione instancje main.py (Ctrl+C)")
                            print(f"{Fore.YELLOW}  2. Zamknij wszystkie terminale z Pythonem")
                            print(f"{Fore.YELLOW}  3. Uruchom ponownie: python ingest.py")
                            raise

            # Twórz nową bazę wektorową z batch processingiem
            print(f"{Fore.CYAN}Tworzenie embeddingów dla {len(chunks)} fragmentów...")
            print(f"{Fore.YELLOW}⏳ To może potrwać kilka minut - proszę czekać...")
            
            # Proces w batch'ach po 10 fragmentów
            batch_size = 10
            vectorstore = None
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                print(f"{Fore.CYAN}  Przetwarzanie batch {batch_num}/{total_batches} ({len(batch)} fragmentów)...", end="", flush=True)
                
                if vectorstore is None:
                    # Pierwszy batch - tworzenie bazy
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        collection_name=config.CHROMA_COLLECTION_NAME,
                        persist_directory=str(self.chroma_dir),
                    )
                else:
                    # Kolejne batche - dodawanie do istniejącej bazy
                    vectorstore.add_documents(batch)
                
                print(f" {Fore.GREEN}✓")
            
            print(f"{Fore.GREEN}✓ Baza wektorowa utworzona pomyślnie!")
            print(f"{Fore.GREEN}✓ Lokalizacja: {self.chroma_dir}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd tworzenia bazy wektorowej: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def run(self) -> None:
        """Wykonuje pełny proces ingestii dokumentów."""
        print(f"\n{Fore.MAGENTA}{'=' * 60}")
        print(f"{Fore.MAGENTA}{'INGESTIA DOKUMENTÓW PDF DO RAG':^60}")
        print(f"{Fore.MAGENTA}{'=' * 60}\n")

        try:
            # 1. Wczytaj dokumenty PDF
            documents = self.load_pdf_documents()

            # 2. Podziel na fragmenty
            chunks = self.split_documents(documents)

            # 3. Utwórz bazę wektorową
            self.create_vector_store(chunks)

            print(f"\n{Fore.GREEN}{'=' * 60}")
            print(f"{Fore.GREEN}{'✓ INGESTIA ZAKOŃCZONA POMYŚLNIE':^60}")
            print(f"{Fore.GREEN}{'=' * 60}\n")

        except FileNotFoundError as e:
            print(f"\n{Fore.RED}✗ BŁĄD: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n{Fore.RED}✗ NIEOCZEKIWANY BŁĄD: {e}")
            sys.exit(1)


def main() -> None:
    """Główna funkcja uruchamiająca proces ingestii."""
    ingestor = DocumentIngestor()
    ingestor.run()


if __name__ == "__main__":
    main()
