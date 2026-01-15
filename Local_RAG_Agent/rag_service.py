"""
Serwis RAG (Retrieval-Augmented Generation).

Zawiera klasę RAGAgent odpowiedzialną za pobieranie kontekstu
i generowanie odpowiedzi przy użyciu lokalnego LLM.
"""

import sys
from typing import Optional, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from colorama import Fore, Style, init

import config

# Inicjalizacja kolorowego outputu
init(autoreset=True)


class RAGAgent:
    """
    Agent RAG odpowiedzialny za pobieranie kontekstu z bazy wektorowej
    i generowanie odpowiedzi opartych wyłącznie na dostarczonych dokumentach.
    """

    def __init__(self) -> None:
        """
        Inicjalizuje agenta RAG z połączeniem do ChromaDB i Ollama.

        Raises:
            ConnectionError: Gdy nie można połączyć się z Ollama.
            FileNotFoundError: Gdy baza ChromaDB nie istnieje.
        """
        self._initialize_embeddings()
        self._initialize_vectorstore()
        self._initialize_llm()
        self._initialize_qa_chain()

    def _initialize_embeddings(self) -> None:
        """Inicjalizuje model embeddingów Ollama."""
        try:
            self.embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
            print(f"{Fore.GREEN}✓ Embeddings zainicjalizowane ({config.EMBEDDING_MODEL})")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd inicjalizacji embeddingów: {e}")
            raise ConnectionError(
                f"Nie można połączyć się z Ollama. Upewnij się, że działa na {config.OLLAMA_BASE_URL}"
            )

    def _initialize_vectorstore(self) -> None:
        """Inicjalizuje połączenie z bazą wektorową ChromaDB."""
        if not config.CHROMA_DB_DIR.exists():
            print(f"{Fore.RED}✗ Baza ChromaDB nie istnieje: {config.CHROMA_DB_DIR}")
            print(f"{Fore.YELLOW}Uruchom najpierw: python ingest.py")
            raise FileNotFoundError(
                f"Baza danych nie znaleziona w {config.CHROMA_DB_DIR}. "
                "Uruchom najpierw skrypt ingest.py aby przetworzyć dokumenty."
            )

        try:
            self.vectorstore = Chroma(
                persist_directory=str(config.CHROMA_DB_DIR),
                embedding_function=self.embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
            
            # Sprawdź czy baza zawiera dokumenty
            collection = self.vectorstore._collection
            count = collection.count()
            
            if count == 0:
                raise ValueError("Baza wektorowa jest pusta. Uruchom ponownie ingest.py")
            
            print(f"{Fore.GREEN}✓ ChromaDB połączone ({count} dokumentów)")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd połączenia z ChromaDB: {e}")
            raise

    def _initialize_llm(self) -> None:
        """Inicjalizuje lokalny model LLM przez Ollama."""
        try:
            self.llm = Ollama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                num_predict=config.LLM_MAX_TOKENS,
            )
            print(f"{Fore.GREEN}✓ LLM zainicjalizowany ({config.LLM_MODEL})")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd inicjalizacji LLM: {e}")
            raise ConnectionError(
                f"Nie można załadować modelu {config.LLM_MODEL}. "
                f"Upewnij się, że model jest pobrany: ollama pull {config.LLM_MODEL}"
            )

    def _initialize_qa_chain(self) -> None:
        """Inicjalizuje łańcuch pytań-odpowiedzi z rygorystycznym promptem."""
        # Tworzenie retrievera z parametrami z config
        search_kwargs = {"k": config.RETRIEVER_K}
        if config.RETRIEVER_SEARCH_TYPE == "mmr":
            search_kwargs["fetch_k"] = config.RETRIEVER_FETCH_K
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=config.RETRIEVER_SEARCH_TYPE,
            search_kwargs=search_kwargs
        )

        # Tworzenie promptu z rygorystycznymi zasadami
        self.prompt_template = PromptTemplate(
            template=config.SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )

        # Prosty chain: {context} -> retriever, {question} ->Input
        # chain = prompt | llm | output_parser
        self.qa_chain = (
            RunnablePassthrough.assign(context=self.retriever)
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        print(f"{Fore.GREEN}✓ RAG Chain zainicjalizowany")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Zadaje pytanie i zwraca odpowiedź wraz z dokumentami źródłowymi.

        Args:
            question: Pytanie użytkownika.

        Returns:
            Dict zawierający:
                - 'answer': Odpowiedź wygenerowana przez model
                - 'source_documents': Lista dokumentów źródłowych
                - 'sources': Lista nazw plików źródłowych

        Raises:
            ValueError: Gdy pytanie jest puste.
            RuntimeError: Gdy wystąpi błąd podczas generowania odpowiedzi.
        """
        if not question or not question.strip():
            raise ValueError("Pytanie nie może być puste")

        try:
            # Wykonaj zapytanie - retrieve context manualnie
            docs = self.retriever.invoke({"question": question})
            
            # Przygotuj context z dokumentów
            context_str = "\n---\n".join([doc.page_content for doc in docs])
            
            # Użyj chain'a do wygenerowania odpowiedzi (LLM z promptem z config)
            answer = self.qa_chain.invoke({"context": context_str, "question": question})
            
            # Wyciągnij unikalne źródła
            sources = []
            seen_sources = set()
            for doc in docs:
                source = doc.metadata.get("source", "Nieznane źródło")
                if source not in seen_sources:
                    sources.append(source)
                    seen_sources.add(source)

            return {
                "answer": answer,
                "source_documents": docs,
                "sources": sources
            }

        except Exception as e:
            print(f"{Fore.RED}✗ Błąd podczas generowania odpowiedzi: {e}")
            raise RuntimeError(f"Nie udało się wygenerować odpowiedzi: {e}")

    def get_stats(self) -> Dict[str, int]:
        """
        Zwraca statystyki bazy wektorowej.

        Returns:
            Dict ze statystykami (liczba dokumentów, etc.)
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "total_documents": count,
                "collection_name": config.CHROMA_COLLECTION_NAME
            }
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd pobierania statystyk: {e}")
            return {"total_documents": 0, "collection_name": "unknown"}


def main() -> None:
    """Funkcja testowa - przykładowe użycie RAGAgent."""
    print(f"\n{Fore.MAGENTA}{'=' * 60}")
    print(f"{Fore.MAGENTA}{'TEST RAG AGENT':^60}")
    print(f"{Fore.MAGENTA}{'=' * 60}\n")

    try:
        # Inicjalizacja agenta
        agent = RAGAgent()
        
        # Statystyki
        stats = agent.get_stats()
        print(f"\n{Fore.CYAN}Statystyki bazy:")
        print(f"  • Dokumentów: {stats['total_documents']}")
        print(f"  • Kolekcja: {stats['collection_name']}\n")

        # Przykładowe pytanie
        test_question = "O czym jest ten dokument?"
        print(f"{Fore.YELLOW}Pytanie: {test_question}")
        
        result = agent.ask(test_question)
        
        print(f"\n{Fore.GREEN}Odpowiedź:")
        print(f"{Style.BRIGHT}{result['answer']}\n")
        
        if result['sources']:
            print(f"{Fore.CYAN}Źródła:")
            for source in result['sources']:
                print(f"  • {source}")

    except Exception as e:
        print(f"\n{Fore.RED}✗ BŁĄD: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
