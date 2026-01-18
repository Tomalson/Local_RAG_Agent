"""
Advanced RAG System z Query Decomposition, Hybrid Search, i Context Awareness.
Professional Local RAG Agent - Initial Release"""

import sys
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from colorama import Fore, Style, init
from rank_bm25 import BM25Okapi

import config

init(autoreset=True)


class HybridRetriever:
    """
    Custom Hybrid Retriever łączący Vector Search i BM25.
    
    Kombinuje siłę wyszukiwania semantycznego (wektory) z wyszukiwaniem 
    słów kluczowych (BM25) z równymi wagami [0.5, 0.5].
    """
    
    def __init__(self, vector_retriever, bm25_retriever, weights=None):
        """
        Args:
            vector_retriever: Vector search retriever (Chroma MMR)
            bm25_retriever: BM25 keyword search retriever
            weights: Wagi [vector_weight, bm25_weight], domyślnie [0.5, 0.5]
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights or [0.5, 0.5]
        
        # Normalizuj wagi
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def invoke(self, query: str, k: int = 8):
        """
        Hybrid Search: zwraca topowe dokumenty łącząc oba retrievers.
        
        Args:
            query: Zapytanie
            k: Liczba dokumentów do zwrócenia
            
        Returns:
            Lista dokumentów posortowanych wg hybrid score
        """
        # Vector search
        try:
            vector_docs = self.vector_retriever.invoke(query)
            vector_dict = {doc.page_content: doc for doc in vector_docs}
        except:
            vector_dict = {}
        
        # BM25 search
        try:
            bm25_docs = self.bm25_retriever.invoke(query)
            bm25_dict = {doc.page_content: doc for doc in bm25_docs}
        except:
            bm25_dict = {}
        
        # Merge z wagami
        scores = {}
        
        # Vector scores - rank based (1.0 dla pierwszego, maleje)
        for idx, (content, doc) in enumerate(vector_dict.items()):
            score = (len(vector_dict) - idx) / len(vector_dict) if vector_dict else 0
            scores[content] = scores.get(content, 0) + self.weights[0] * score
        
        # BM25 scores
        for idx, (content, doc) in enumerate(bm25_dict.items()):
            score = (len(bm25_dict) - idx) / len(bm25_dict) if bm25_dict else 0
            scores[content] = scores.get(content, 0) + self.weights[1] * score
        
        # Sort i zwróć top k
        sorted_contents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Zwróć dokumenty w oryginalnej formie
        results = []
        for content, _ in sorted_contents:
            if content in vector_dict:
                results.append(vector_dict[content])
            elif content in bm25_dict:
                results.append(bm25_dict[content])
        
        return results


init(autoreset=True)


class AdvancedRAGAgent:
    """
    Advanced RAG z Query Decomposition, Hybrid Search (BM25 + Vector), 
    i Context Expansion.
    """

    def __init__(self) -> None:
        """Inicjalizuje advanced RAG agent."""
        self._initialize_embeddings()
        self._initialize_vectorstore()
        self._initialize_llm()
        self._initialize_bm25_index()
        self._initialize_qa_chain()

    def _initialize_embeddings(self) -> None:
        """Inicjalizuje embeddingi."""
        try:
            self.embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
            print(f"{Fore.GREEN}✓ Embeddings zainicjalizowane ({config.EMBEDDING_MODEL})")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd embeddings: {e}")
            raise

    def _initialize_vectorstore(self) -> None:
        """Inicjalizuje ChromaDB vectorstore."""
        if not config.CHROMA_DB_DIR.exists():
            print(f"{Fore.RED}✗ Baza ChromaDB nie istnieje: {config.CHROMA_DB_DIR}")
            raise FileNotFoundError("Uruchom najpierw: python ingest.py")

        try:
            self.vectorstore = Chroma(
                persist_directory=str(config.CHROMA_DB_DIR),
                embedding_function=self.embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
            collection = self.vectorstore._collection
            count = collection.count()
            if count == 0:
                raise ValueError("Baza wektorowa jest pusta.")
            
            # Pobierz wszystkie dokumenty dla BM25 indexu
            self.all_documents = collection.get(include=["documents", "metadatas"])
            print(f"{Fore.GREEN}✓ ChromaDB połączone ({count} dokumentów)")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd ChromaDB: {e}")
            raise

    def _initialize_bm25_index(self) -> None:
        """Inicjalizuje BM25 index dla keyword search."""
        try:
            texts = self.all_documents["documents"]
            # Tokenizacja: split na słowa
            tokenized_docs = [doc.lower().split() for doc in texts]
            self.bm25 = BM25Okapi(tokenized_docs)
            self.bm25_docs = texts
            print(f"{Fore.GREEN}✓ BM25 index zbudowany ({len(texts)} dokumentów)")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd BM25: {e}")
            raise

    def _initialize_llm(self) -> None:
        """Inicjalizuje LLM."""
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
            print(f"{Fore.RED}✗ Błąd LLM: {e}")
            raise

    def _initialize_qa_chain(self) -> None:
        """Inicjalizuje QA chain z Hybrid Search (Vector + BM25)."""
        # Vector Retriever
        vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_FETCH_K}
        )
        print(f"{Fore.GREEN}✓ Vector Retriever zainicjalizowany (MMR)")

        # BM25 Retriever
        try:
            documents = [
                Document(
                    page_content=text,
                    metadata=meta if meta else {}
                )
                for text, meta in zip(
                    self.all_documents["documents"],
                    self.all_documents["metadatas"]
                )
            ]
            bm25_retriever = BM25Retriever.from_documents(
                documents=documents,
                k=config.RETRIEVER_K
            )
            print(f"{Fore.GREEN}✓ BM25 Retriever zainicjalizowany (Keyword Search)")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd BM25 Retriever: {e}")
            raise

        # Hybrid Retriever - łączy Vector + BM25 z wagami [0.5, 0.5]
        try:
            self.retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                weights=[0.5, 0.5]  # Równoważyć semantic search i keyword search
            )
            print(f"{Fore.GREEN}✓ Hybrid Retriever zainicjalizowany (Vector 0.5 + BM25 0.5)")
        except Exception as e:
            print(f"{Fore.RED}✗ Błąd Hybrid Retriever: {e}")
            raise

        self.prompt_template = PromptTemplate(
            template=config.SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )

        self.qa_chain = (
            {
                "context": lambda x: self.retriever.invoke(x),
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        print(f"{Fore.GREEN}✓ QA Chain (Hybrid Search) zainicjalizowany")

    def decompose_query(self, question: str) -> List[str]:
        """
        Rozbija złożone pytanie na prostsze sub-pytania.
        
        Args:
            question: Pytanie użytkownika
            
        Returns:
            Lista sub-pytań
        """
        decompose_prompt = f"""Jesteś asystentem specjalizującym się w rozbiciu złożonych pytań na prostsze.

Pytanie użytkownika: "{question}"

Rozbij to pytanie na 2-4 prostsze, konkretne sub-pytania, które razem odpowiadają na oryginalne pytanie.
Każde sub-pytanie powinno być niezależne i możliwe do udzielenia na podstawie tekstu.

Zwróć odpowiedź jako JSON array:
{{"subqueries": ["sub-pytanie 1", "sub-pytanie 2", ...]}}

SUBQUERIES:"""

        try:
            response = self.llm.invoke(decompose_prompt)
            # Parsuj JSON
            try:
                data = json.loads(response)
                subqueries = data.get("subqueries", [question])
            except:
                # Fallback: zwróć oryginalne pytanie
                subqueries = [question]
            
            return subqueries if subqueries else [question]
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Decomposition failed, using original query: {e}")
            return [question]

    def hybrid_search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """
        Hybrid Search: BM25 (keywords) + Vector (semantic).
        
        Args:
            query: Zapytanie
            k: Liczba dokumentów do zwrócenia
            
        Returns:
            Lista dokumentów z score'ami
        """
        # Vector search
        vector_results = self.retriever.invoke({"question": query})
        vector_docs = {doc.page_content: {"doc": doc, "score": 1.0} for doc in vector_results}

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        bm25_results = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                bm25_results.append((self.bm25_docs[idx], score))
        
        # Normalizuj BM25 scores
        if bm25_results:
            max_score = max(s for _, s in bm25_results)
            bm25_results = [(doc, s / max_score) for doc, s in bm25_results]

        # Merge: vector + BM25 (average score)
        merged = {}
        for doc in vector_docs:
            merged[doc] = {"score": vector_docs[doc]["score"], "doc": vector_docs[doc]["doc"]}
        
        for doc, bm25_score in bm25_results:
            if doc in merged:
                merged[doc]["score"] = (merged[doc]["score"] + bm25_score) / 2
            else:
                # Znajdź metadata
                try:
                    idx = self.bm25_docs.index(doc)
                    meta = self.all_documents["metadatas"][idx] if idx < len(self.all_documents["metadatas"]) else {}
                    from langchain_core.documents import Document
                    merged[doc] = {"score": bm25_score, "doc": Document(page_content=doc, metadata=meta)}
                except:
                    pass

        # Sort i return top k
        sorted_results = sorted(merged.items(), key=lambda x: x[1]["score"], reverse=True)[:k]
        return [item[1]["doc"] for item in sorted_results]

    def expand_context(self, doc_content: str, k: int = 1) -> str:
        """
        Rozszerza kontekst dokumentu o sąsiadujące chunki.
        
        Args:
            doc_content: Zawartość chunka
            k: Liczba sąsiadów do dodania (przed i po)
            
        Returns:
            Rozszerzony kontekst
        """
        try:
            idx = self.bm25_docs.index(doc_content)
            start = max(0, idx - k)
            end = min(len(self.bm25_docs), idx + k + 1)
            
            expanded = "\n[...]\n".join(self.bm25_docs[start:end])
            return expanded
        except ValueError:
            # Nie znaleziono, zwróć oryginał
            return doc_content

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Advanced ask z decomposition i Hybrid Search (EnsembleRetriever).
        
        Hybrid Search łączy:
        - Vector Search (semantyczne rozumienie)
        - BM25 Search (matching słów kluczowych)
        
        Args:
            question: Pytanie użytkownika
            
        Returns:
            Dict z odpowiedzią i źródłami
        """
        if not question or not question.strip():
            raise ValueError("Pytanie nie może być puste")

        try:
            print(f"\n{Fore.CYAN}🔍 Decomposing query...")
            subqueries = self.decompose_query(question)
            print(f"{Fore.CYAN}Found {len(subqueries)} sub-queries:")
            for i, sq in enumerate(subqueries, 1):
                print(f"  {i}. {sq}")

            # Hybrid search dla każdego sub-query (używa EnsembleRetriever)
            all_docs = []
            doc_contents_seen = set()

            for subq in subqueries:
                print(f"\n{Fore.CYAN}  Searching for: {subq} (Hybrid: Vector + BM25)")
                # EnsembleRetriever łączy wektory i BM25 z wagami [0.5, 0.5]
                docs = self.retriever.invoke(subq)
                for doc in docs:
                    # Avoid duplicates
                    if doc.page_content not in doc_contents_seen:
                        all_docs.append(doc)
                        doc_contents_seen.add(doc.page_content)

            # Merge contexts
            context_str = "\n---\n".join([doc.page_content for doc in all_docs])

            print(f"\n{Fore.CYAN}📚 Using {len(all_docs)} documents (Hybrid Search result)")

            # LLM answer
            answer = self.qa_chain.invoke({"context": context_str, "question": question})

            # Unique sources
            sources = []
            seen_sources = set()
            for doc in all_docs:
                source = doc.metadata.get("source", "Nieznane źródło")
                if source not in seen_sources:
                    sources.append(source)
                    seen_sources.add(source)

            return {
                "answer": answer,
                "source_documents": all_docs,
                "sources": sources,
                "subqueries": subqueries,
                "num_docs_used": len(all_docs)
            }

        except Exception as e:
            print(f"{Fore.RED}✗ Error: {e}")
            raise

    def get_stats(self) -> Dict[str, int]:
        """Zwraca statystyki bazy."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "total_documents": count,
                "collection_name": config.CHROMA_COLLECTION_NAME,
                "retrieval_type": "Hybrid (BM25 + Vector) + Decomposition"
            }
        except Exception as e:
            print(f"{Fore.RED}✗ Error: {e}")
            return {"total_documents": 0}
