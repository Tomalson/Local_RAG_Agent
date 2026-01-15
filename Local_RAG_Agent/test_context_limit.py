"""
Test maksymalnego rozmiaru kontekstu dla llama3.
Stopniowo zwiększa liczbę dokumentów aż do błędu.
"""

import sys
from pathlib import Path
from colorama import Fore, Style, init
import config

# Importuj AdvancedRAGAgent
from advanced_rag import AdvancedRAGAgent

init(autoreset=True)

def test_with_k_docs(k_value: int, test_question: str) -> bool:
    """
    Testuje z daną liczbą dokumentów.
    
    Args:
        k_value: Liczba dokumentów do pobrania
        test_question: Pytanie testowe
        
    Returns:
        True jeśli udało się, False jeśli błąd
    """
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}Testing with k={k_value} documents...")
    print(f"{Fore.CYAN}{'='*70}")
    
    # Tymczasowo zmień config
    original_k = config.RETRIEVER_K
    config.RETRIEVER_K = k_value
    
    try:
        # Reinicjalizuj agenta z nowymi parametrami
        agent = AdvancedRAGAgent()
        
        # Nadpisz hybrid_search k
        docs = agent.hybrid_search(test_question, k=k_value)
        
        # Oblicz rozmiar kontekstu
        context_str = "\n---\n".join([doc.page_content for doc in docs])
        context_size = len(context_str)
        context_tokens = context_size // 4  # ~4 chars per token
        
        print(f"{Fore.YELLOW}📊 Context stats:")
        print(f"  • Documents: {len(docs)}")
        print(f"  • Characters: {context_size:,}")
        print(f"  • Estimated tokens: ~{context_tokens:,}")
        
        # Spróbuj zapytać
        print(f"{Fore.CYAN}🤖 Asking LLM...")
        result = agent.ask(test_question)
        
        answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
        print(f"{Fore.GREEN}✓ SUCCESS!")
        print(f"{Fore.GREEN}  Answer preview: {answer_preview}")
        
        # Przywróć config
        config.RETRIEVER_K = original_k
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "context length" in error_msg.lower():
            print(f"{Fore.RED}✗ CONTEXT LENGTH ERROR!")
            print(f"{Fore.RED}  Error: {error_msg}")
        else:
            print(f"{Fore.RED}✗ OTHER ERROR: {error_msg}")
        
        # Przywróć config
        config.RETRIEVER_K = original_k
        return False


def find_max_k():
    """Znajduje maksymalne k poprzez binary search."""
    
    TEST_QUESTION = "Jakie rodzaje przerw przysługują pracownikom pracującym przy monitorze ekranowym (w tym kobietom w ciąży) i czy można je kumulować, aby wyjść wcześniej z pracy?"
    
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{'CONTEXT LIMIT FINDER':^70}")
    print(f"{Fore.MAGENTA}{'='*70}\n")
    
    print(f"{Fore.CYAN}Test question: {TEST_QUESTION}\n")
    
    # Testuj od małych do dużych wartości
    k_values = [3, 5, 7, 10, 12, 15, 20]
    
    max_working_k = None
    
    for k in k_values:
        success = test_with_k_docs(k, TEST_QUESTION)
        
        if success:
            max_working_k = k
            print(f"{Fore.GREEN}✓ k={k} works!")
        else:
            print(f"{Fore.RED}✗ k={k} FAILED - stopping here")
            break
    
    # Summary
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{'RESULTS':^70}")
    print(f"{Fore.MAGENTA}{'='*70}\n")
    
    if max_working_k:
        print(f"{Fore.GREEN}✓ Maximum working k: {max_working_k}")
        print(f"{Fore.YELLOW}📝 Recommended safe k: {max_working_k - 1} (with safety margin)")
        
        print(f"\n{Fore.CYAN}Update config.py:")
        print(f"{Fore.WHITE}  RETRIEVER_K: Final[int] = {max_working_k - 1}")
        print(f"\n{Fore.CYAN}Update advanced_rag.py (hybrid_search call):")
        print(f"{Fore.WHITE}  docs = self.hybrid_search(subq, k={max_working_k - 1})")
    else:
        print(f"{Fore.RED}✗ All tests failed!")


if __name__ == "__main__":
    try:
        find_max_k()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user")
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}")
        import traceback
        traceback.print_exc()
