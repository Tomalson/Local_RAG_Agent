"""
Interfejs CLI dla systemu Advanced Local RAG.

Prosty interfejs wiersza poleceń z Query Decomposition, Hybrid Search i Context Expansion.
"""

import sys
from pathlib import Path

from colorama import Fore, Style, init

from advanced_rag import AdvancedRAGAgent
import config

# Inicjalizacja kolorowego outputu
init(autoreset=True)


def print_header() -> None:
    """Wyświetla nagłówek aplikacji."""
    print(f"\n{Fore.MAGENTA}{'=' * 70}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}{'ADVANCED LOCAL RAG SYSTEM':^70}")
    print(f"{Fore.MAGENTA}{'Query Decomposition + Hybrid Search + Context Expansion':^70}")
    print(f"{Fore.MAGENTA}{'=' * 70}\n")


def print_instructions() -> None:
    """Wyświetla instrukcje użytkowania."""
    print(f"{Fore.CYAN}Instrukcje:")
    print(f"  • Wpisz pytanie i naciśnij Enter")
    print(f"  • Wpisz {Fore.YELLOW}'exit'{Fore.CYAN}, {Fore.YELLOW}'quit'{Fore.CYAN} lub {Fore.YELLOW}'q'{Fore.CYAN} aby zakończyć")
    print(f"  • Wpisz {Fore.YELLOW}'stats'{Fore.CYAN} aby zobaczyć statystyki bazy")
    print(f"  • Wpisz {Fore.YELLOW}'help'{Fore.CYAN} aby wyświetlić tę pomoc\n")


def print_stats(agent: AdvancedRAGAgent) -> None:
    """
    Wyświetla statystyki bazy dokumentów.

    Args:
        agent: Instancja AdvancedRAGAgent.
    """
    stats = agent.get_stats()
    print(f"\n{Fore.CYAN}{'─' * 70}")
    print(f"{Fore.CYAN}📊 STATYSTYKI BAZY DOKUMENTÓW")
    print(f"{Fore.CYAN}{'─' * 70}")
    print(f"{Fore.WHITE}  • Liczba dokumentów: {Fore.GREEN}{stats['total_documents']}")
    print(f"{Fore.WHITE}  • Nazwa kolekcji: {Fore.GREEN}{stats['collection_name']}")
    print(f"{Fore.WHITE}  • Tryb wyszukiwania: {Fore.GREEN}{stats.get('retrieval_type', 'N/A')}")
    print(f"{Fore.WHITE}  • Model LLM: {Fore.GREEN}{config.LLM_MODEL}")
    print(f"{Fore.WHITE}  • Model Embeddings: {Fore.GREEN}{config.EMBEDDING_MODEL}")
    print(f"{Fore.CYAN}{'─' * 70}\n")


def print_answer(result: dict) -> None:
    """
    Wyświetla odpowiedź wraz z metadanymi.

    Args:
        result: Słownik z odpowiedzią i metadanymi.
    """
    # Debug: pokaż sub-queries
    if "subqueries" in result and result["subqueries"]:
        print(f"\n{Fore.YELLOW}{'─' * 70}")
        print(f"{Fore.YELLOW}🔍 SUB-QUERIES:")
        print(f"{Fore.YELLOW}{'─' * 70}")
        for i, sq in enumerate(result["subqueries"], 1):
            print(f"{Fore.WHITE}  {i}. {sq}")
    
    # Debug: pokaż ile dokumentów użyto
    if "num_docs_used" in result:
        print(f"\n{Fore.CYAN}📚 Documents used: {result['num_docs_used']}")

    # Odpowiedź
    print(f"\n{Fore.GREEN}{'─' * 70}")
    print(f"{Fore.GREEN}{Style.BRIGHT}💡 ODPOWIEDŹ:")
    print(f"{Fore.GREEN}{'─' * 70}")
    print(f"{Fore.WHITE}{result['answer']}\n")
    
    # Źródła
    if result.get('sources'):
        print(f"{Fore.CYAN}{'─' * 70}")
        print(f"{Fore.CYAN}📄 ŹRÓDŁA:")
        print(f"{Fore.CYAN}{'─' * 70}")
        for idx, source in enumerate(result['sources'], 1):
            source_name = Path(source).name
            print(f"{Fore.WHITE}  {idx}. {source_name}")
        print()


def handle_command(command: str, agent: AdvancedRAGAgent) -> bool:
    """
    Obsługuje specjalne komendy użytkownika.

    Args:
        command: Komenda wprowadzona przez użytkownika.
        agent: Instancja AdvancedRAGAgent.

    Returns:
        True jeśli aplikacja powinna kontynuować, False jeśli zakończyć.
    """
    command_lower = command.lower().strip()
    
    if command_lower in ['exit', 'quit', 'q']:
        print(f"\n{Fore.YELLOW}Dziękuję za skorzystanie z Advanced RAG. Do widzenia! 👋\n")
        return False
    
    elif command_lower == 'stats':
        print_stats(agent)
        return True
    
    elif command_lower == 'help':
        print_instructions()
        return True
    
    elif command_lower == 'clear':
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print_header()
        return True
    
    return True


def main() -> None:
    """Główna funkcja uruchamiająca interfejs CLI."""
    print_header()
    
    # Sprawdź czy folder docs istnieje
    if not config.DOCS_DIR.exists() or (not list(config.DOCS_DIR.glob("*.pdf")) and not list(config.DOCS_DIR.glob("*.md"))):
        print(f"{Fore.RED}✗ Brak dokumentów w folderze {config.DOCS_DIR}")
        print(f"{Fore.YELLOW}1. Dodaj pliki PDF lub Markdown do folderu 'docs'")
        print(f"{Fore.YELLOW}2. Uruchom: python ingest.py lub python ingest_md.py")
        print(f"{Fore.YELLOW}3. Uruchom ponownie: python main.py\n")
        sys.exit(1)
    
    # Sprawdź czy baza ChromaDB istnieje
    if not config.CHROMA_DB_DIR.exists():
        print(f"{Fore.RED}✗ Baza ChromaDB nie została utworzona")
        print(f"{Fore.YELLOW}Uruchom najpierw: python ingest.py\n")
        sys.exit(1)
    
    # Inicjalizacja Advanced RAG
    try:
        print(f"{Fore.CYAN}Inicjalizacja Advanced RAG...\n")
        agent = AdvancedRAGAgent()
        print(f"\n{Fore.GREEN}✓ System gotowy do pracy!\n")
        
        # Wyświetl statystyki na start
        print_stats(agent)
        
        # Wyświetl instrukcje
        print_instructions()
        
    except Exception as e:
        print(f"\n{Fore.RED}✗ Błąd inicjalizacji: {e}")
        print(f"{Fore.YELLOW}Sprawdź czy Ollama jest uruchomiona\n")
        sys.exit(1)
    
    # Główna pętla CLI
    print(f"{Fore.MAGENTA}{'=' * 70}\n")
    
    while True:
        try:
            question = input(f"{Fore.YELLOW}{Style.BRIGHT}❓ Twoje pytanie: {Style.RESET_ALL}").strip()
            
            if not question:
                continue
            
            # Obsłuż specjalne komendy
            if question.lower() in ['exit', 'quit', 'q', 'stats', 'help', 'clear']:
                if not handle_command(question, agent):
                    break
                continue
            
            # Zadaj pytanie do Advanced RAG
            print(f"\n{Fore.CYAN}⚙ Przetwarzam pytanie...\n")
            
            result = agent.ask(question)
            print_answer(result)
            
            print(f"{Fore.MAGENTA}{'=' * 70}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Przerwano przez użytkownika. Do widzenia! 👋\n")
            break
            
        except Exception as e:
            print(f"\n{Fore.RED}✗ Błąd: {e}\n")
            print(f"{Fore.MAGENTA}{'=' * 70}\n")
            continue


if __name__ == "__main__":
    main()
