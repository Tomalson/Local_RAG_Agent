#!/usr/bin/env python
"""Test RAG system z nowym retrieverem MMR"""

from rag_service import RAGAgent

try:
    print('Inicjalizacja...')
    agent = RAGAgent()
    
    question = 'Jakie rodzaje przerw przysługują pracownikom pracującym przy monitorze ekranowym w tym kobietom w ciąży i czy można je kumulować?'
    
    print('\n' + '='*80)
    print('POBRANE FRAGMENTY:')
    print('='*80)
    
    docs = agent.retriever.invoke({'question': question})
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        print(f'\n--- Fragment {i} ---')
        print(preview)
    
    print('\n' + '='*80)
    print('ODPOWIEDŹ MODELU:')
    print('='*80)
    
    result = agent.ask(question)
    print(result['answer'])
    
except Exception as e:
    import traceback
    print(f'Błąd: {e}')
    traceback.print_exc()
