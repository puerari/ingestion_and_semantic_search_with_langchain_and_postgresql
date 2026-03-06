import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

load_dotenv()
for k in ("PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

def semantic_search(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """
    Realiza busca semântica no banco vetorial PostgreSQL.
    
    Args:
        query: Texto da consulta para busca
        k: Número de resultados a retornar (padrão: 10)
        
    Returns:
        Lista de tuplas (documento, score) ordenados por relevância
    """
    # Configurar embeddings
    if os.getenv("GOOGLE_API_KEY"):
        embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        )
    elif os.getenv("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    else:
        raise RuntimeError(f"Set 'GOOGLE_API_KEY' or 'OPENAI_API_KEY' environment variable")

    # Configurar conexão com banco vetorial
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )
    
    # Realizar busca semântica
    results = store.similarity_search_with_score(query, k=k)
    
    return results

def print_search_results(query: str, k: int = 10):
    """
    Função para testar busca e imprimir resultados no terminal.
    
    Args:
        query: Texto da consulta
        k: Número de resultados a imprimir
    """
    results = semantic_search(query, k)
    
    for i, (doc, score) in enumerate(results, start=1):
        print("=" * 50)
        print(f"Resultado {i} (score: {score:.2f}):")
        print("=" * 50)

        print("\nTexto:\n")
        print(doc.page_content.strip())

        print("\nMetadados:\n")
        for k, v in doc.metadata.items():
            print(f"{k}: {v}")



if __name__ == "__main__":
    print_search_results("test query")
