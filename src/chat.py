#!/usr/bin/env python3
import os
import sys
from typing import List, Tuple
from dotenv import load_dotenv
from search import semantic_search
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Verificar variáveis de ambiente necessárias
for var in ["PGVECTOR_URL", "PGVECTOR_COLLECTION"]:
    if not os.getenv(var):
        raise RuntimeError(f"Environment variable {var} is not set")

# Verificar se pelo menos um provedor de LLM está configurado
has_google = bool(os.getenv("GOOGLE_API_KEY"))
has_openai = bool(os.getenv("OPENAI_API_KEY"))

if not has_google and not has_openai:
    raise RuntimeError("At least one LLM provider should be configured: GOOGLE_API_KEY or OPENAI_API_KEY")

class ChatRAG:
    def __init__(self):
        if has_google:
            self.provider_name = "Gemini"
            if not os.getenv("GOOGLE_API_KEY"):
                raise RuntimeError("GOOGLE_API_KEY is not configured!")

            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
                temperature=0.1,
                max_tokens=1000
            )
        else:
            self.provider_name = "OpenAI"
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not configured!")

            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                temperature=0.1,
                max_tokens=1000
            )

        # Template de prompt
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Você é um assistente especializado em responder perguntas com base no contexto fornecido.

Contexto relevante:
{context}

Pergunta do usuário:
{question}

Com base apenas no contexto fornecido, responda à pergunta de forma clara e objetiva. 
Se a informação não estiver no contexto, informe que não consegue responder com base nos documentos disponíveis.

Resposta:"""
        )
        
        # Criar chain
        self.chain = self.prompt_template | self.llm
    
    def search_documents(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Busca documentos relevantes no banco vetorial.
        Args:
            query: Pergunta do usuário
            k: Número de resultados a retornar
        Returns:
            Lista de tuplas (documento, score)
        """
        try:
            return semantic_search(query, k)
        except Exception as e:
            print(f"Erro na busca vetorial: {e}")
            return []
    
    def format_context(self, search_results: List[Tuple[Document, float]]) -> str:
        """
        Formata os resultados da busca em um contexto para o LLM.
        Args:
            search_results: Resultados da busca vetorial
        Returns:
            Contexto formatado
        """
        if not search_results:
            return "Nenhum documento relevante encontrado."
        
        context_parts = []
        for i, (doc, score) in enumerate(search_results, start=1):
            context_parts.append(
                f"Documento {i} (distância: {score:.3f}):\n{doc.page_content.strip()}"
            )
        
        return "\n\n".join(context_parts)
    
    def answer_question(self, question: str) -> str:
        """
        Processa uma pergunta e retorna a resposta.
        Args:
            question: Pergunta do usuário
        Returns:
            Resposta gerada pela LLM
        """
        print("🔍 Buscando documentos relevantes...")
        search_results = self.search_documents(question, k=10)
        
        if not search_results:
            return "❌ Não foi possível encontrar documentos relevantes para sua pergunta."
        
        print(f"📚 Encontrados {len(search_results)} documentos relevantes")
        
        # Formatar contexto
        context = self.format_context(search_results)
        
        print(f"🤖 Gerando resposta com {self.provider_name}...")
        try:
            # Gerar resposta
            response = self.chain.invoke({"context": context, "question": question})
            return response.content.strip()
        except Exception as e:
            return f"❌ Erro ao gerar resposta: {e}"
    
    def run_chat(self):
        print("=" * 60)
        print(f"🤖 Chat RAG com {self.provider_name} e PostgreSQL")
        print("Digite 'exit' para encerrar")
        print("=" * 60)
        while True:
            try:
                # Obter entrada do usuário
                question = input("\n❓ Faça sua pergunta (exit para sair): ").strip()
                
                # Verificar comando de saída
                if question.lower() == 'exit':
                    print("\n👋 Até logo!")
                    break
                
                if not question:
                    print("⚠️ Por favor, digite uma pergunta válida.")
                    continue
                
                # Processar pergunta
                print("\n" + "-" * 40)
                answer = self.answer_question(question)
                print(f"\n💬 Resposta:\n{answer}")
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat encerrado pelo usuário.")
                break
            except Exception as e:
                print(f"\n❌ Ocorreu um erro: {e}")

def main():
    import argparse
    
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Chat RAG com busca vetorial")
    args = parser.parse_args()
    
    try:
        chat = ChatRAG()
        chat.run_chat()
    except Exception as e:
        print(f"❌ Erro ao inicializar o chat: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()