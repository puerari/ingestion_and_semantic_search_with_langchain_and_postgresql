# Sistema de Busca Semântica com LangChain e PostgreSQL

Este projeto é a implementação de um desafio técnico do MBA em Engenharia de Software com IA da Faculdade FullCycle.

Consiste em um sistema de RAG (Retrieval-Augmented Generation) utilizando LangChain, PostgreSQL com pgvector para 
armazenamento vetorial, e Google Gemini para embeddings e geração de respostas.

## Funcionalidades

- **Ingestão de Documentos PDF**: Processa e armazena documentos PDF em um banco vetorial
- **Busca Semântica**: Realiza buscas baseadas em significado semântico
- **Chat RAG**: Interface de chat interativa que responde perguntas com base nos documentos ingeridos

## Configuração do Ambiente

Para configurar o ambiente e instalar as dependências do projeto, siga os passos abaixo:

1. **Criar e ativar um ambiente virtual (`venv`):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

2. **Instalar as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar as variáveis de ambiente:**

   - Duplique o arquivo `.env.example` e renomeie para `.env`
   ```bash
   cp .env.example .env
   ```
   - Abra o arquivo `.env` e substitua os valores pelas suas chaves de API reais obtidas conforme instruções abaixo

4. **Instalar PostgreSQL via Docker:**

   ```bash
   docker-compose up -d
   ```

## Requisitos para Execução dos Códigos

Para executar os códigos, é necessário criar chaves de API (API Keys) para os serviços da OpenAI ou do Google Gemini.
Abaixo, fornecemos instruções detalhadas para a criação dessas chaves.

**Para utilizar a OpenAi, comente a chave do Google no .env**

### Criando uma API Key na OpenAI

1. **Acesse o site da OpenAI:**

   [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

2. **Faça login ou crie uma conta:**

   - Se já possuir uma conta, clique em "Log in" e insira suas credenciais.
   - Caso contrário, clique em "Sign up" para criar uma nova conta.

3. **Navegue até a seção de API Keys:**

   - Após o login, clique em "API Keys" no menu lateral esquerdo.

4. **Crie uma nova API Key:**

   - Clique no botão "Create new secret key".
   - Dê um nome para a chave que a identifique facilmente.
   - Clique em "Create secret key".

5. **Copie e armazene sua API Key:**

   - A chave será exibida uma única vez. Copie-a e cole no arquivo `.env` na variável `OPENAI_API_KEY`.

Para mais detalhes, consulte o tutorial completo: [Como Gerar uma API Key na OpenAI?](https://hub.asimov.academy/tutorial/como-gerar-uma-api-key-na-openai/)

### Criando uma API Key no Google Gemini

1. **Acesse o Google AI Studio:**

   [https://ai.google.dev/gemini-api/docs/api-key?hl=pt-BR](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-BR)

2. **Faça login com sua conta Google:**

   - Utilize sua conta Google para acessar o AI Studio.

3. **Navegue até a seção de chaves de API:**

   - No painel de controle, clique em "API Keys" ou "Chaves de API".

4. **Crie uma nova API Key:**

   - Clique em "Create API Key" ou "Criar chave de API".
   - Dê um nome para a chave que a identifique facilmente.
   - A chave será gerada e exibida na tela.

5. **Copie e armazene sua API Key:**

   - Copie a chave gerada e cole no arquivo `.env` na variável `GOOGLE_API_KEY`.

Para mais informações, consulte a documentação oficial: [Como usar chaves da API Gemini](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-BR)

### Arquivo PDF
Este repositório contém o arquivo document.pdf com um relato sobre a história do Linux, o qual pode ser utilizado como
para ingestão de conteúdo no banco vetorial. Caso deseje, você pode substituir este arquivo por um PDF de sua escolha.

---

## Execução

### 1. Ingestão do PDF

   ```bash
   python3 src/ingest.py
   ```

### 2. Iniciar o chat

   ```bash
   python3 src/chat.py
   ```

- Faça perguntas sobre o conteúdo do PDF.

### 3. Encerrar

- Digite **'sair'**, **'exit'** ou **'quit'** para encerrar o script.