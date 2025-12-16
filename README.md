# Conversational-SQL-Assistant

A Streamlit-based conversational AI assistant that supports:
- Direct LLM chat  
- Retrieval-Augmented Generation (RAG) over documents  
- Natural-language querying of SQL databases (MySQL and SQLite)  

Built using LangChain, OpenRouter-hosted LLMs, HuggingFace embeddings, and vector databases (FAISS and Chroma).

---

## Features

### 1. Direct Chat
- Chat with large language models without external context
- Supports multiple OpenRouter-hosted LLMs
- Configurable temperature for response control

### 2. RAG on Documents
- Upload and query PDF, TXT, MD, and CSV files
- Two vector backends:
  - FAISS: in-memory, session-based
  - Chroma: persistent on-disk vector store
- Uses sentence-transformers/all-MiniLM-L6-v2 embeddings
- Context-aware question answering

### 3. Conversational SQL Database Chat
- Ask questions in natural language
- Automatic SQL query generation using LangChain
- Supported databases:
  - MySQL
  - SQLite
- Safety features:
  - Write and DDL queries blocked by default
  - Optional enable for advanced users
  - Automatic LIMIT enforcement
- Displays:
  - Generated SQL
  - Raw query results
  - Natural-language summarized answers

### 4. Persistent Chat History
- Chat history stored locally in chat_history.json
- Persists across application restarts
- Can be cleared from the sidebar

---

## Supported LLMs (via OpenRouter)

Examples include:
- GPT-OSS 20B
- DeepSeek Chat and R1
- Qwen 3 and Qwen 235B
- Meta LLaMA 3.3 and LLaMA 4 Maverick
- Google Gemini 2.0 Flash
- NVIDIA Nemotron Nano
- Sonoma Alpha models

Free and experimental models are supported depending on availability.

---

## Tech Stack

- Frontend: Streamlit  
- LLM orchestration: LangChain  
- LLM provider: OpenRouter (ChatOpenAI wrapper)  
- Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)  
- Vector stores: FAISS, Chroma  
- Databases: MySQL, SQLite  
- Language: Python 3.9 or later  

---

## Project Structure
├── app.py # Main Streamlit application
├── chroma_storage/ # Persistent Chroma vector database
├── uploaded_docs/ # Uploaded documents
├── chat_history.json # Persistent chat history
├── requirements.txt
└── README.md


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/conversational-sql-assistant.git
cd conversational-sql-assistant
```
### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # Linux or macOS
venv\Scripts\activate       # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Environment Variables
Set the following API keys.
Using Streamlit secrets
Create .streamlit/secrets.toml
```toml
OPENROUTER_API_KEY = "your_openrouter_api_key"
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_token"
```
Using environment variables
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
export HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token"
```
### 4. Running the Application
```bash
streamlit run app.py
```


