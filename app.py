
import os
import io
import json
import re
import tempfile
from datetime import datetime

import streamlit as st

# LLM
from langchain_openai import ChatOpenAI

# Prompts / Chains
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import create_sql_query_chain  # NEW

# SQL Database (NEW import path for modern LangChain)
from langchain_community.utilities import SQLDatabase

# Docs / Splitting
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector stores & embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_community.document_loaders import CSVLoader
    HAVE_CSV = True
except Exception:
    HAVE_CSV = False


# ======================
# CONFIG / KEYS
# ======================
st.set_page_config(page_title="Conversational SQL Assisstant MAIN", layout="wide")

# Use Streamlit secrets for API keys
# Use Streamlit secrets for API keys
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
HF_API_KEY = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
if HF_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

MODELS = [
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3-coder:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-chat-v3.1:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-4-maverick:free",
    "google/gemini-2.0-flash-exp:free",
    "openrouter/sonoma-sky-alpha",
    "openrouter/sonoma-dusk-alpha"
]

# Persistent paths
CHROMA_DB_DIR = "chroma_storage"
UPLOAD_DIR = "uploaded_docs"
CHAT_HISTORY_FILE = "chat_history.json"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ======================
# Chat History (persistent)
# ======================
def load_chat_history() -> list:
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(history: list):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if "history" not in st.session_state:
    st.session_state.history = load_chat_history()

def add_to_history(role: str, content: str, meta: dict = None):
    item = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": content,
        "meta": meta or {}
    }
    st.session_state.history.append(item)
    save_chat_history(st.session_state.history)

def clear_history():
    st.session_state.history = []
    save_chat_history([])


# ======================
# LLM helper
# ======================
def get_chat_model(model_name: str, temperature: float = 0.7):
    if not os.getenv("OPENROUTER_API_KEY"):
        st.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY in Streamlit secrets or environment.")
    return ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        temperature=temperature,
    )


# ======================
# RAG Helpers
# ======================
def split_text_docs(text: str, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def load_and_split_file_to_docs(saved_path: str):
    ext = os.path.splitext(saved_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(saved_path)
        docs = loader.load()
    elif ext in [".txt", ".md"]:
        loader = TextLoader(saved_path, encoding="utf-8")
        docs = loader.load()
    elif ext == ".csv" and HAVE_CSV:
        loader = CSVLoader(saved_path, encoding="utf-8")
        docs = loader.load()
    else:
        try:
            with io.open(saved_path, "r", encoding="utf-8", errors="ignore") as f:
                return split_text_docs(f.read())
        except Exception:
            return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

def build_faiss_retriever(files):
    docs = []
    for file in files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        docs.extend(load_and_split_file_to_docs(path))
    if not docs:
        return None
    store = FAISS.from_documents(docs, embeddings)
    return store.as_retriever(search_kwargs={"k": 3})

def get_chroma_vectorstore() -> Chroma:
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def add_files_to_chroma(files):
    vs = get_chroma_vectorstore()
    for file in files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        split_docs = load_and_split_file_to_docs(path)
        if split_docs:
            vs.add_documents(split_docs)
    vs.persist()
    return vs


# ======================
# SQL Helpers (modern chain)
# ======================
WRITE_PAT = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE|MERGE)\b",
    re.I,
)

def is_write_query(sql_text: str) -> bool:
    return bool(WRITE_PAT.search(sql_text or ""))

def list_tables_info(db: SQLDatabase) -> str:
    try:
        tables = db.get_usable_table_names()
        preview = []
        for t in sorted(list(tables))[:20]:
            preview.append(f"- {t}")
        return "\n".join(preview) if preview else "No tables found."
    except Exception as e:
        return f"Could not list tables: {e}"

def answer_from_sql(llm: ChatOpenAI, question: str, sql: str, rows) -> str:
    """Turn raw SQL results into a friendly answer."""
    # Coerce rows to displayable text
    rows_str = ""
    try:
        if isinstance(rows, list):
            rows_str = "\n".join([str(r) for r in rows[:50]])  # limit
        else:
            rows_str = str(rows)
    except Exception:
        rows_str = str(rows)

    prompt_tmpl = PromptTemplate.from_template(
        """You are a helpful data analyst.

Question:
{question}

The SQL query you ran:
{sql}

Raw results:
{rows}

Produce a concise, human-friendly answer. If results are empty, say so. If appropriate, summarize and include key numbers."""
    )
    prompt = prompt_tmpl.format(question=question, sql=sql, rows=rows_str)
    return llm.invoke(prompt).content


# ======================
# UI
# ======================
st.title("Conversational SQL Assistant")

st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Select LLM (OpenRouter)", MODELS, index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.7, 0.1)
mode = st.sidebar.radio("Choose Mode", ["Direct Chat", "RAG on Docs", "SQL Database"])

st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat"):
    clear_history()
    st.sidebar.success("Chat history cleared.")


# ======================
# MODE: Direct Chat
# ======================
if mode == "Direct Chat":
    st.subheader("Direct Chat (No RAG)")
    user_input = st.text_area("Your message:", height=120, key="direct_input")
    if st.button("Send", key="direct_send") and user_input.strip():
        llm = get_chat_model(selected_model, temperature)
        add_to_history("user", user_input, {"mode": "direct"})
        try:
            resp = llm.invoke(user_input).content
        except Exception as e:
            resp = f"Error from LLM: {e}"
        add_to_history("assistant", resp, {"mode": "direct"})
        st.success("Response added below 👇")


# ======================
# MODE: RAG on Docs
# ======================
elif mode == "RAG on Docs":
    st.subheader("RAG on Documents")
    rag_backend = st.radio("Vector Backend", ["FAISS (Session)", "Chroma (Persistent)"], horizontal=True)

    accepted_types = ["pdf", "txt", "md"]
    if HAVE_CSV:
        accepted_types += ["csv"]
    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=accepted_types
    )

    if rag_backend == "FAISS (Session)":
        st.caption("FAISS is in-memory for this session.")
        retriever = None
        if uploaded_files:
            retriever = build_faiss_retriever(uploaded_files)
            if retriever:
                st.success("FAISS index built for this session.")
        user_input = st.text_area("Ask based on your docs:", height=120, key="rag_faiss_q")
        if st.button("Run RAG", key="rag_faiss_btn") and user_input.strip():
            if not retriever:
                st.warning("Upload docs first to build FAISS index.")
            else:
                llm = get_chat_model(selected_model, temperature)
                template = """Use the context to answer. If unknown, say you don't know.

Context:
{context}

Question:
{question}"""
                prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt}
                )
                add_to_history("user", user_input, {"mode": "rag-faiss"})
                try:
                    resp = qa.run(user_input)
                except Exception as e:
                    resp = f"RAG error: {e}"
                add_to_history("assistant", resp, {"mode": "rag-faiss"})
                st.success("Answer added below")

    else:  # Chroma persistent
        st.caption(f"Chroma persists to: `{CHROMA_DB_DIR}`")
        if uploaded_files:
            vs = add_files_to_chroma(uploaded_files)
            st.success("Documents added to persistent Chroma.")
        user_input = st.text_area("Ask based on your (persistent) KB:", height=120, key="rag_chroma_q")
        if st.button("Run RAG", key="rag_chroma_btn") and user_input.strip():
            vs = get_chroma_vectorstore()
            retriever = vs.as_retriever(search_kwargs={"k": 3})
            llm = get_chat_model(selected_model, temperature)
            template = """Answer using only the context. If unknown, say you don't know.
Return concise, helpful answers.

Context:
{context}

Question:
{question}"""
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )
            add_to_history("user", user_input, {"mode": "rag-chroma"})
            try:
                resp = qa.run(user_input)
            except Exception as e:
                resp = f"RAG error: {e}"
            add_to_history("assistant", resp, {"mode": "rag-chroma"})
            st.success("Answer added below")


# ======================
# MODE: SQL Database (MySQL + SQLite)
# ======================
# else:
  # ======================
# MODE: SQL Database (MySQL + SQLite)
# ======================
# else:
    # ======================
# MODE: SQL Database (MySQL + SQLite)
# ======================
else:
    import re, tempfile, pandas as pd
    from langchain_community.utilities import SQLDatabase
    from langchain.chains import create_sql_query_chain

    st.subheader("🗄️ SQL Database Chat")
    db_type = st.radio("Database Type", ["MySQL", "SQLite"], horizontal=True)
    allow_writes = st.checkbox("Allow write/DDL queries (dangerous)", value=False)
    st.caption("By default, potentially destructive SQL (INSERT/UPDATE/DELETE/DDL) is blocked.")

    db = None
    uri = ""

    # === Database Connection ===
    if db_type == "MySQL":
        st.markdown("**Connect to MySQL**")
        c1, c2 = st.columns(2)
        with c1:
            mysql_user = st.text_input("Username", value="root")
            mysql_host = st.text_input("Host", value="localhost")
            mysql_port = st.number_input("Port", min_value=1, max_value=65535, value=3306)
        with c2:
            mysql_password = st.text_input("Password", type="password")
            mysql_dbname = st.text_input("Database Name")
        connect = st.button("🔌 Connect to MySQL")
        if connect:
            try:
                uri = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{int(mysql_port)}/{mysql_dbname}"
                db = SQLDatabase.from_uri(uri)
                st.success("✅ Connected to MySQL.")
                st.text_area("Tables (preview):", value=list_tables_info(db), height=180, disabled=True)
                st.session_state.sql_db_uri = uri
            except Exception as e:
                st.error(f"Connection failed: {e}")
        elif "sql_db_uri" in st.session_state and str(st.session_state.sql_db_uri).startswith("mysql+"):
            try:
                db = SQLDatabase.from_uri(st.session_state.sql_db_uri)
                st.info("Reusing previous MySQL connection.")
                st.text_area("Tables (preview):", value=list_tables_info(db), height=180, disabled=True)
            except Exception as e:
                st.error(f"Re-connect failed: {e}")

    else:  # SQLite
        st.markdown("**Upload a SQLite file (.db/.sqlite)**")
        uploaded_db = st.file_uploader("Upload SQLite DB", type=["db", "sqlite"])
        if uploaded_db:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(uploaded_db.getbuffer())
                sqlite_path = tmp.name
            try:
                uri = f"sqlite:///{sqlite_path}"
                db = SQLDatabase.from_uri(uri)
                st.success("SQLite loaded.")
                st.text_area("Tables (preview):", value=list_tables_info(db), height=180, disabled=True)
                st.session_state.sql_db_uri = uri
            except Exception as e:
                st.error(f"SQLite load failed: {e}")
        elif "sql_db_uri" in st.session_state and str(st.session_state.sql_db_uri).startswith("sqlite:///"):
            try:
                db = SQLDatabase.from_uri(st.session_state.sql_db_uri)
                st.info("Reusing previously uploaded SQLite file.")
                st.text_area("Tables (preview):", value=list_tables_info(db), height=180, disabled=True)
            except Exception as e:
                st.error(f"Re-open failed: {e}")

    # === Query Input ===
    question = st.text_area("Ask your database (natural language):", height=120, key="sql_question")
    run_sql = st.button("Run", key="sql_run")

    if run_sql and question.strip():
      if not db:
          st.warning("Connect to a database first.")
      else:
          # 1) Create LLM
          llm_zero_temp = get_chat_model(selected_model, temperature=0.0)

          # 2) Build the SQL generator chain
          try:
              sql_gen_chain = create_sql_query_chain(llm_zero_temp, db)
          except Exception as e:
              st.error(f"Failed to initialize SQL chain: {e}")
              sql_gen_chain = None

          if sql_gen_chain:
              try:
                  # 3) Generate SQL from the question
                  generated_sql = sql_gen_chain.invoke({"question": question})

                  # --- Clean output to extract only SQL ---
                  if isinstance(generated_sql, dict) and "sql" in generated_sql:
                      generated_sql = generated_sql["sql"]
                  generated_sql = str(generated_sql)

                  # Remove markdown or explanations
                  generated_sql = (
                      generated_sql.replace("```sql", "")
                      .replace("```", "")
                      .replace("SQLQuery:", "")
                      .replace("Question:", "")
                      .strip()
                  )

                  # ✅ NEW robust cleanup: keep only text after first SQL keyword
                  generated_sql = re.sub(
                      r"(?is)^.*?\b(SELECT|SHOW|WITH|EXPLAIN)\b",
                      r"\1",
                      generated_sql
                  )

                  # ✅ Optional: enforce SELECT-only if writes disabled
                  if not allow_writes and is_write_query(generated_sql):
                      st.error("❌ Blocked: This SQL may modify data. Enable 'Allow write' to proceed.")
                      st.stop()

                  # ✅ Keep only first SQL statement
                  if ";" in generated_sql:
                      generated_sql = generated_sql.split(";")[0].strip() + ";"

                  # ✅ Ensure it's a valid SELECT-like query
                  if not re.match(r"(?is)^(SELECT|SHOW|WITH|EXPLAIN)", generated_sql):
                      st.error("❌ Generated query does not appear to be valid SQL.")
                      st.code(generated_sql, language="sql")
                      st.stop()

                  # ✅ --- NEW LIMIT CONTROL LOGIC ---
                  default_limit = 1000  # or any number you prefer as max
                  if re.search(r"\ball\b", question, re.I):
                      # If the user asked for "all", remove LIMIT
                      generated_sql = re.sub(r"(?i)\s+LIMIT\s+\d+\s*;?", "", generated_sql)
                      st.info("Detected 'all' → removed LIMIT clause.")
                  elif re.search(r"(?i)LIMIT\s+\d+", generated_sql):
                      # If query already has a limit, optionally raise it if too low
                      match = re.search(r"(?i)LIMIT\s+(\d+)", generated_sql)
                      if match and int(match.group(1)) < default_limit:
                          generated_sql = re.sub(r"(?i)LIMIT\s+\d+", f"LIMIT {default_limit}", generated_sql)
                          st.info(f"Increased LIMIT to {default_limit}")
                  else:
                      # If no limit and user didn’t say “all”, add a default limit
                      generated_sql = generated_sql.rstrip(";") + f" LIMIT {default_limit};"

                  # 4) Execute SQL safely
                  try:
                      rows = db.run(generated_sql)
                  except Exception as e:
                      rows = f"Execution error: {e}"

                  # 5) Convert results into natural-language answer
                  final_answer = answer_from_sql(llm_zero_temp, question, generated_sql, rows)

                  # Save to chat history
                  add_to_history("user", question, {"mode": "sql", "db_type": db_type})
                  add_to_history(
                      "assistant",
                      final_answer,
                      {"mode": "sql", "db_type": db_type, "sql": generated_sql},
                  )

                  # Display results
                  with st.expander("🔎 Generated SQL", expanded=True):
                      st.code(generated_sql, language="sql")

                  with st.expander("🧮 Raw Rows (truncated)", expanded=False):
                      st.write(rows)

                  st.success("✅ Answer generated successfully!")

              except Exception as e:
                  st.error(f"SQL pipeline error: {e}")




# ======================
# CHAT WINDOW
# ======================
st.markdown("---")
st.markdown("### Conversation")
if not st.session_state.history:
    st.info("No messages yet. Ask something!")
else:
    for msg in st.session_state.history[-200:]:
        role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Assistant"
        meta = msg.get("meta") or {}
        tag_mode = meta.get("mode", "")
        show_tag = f" · *{tag_mode}*" if tag_mode else ""
        st.markdown(f"**{role}{show_tag}**  \n{msg['content']}")
