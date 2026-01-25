from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from groq import Groq

if "ingested" not in st.session_state:
    st.session_state.ingested = False

from agents.router_agent import route_question
from tools.rag_tools import rag_answer
from rag.ingestion import ingest_pdfs



# Initialize Groq client
client = Groq()


# ---------------- Streamlit config ---------------- #
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Agentic RAG Chatbot")
st.write("Ask questions about your uploaded documents.")


# ---------------- Direct LLM answer ---------------- #
def direct_answer(question: str) -> str:
    """
    Answer directly using LLM (no retrieval).
    """
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


# ---------------- Sidebar: PDF Upload ---------------- #
st.sidebar.title("📄 Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.ingested:
    os.makedirs("data/pdfs", exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(
            os.path.join("data/pdfs", uploaded_file.name),
            "wb"
        ) as f:
            f.write(uploaded_file.read())

    if st.sidebar.button("📥 Ingest Documents"):
        with st.spinner("Ingesting PDFs..."):
            ingest_pdfs(pdf_dir="data/pdfs")
        st.session_state.ingested = True
        st.sidebar.success("✅ Documents ingested successfully!")


# ---------------- Chat UI ---------------- #
user_question = st.text_input("Ask a question:")

if user_question:
    with st.spinner("Thinking..."):
        # 1. Router decision
        decision = route_question(user_question)

        # Show router decision (debug/demo)
        st.caption(f"🧭 Router decision: `{decision}`")

        # 2. Execute decision
        if decision == "direct_answer":
            answer = direct_answer(user_question)

        elif decision == "rag_search":
            answer = rag_answer(user_question)

        else:
            # Safety fallback
            answer = rag_answer(user_question)

    # 3. Output
    st.markdown("### ✅ Answer")
    st.write(answer)
