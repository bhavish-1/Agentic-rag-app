from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from groq import Groq

from agents.router_agent import route_question
from tools.rag_tool import rag_answer


# Initialize Groq client
client = Groq()


st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Agentic RAG Chatbot")
st.write("Ask questions about your uploaded documents.")


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


# ---------------- UI ---------------- #

user_question = st.text_input("Ask a question:")

if user_question:
    with st.spinner("Thinking..."):
        # 1. Decide what to do
        decision = route_question(user_question)

        # Optional: show decision (great for demo/debug)
        st.caption(f"🧭 Router decision: `{decision}`")

        # 2. Act based on decision
        if decision == "direct_answer":
            answer = direct_answer(user_question)

        elif decision == "rag_search":
            answer = rag_answer(user_question)

        else:
            # Safety fallback
            answer = rag_answer(user_question)

    # 3. Show final answer
    st.markdown("### ✅ Answer")
    st.write(answer)
