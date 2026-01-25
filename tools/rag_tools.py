from groq import Groq

from rag.retriever import get_retriever
from rag.prompts import RAG_PROMPT


# Initialize Groq client
client = Groq()


def rag_answer(question: str) -> str:
    """
    Tool: Answer a question using retrieved documents (RAG).
    Returns ONLY the final answer.
    """

    # 1. Get retriever
    retriever = get_retriever()

    # 2. Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)

    # 3. Combine context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # 4. Format prompt
    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    # 5. Call LLM
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    # 6. Return final answer
    return response.choices[0].message.content.strip()
