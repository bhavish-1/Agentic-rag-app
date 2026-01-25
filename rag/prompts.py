RAG_PROMPT = """
You are a helpful assistant.

Use the following context to answer the user's question.
If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
