from groq import Groq


# Initialize Groq client
client = Groq()


def route_question(question: str) -> str:
    """
    LLM-based router agent.
    Decides what action to take for a given user question.
    """

    router_prompt = f"""
You are a router agent.

Your task is to classify the user's question into exactly ONE of the following actions:

1. direct_answer – if the question can be answered without using the uploaded documents.
2. rag_search – if the question requires searching or referencing the uploaded documents.
3. summarize – if the user is asking for a summary or overview of the document.

Rules:
- Respond with ONLY ONE word.
- Do NOT explain your reasoning.
- Do NOT answer the question.
- Your response must be exactly one of:
  direct_answer, rag_search, summarize.

User question:
"{question}"
"""

    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "user", "content": router_prompt}
        ],
        temperature=0.0  # VERY IMPORTANT: deterministic output
    )

    decision = response.choices[0].message.content.strip().lower()

    # Safety fallback
    if decision not in {"direct_answer", "rag_search", "summarize"}:
        return "rag_search"

    return decision
