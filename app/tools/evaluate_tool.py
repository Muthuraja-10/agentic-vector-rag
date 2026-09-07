from groq import Groq,RateLimitError

from app.core.config import settings


client = Groq(
    api_key=settings.GROQ_API_KEY
)


def evaluate_context(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are a retrieval evaluator for an Agentic RAG system.

Question:
{question}

Retrieved Context:
{context}

Your job is ONLY to decide whether the retrieved context contains enough
information to answer the user's question.

Rules:

- Return YES only if the answer is directly supported by the retrieved context.
- Return NO if the context is unrelated.
- Return NO if only a few keywords match.
- Return NO if the context is incomplete.
- Do NOT use your own knowledge.
- Judge ONLY from the retrieved context.

Reply with ONLY one word:

YES

or

NO
"""
    try:

        response = client.chat.completions.create(

          model="openai/gpt-oss-120b",

          messages=[
            {
                "role": "user",
                "content": prompt
            }
         ],

         temperature=0

       )
    except RateLimitError:

           return "NO"
        

    decision = response.choices[0].message.content.strip().upper()

    if "YES" in decision:
        return "YES"

    return "NO"