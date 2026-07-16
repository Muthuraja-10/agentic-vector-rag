from groq import Groq,RateLimitError

from app.core.config import settings


client = Groq(
    api_key=settings.GROQ_API_KEY
)


def rewrite_query(question: str) -> str:

    prompt = f"""
You are an expert search query optimizer.

Rewrite the user's question so that it is
better suited for semantic vector search.

Do NOT answer the question.

Only return the improved search query.

Question:

{question}
"""
    try:

       response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0

       )
    except RateLimitError:

           return question

    return response.choices[0].message.content.strip()