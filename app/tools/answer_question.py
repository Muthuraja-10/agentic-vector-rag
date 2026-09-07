from groq import Groq,RateLimitError

from app.core.config import settings


client = Groq(
    api_key=settings.GROQ_API_KEY
)


def answer_question(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are a RAG assistant.

Answer the user's question ONLY using the retrieved context.

If the context does not contain enough information,
reply:

"I couldn't find enough information in the uploaded documents."

Question:
{question}

Retrieved Context:
{context}
"""
    try : 

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

          return "Groq daily token limit reached. Please try again later."

    return response.choices[0].message.content.strip()