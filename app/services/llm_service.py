from groq import Groq, RateLimitError

from app.core.config import settings


class LLMService:

    def __init__(self):

        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    def generate_answer(
        self,
        question: str,
        chunks: list[dict]
    ) -> str:

        context = "\n\n".join(
            chunk["text"]
            for chunk in chunks[:3]
        )

        prompt = f"""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:

"I couldn't find the answer in the uploaded documents."

Context:

{context}

Question:

{question}
"""

        try:

            response = self.client.chat.completions.create(
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

            return "Groq daily token limit reached. Please try again later."

        return response.choices[0].message.content