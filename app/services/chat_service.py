from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService


class ChatService:

    def __init__(self):

        self.retrieval_service = RetrievalService()
        self.llm_service = LLMService()

    def chat(
        self,
        question: str
    ):

        # Retrieve relevant chunks
        chunks = self.retrieval_service.retrieve(question)

        # Generate answer using Groq
        answer = self.llm_service.generate_answer(
            question=question,
            chunks=chunks
        )

        return {
            "question": question,
            "answer": answer
        }