from app.services.retrieval_service import RetrievalService

retrieval_service = RetrievalService()

def retrieve_documents(question:str):

    return retrieval_service.retrieve(question)
