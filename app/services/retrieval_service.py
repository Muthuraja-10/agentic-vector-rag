from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


class RetrievalService:

    def __init__(self):

        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()

    def retrieve(
        self,
        question: str,
        top_k: int = 3
    ) -> list[dict]:

        # Step 1: Convert question into embedding
        query_embedding = self.embedding_service.generate_query_embedding(
            question
        )

        # Step 2: Search similar vectors in Pinecone
        matches = self.vector_service.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        
        # Step 3: Store retrieved chunks
        retrieved_chunks = []

        for match in matches:

            print(
        f"\n--- Retrieved Chunk ---"
        f"\nRank: {len(retrieved_chunks) + 1}"
        f"\nScore: {match.score}"
        f"\nFile: {match.metadata['filename']}"
        f"\nChunk Index: {match.metadata['chunk_index']}"
        f"\nText: {match.metadata['text'][:300]}..."
    )
            retrieved_chunks.append(
                {
                    "score": match.score,
                    "filename": match.metadata["filename"],
                    "chunk_index": match.metadata["chunk_index"],
                    "text": match.metadata["text"]
                }
            )

        # Step 4: Return retrieved chunks
        return retrieved_chunks