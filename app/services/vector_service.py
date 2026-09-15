import uuid

from app.core.pinecone_client import index


class VectorService:

    def upsert_document(
        self,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]]
    ) -> None:

        vectors = []

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunk
                    }
                }
            )

        # Upload vectors in batches
        batch_size = 100

        for start in range(0, len(vectors), batch_size):

            batch = vectors[start:start + batch_size]

            index.upsert(
                vectors=batch
            )

            print(
                f"Stored vectors "
                f"{start + 1}-{min(start + batch_size, len(vectors))} "
                f"of {len(vectors)}"
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ):

        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return response.matches