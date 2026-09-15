from app.core.pinecone_client import pc


class EmbeddingService:

    def generate_document_embeddings(
        self,
        chunks: list[str]
    ) -> list[list[float]]:

        embeddings = []

        batch_size = 96

        for start in range(0, len(chunks), batch_size):

            batch = chunks[start:start + batch_size]

            print(
                f"Embedding batch: "
                f"{start + 1}-{min(start + batch_size, len(chunks))} "
                f"of {len(chunks)} chunks"
            )

            response = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=batch,
                parameters={
                    "input_type": "passage",
                    "truncate": "END"
                }
            )

            embeddings.extend(
                item.values
                for item in response.data
            )

        return embeddings


    def generate_query_embedding(
        self,
        query: str
    ) -> list[float]:

        response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={
                "input_type": "query",
                "truncate": "END"
            }
        )

        return response.data[0].values