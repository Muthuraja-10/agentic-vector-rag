from app.core.pinecone_client import pc


class EmbeddingService:

    def generate_document_embeddings(
        self,
        chunks: list[str]
    ) -> list[list[float]]:

        response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=chunks,
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )

        return [
            item.values
            for item in response.data
        ]


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