from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker:

    #Splits cleaned text into overlapping chunks for embedding.

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)