import ollama

class Embedder:

    def __init__(self):

        self.EMBEDDING_MODEL: str = r'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
        self.VECTOR_DB: list[tuple[str, list[float]]] = []

    def add_chunk_to_database(self, chunk):

        """Embedder Model tokenises and vectorises incoming chunk by contextual embedding."""

        embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        self.VECTOR_DB.append((chunk, embedding))

