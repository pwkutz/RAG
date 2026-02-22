import ollama

from setup import dataset


class Embedder:

    def __init__(self, dataset: list[str]):

        self.EMBEDDING_MODEL: str = r'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
        self.VECTOR_DB: list[tuple[str, list[float]]] = []
        self.dataset: list[str] = dataset


    def add_chunk_to_database(self, chunk):

        """Embedder Model tokenises and vectorises incoming chunk by contextual embedding."""

        embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        self.VECTOR_DB.append((chunk, embedding))

    def embed_database(self):

        for i, chunk in enumerate(self.dataset):
            self.add_chunk_to_database(chunk)
            print(f'Added chunk {i + 1}/{len(self.dataset)} to the database')

def main(dataset: list[str]):

    embedding: Embedder = Embedder(dataset = dataset)
    embedding.embed_database()
