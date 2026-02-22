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

        """embed one after another all chunks of the given dataset"""

        for i, chunk in enumerate(self.dataset):
            self.add_chunk_to_database(chunk)
            print(f'Added chunk {i + 1}/{len(self.dataset)} to the database')

def cosine_similarity(a, b) -> float:

    """cosine similarity function: used to compute the distance between two vectors in vector space aka. compute their similarity"""

    dot_product: float = sum([x * y for x, y in zip(a, b)])
    norm_a: float = sum([x ** 2 for x in a]) ** 0.5
    norm_b: float = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)




def main(dataset: list[str]):

    embedding: Embedder = Embedder(dataset = dataset)
    embedding.embed_database()
