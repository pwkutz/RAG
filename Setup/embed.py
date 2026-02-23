import ollama

from setup import dataset


class Embedder:

    def __init__(self, dataset: list[str]):

        self.EMBEDDING_MODEL: str = r'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
        self.VECTOR_DB: list[tuple[str, list[float]]] = []
        self.dataset: list[str] = dataset
        self.query: str = ""
        self.similarities: list[tuple[str, float]] = []


    def add_chunk_to_database(self, chunk):

        """Embedder Model tokenises and vectorises incoming chunk by contextual embedding."""

        embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        self.VECTOR_DB.append((chunk, embedding))

    def embed_database(self):

        """embed one after another all chunks of the given dataset"""

        for i, chunk in enumerate(self.dataset):
            self.add_chunk_to_database(chunk)
            print(f'Added chunk {i + 1}/{len(self.dataset)} to the database')

    def retrieve(self, query, top_n=3):
        query_embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=query)['embeddings'][0]
        # temporary list to store (chunk, similarity) pairs
        self.similarities: list[tuple[str, float]] = []
        for chunk, embedding in self.VECTOR_DB:
            similarity: float = cosine_similarity(query_embedding, embedding)
            self.similarities.append((chunk, similarity))
        # sort by similarity (1=similar | 0 = not similar) in descending order, because higher similarity means more relevant chunks
        self.similarities.sort(key=lambda x: x[1], reverse=True)
        # finally, return the top N most relevant chunks
        self.similarities: list[tuple[str, float]] = self.similarities[:top_n]
        return self.similarities


def cosine_similarity(a, b) -> float:

    """cosine similarity function: used to compute the distance between two vectors in vector space aka. compute their similarity"""

    dot_product: float = sum([x * y for x, y in zip(a, b)])
    norm_a: float = sum([x ** 2 for x in a]) ** 0.5
    norm_b: float = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def show_knowledge(retrieved_knowledge: list[tuple[str, float]]):

    print('Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = f'''You are a helpful chatbot.
     Use only the following pieces of context to answer the question. Don't make up any new information:
     {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
     '''


def main(dataset: list[str], input_query: str):

    embedding: Embedder = Embedder(dataset = dataset)
    embedding.embed_database() # embed aka. vectorise all chunks

    retrieved_knowledge = embedding.retrieve(input_query) # find N most similar text chunks to the inputted query


