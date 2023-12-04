from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


client = OpenAI()


def compute_similarity(code_snippets, client, model="text-embedding-ada-002"):
    """
    Embed multiple code snippets using OpenAI's embedding API and compute the cosine similarity matrix.
    """
    response = client.embeddings.create(
        input=code_snippets,
        model=model,
    )
    embeddings = [_data.embedding for _data in response.data]
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix



if __name__ == '__main__':

    # Example usage
    code_snippets = [
        "def add(a, b): return a + b",
        "def subtract(a, b): return a - b",
        "def multiply(a, b): return a * b"
    ]

    similarity_matrix = compute_similarity(code_snippets, client)
