"""
embeddings.py

Embedding functions used to build/query the Chroma vector store.

- LocalEmbeddingFunction: talks to a local, OpenAI-API-compatible server
  (Ollama, vLLM, LM Studio, text-generation-webui, ...) serving
  BGE-large-en-v1.5 (or any other embedding model you point it at) over
  POST /v1/embeddings. Nothing is downloaded or run in-process here --
  this project stays lightweight and just makes HTTP calls; the actual
  model runs wherever you're serving it.
- MockEmbeddingFunction: deterministic hash-based embeddings with no
  network calls, for offline testing (config.app.use_mock_llm: true, or
  env var USE_MOCK_LLM=true).

Both implement chromadb's EmbeddingFunction protocol: __call__(input) ->
list[list[float]], plus embed_query() for asymmetric query/document
embedding. BGE models are trained with a specific instruction prefix on
the QUERY side only (not on documents) -- see QUERY_INSTRUCTION below --
which meaningfully improves retrieval quality and is easy to forget when
switching embedding backends, so it lives here rather than being left to
the caller.
"""

import hashlib
import struct
from chromadb import EmbeddingFunction, Documents, Embeddings


from sentence_transformers import SentenceTransformer   


class BGEEmbeddingFunction(EmbeddingFunction):

    QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )

    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            list(input),
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, input: Documents) -> Embeddings:
        prefixed = [
            self.QUERY_INSTRUCTION + text
            for text in input
        ]

        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True
        )

        return embeddings.tolist()


class MockEmbeddingFunction(EmbeddingFunction):
    """
    Deterministic, dependency-free stand-in for real embeddings.
    Produces a fixed-length vector via hashed bag-of-words, so texts that
    share vocabulary end up with non-trivial cosine similarity -- good
    enough to exercise the retrieval/threshold logic in tests without any
    network access or a running local model server.
    """

    DIM = 256

    def __init__(self, *_, **__):
        pass

    @staticmethod
    def name() -> str:
        return "mock-embedding-function"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        words = [w for w in text.lower().split() if len(w) > 2]
        for w in words:
            h = hashlib.md5(w.encode("utf-8")).digest()
            idx = struct.unpack("I", h[:4])[0] % self.DIM
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._vector(t) for t in input]

    def embed_query(self, input: Documents) -> Embeddings:
        return self.__call__(input)


def get_embedding_function(config: dict):
    if config["app"]["use_mock_llm"]:
        return MockEmbeddingFunction()

    return BGEEmbeddingFunction(
        "BAAI/bge-large-en-v1.5"
    )