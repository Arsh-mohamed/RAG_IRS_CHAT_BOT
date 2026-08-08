from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from langchain.embeddings import Embeddings
from langchain_chroma import Chroma
from openai import OpenAI
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required in the environment or .env file")


class OpenAIEmbeddingsAdapter(Embeddings):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        super().__init__()
        self.model = model
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.batch_size = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "64"))
        if self.batch_size < 1:
            raise ValueError("OPENAI_EMBEDDING_BATCH_SIZE must be at least 1")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    adapter = OpenAIEmbeddingsAdapter()
    return adapter.embed_documents(texts)


def create_chroma_store(
    texts: list[str],
    metadatas: list[dict] | None = None,
    ids: list[str] | None = None,
    collection_name: str = "rag_docs",
    persist_directory: str | Path = "db/chroma",
    model: str = "text-embedding-3-small",
) -> Chroma:
    persist_directory = str(Path(persist_directory))
    embeddings = OpenAIEmbeddingsAdapter(model=model)
    chroma_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return chroma_store


def store_chunks_in_chroma(
    chunks: list,
    collection_name: str = "rag_chunks",
    persist_directory: str | Path = "db/chroma",
) -> Chroma:
    texts = [getattr(chunk, "text", str(chunk)) for chunk in chunks]
    metadatas: list[dict] = []

    for idx, chunk in enumerate(chunks):
        metadata = getattr(chunk, "metadata", None)
        if isinstance(metadata, dict):
            metadata_dict = metadata.copy()
        else:
            metadata_dict = {
                "source": str(getattr(chunk, "origin", None) or metadata or "")
            }
        metadata_dict["chunk_index"] = idx
        metadatas.append(metadata_dict)

    ids = [str(idx) for idx in range(len(chunks))]
    return create_chroma_store(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
