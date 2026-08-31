import chromadb
from src.embeddings import get_embedding_function


def get_collection(config: dict):
    client = chromadb.PersistentClient(path=config["chroma"]["persist_directory"])
    embed_fn = get_embedding_function(config)
    
    collection = client.get_or_create_collection(
        name=config["chroma"]["collection_name"],
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _doc_text(subject: str, body: str) -> str:
    return f"{subject}\n\n{body}".strip()


def index_tickets(collection, tickets: list[dict], batch_size: int = 64) -> int:
    """
    Upsert a list of ticket dicts (id, subject, body, answer, type, queue,
    priority, source) into the collection. Returns the number indexed.
    """
    total = 0
    for i in range(0, len(tickets), batch_size):
        batch = tickets[i:i + batch_size]
        ids = [str(t["id"]) for t in batch]
        docs = [_doc_text(t.get("subject", ""), t.get("body", "")) for t in batch]
        metas = [{
            "ticket_id": t["id"],
            "subject": t.get("subject", "") or "",
            "answer": t.get("answer", "") or "",
            "type": t.get("type", "") or "",
            "queue": t.get("queue", "") or "",
            "priority": t.get("priority", "") or "",
            "source": t.get("source", "historical") or "historical",
        } for t in batch]
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        total += len(batch)
    return total


def add_single_ticket(collection, ticket_id: int, subject: str, body: str,
                       answer: str, type_: str, queue: str, priority: str, source: str) -> None:
    collection.upsert(
        ids=[str(ticket_id)],
        documents=[_doc_text(subject, body)],
        metadatas=[{
            "ticket_id": ticket_id, "subject": subject, "answer": answer,
            "type": type_, "queue": queue, "priority": priority, "source": source,
        }],
    )


def delete_ticket(collection, ticket_id: int) -> None:
    """Remove a ticket from the vector index (id is stored as a string)."""
    collection.delete(ids=[str(ticket_id)])


def query_similar(collection, query_subject: str, query_body: str, top_k: int = 5) -> dict:
    """
    Returns dict with parallel lists: ids, documents, metadatas, similarities
    (cosine similarity in [0, 1]-ish range, higher = more similar).
    """
    query_text = _doc_text(query_subject, query_body)
    embed_fn = collection._embedding_function
    query_embedding = embed_fn.embed_query([query_text])

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    distances = result["distances"][0] if result["distances"] else []
    # chroma cosine "distance" is (1 - cosine_similarity) for cosine space
    similarities = [max(0.0, 1.0 - d) for d in distances]

    return {
        "ids": result["ids"][0] if result["ids"] else [],
        "documents": result["documents"][0] if result["documents"] else [],
        "metadatas": result["metadatas"][0] if result["metadatas"] else [],
        "similarities": similarities,
    }
