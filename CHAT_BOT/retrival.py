import re
from collections import Counter
from langchain_chroma import Chroma


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def make_store(collection_name="rag_chunks", persist_directory="db/chroma"):
    return Chroma(collection_name=collection_name, persist_directory=str(persist_directory))


def dense_search(store, query, k=50):
    hits = store.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in hits:
        text = getattr(doc, "page_content", None) or str(doc)
        results.append({
            "text": text,
            "dense_score": float(score),
            "metadata": getattr(doc, "metadata", None) or {},
        })
    return results


def bm25_scores(query, texts):
    query_tokens = tokenize(query)
    if not query_tokens:
        return [0.0] * len(texts)

    docs = [tokenize(text) for text in texts]
    doc_freq = Counter({token for tokens in docs for token in set(tokens)})
    avg_len = sum(len(tokens) for tokens in docs) / max(len(docs), 1)
    k1, b = 1.5, 0.75

    scores = []
    for tokens in docs:
        counter = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if token not in counter:
                continue
            idf = max(0.0, (len(docs) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            freq = counter[token]
            denom = freq + k1 * (1 - b + b * len(tokens) / avg_len)
            score += idf * freq * (k1 + 1) / denom
        scores.append(score)
    return scores


def rerank(query, text):
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    overlap = query_tokens.intersection(text_tokens)
    return len(overlap) / len(query_tokens)


def hybrid_search(store, query, top_k=10, dense_k=50):
    dense_hits = dense_search(store, query, k=dense_k)
    if not dense_hits:
        return []

    texts = [hit["text"] for hit in dense_hits]
    bm25 = bm25_scores(query, texts)

    results = []
    for hit, score in zip(dense_hits, bm25):
        rerank_score = rerank(query, hit["text"])
        hybrid_score = hit["dense_score"] + 0.5 * score
        final_score = hybrid_score + 0.2 * rerank_score
        results.append({
            "text": hit["text"],
            "metadata": hit["metadata"],
            "dense_score": hit["dense_score"],
            "bm25_score": score,
            "rerank_score": rerank_score,
            "hybrid_score": final_score,
        })

    results.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    store = make_store(collection_name="rag_chunks", persist_directory="db/chroma")
    answer = hybrid_search(store, "tax bracket 2026 standard deduction", top_k=5)
    for item in answer:
        print(item["hybrid_score"], item["dense_score"], item["bm25_score"], item["rerank_score"], item["text"][:200].replace("\n", " "))
