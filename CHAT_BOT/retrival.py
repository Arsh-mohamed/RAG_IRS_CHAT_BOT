import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from langchain_chroma import Chroma
from embedding import OpenAIEmbeddingsAdapter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "DATA"
DB_PATH = BASE_DIR / "db" / "sqlite.db"
CHROMA_PATH = BASE_DIR / "db" / "chroma"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def get_store(
    collection_name="IRS_Publication15T",
    persist_directory=CHROMA_PATH,
):
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=OpenAIEmbeddingsAdapter(),
    )


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


def _filing_status(query):
    normalized = query.lower().replace("-", " ").replace("_", " ")
    statuses = {
        "married filing jointly": "married_filing_jointly",
        "married jointly": "married_filing_jointly",
        "jointly": "married_filing_jointly",
        "married filing separately": "married_filing_separately",
        "married separately": "married_filing_separately",
        "separately": "married_filing_separately",
        "head of household": "head_of_household",
        "single": "single",
    }
    for phrase, status in sorted(statuses.items(), key=lambda item: -len(item[0])):
        if phrase in normalized:
            return status
    return None


def _income(query):
    patterns = (
        r"(?:make|earn|income|salary|wage)[^$\d]{0,20}\$?([\d,]+(?:\.\d+)?)\s*(k|thousand)?",
        r"\$([\d,]+(?:\.\d+)?)\s*(k|thousand)?",
    )
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if not match:
            continue
        value = float(match.group(1).replace(",", ""))
        if match.group(2):
            value *= 1000
        return value
    return None


def _table_with_columns(connection, required_columns):
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    for (table_name,) in tables:
        columns = {
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info("{table_name.replace(chr(34), chr(34) * 2)}")'
            )
        }
        if required_columns.issubset(columns):
            return table_name
    return None


def _source_file(table_name):
    normalized_table = re.sub(r"[^a-z0-9]", "", table_name.lower())
    for path in DATA_DIR.iterdir():
        normalized_stem = re.sub(r"[^a-z0-9]", "", path.stem.lower())
        if (
            normalized_stem == normalized_table
            or normalized_table.startswith(normalized_stem)
            or normalized_stem.startswith(normalized_table)
        ):
            return path.name
    return table_name


def _pdf_document(store):
    collection_name = getattr(store, "_collection_name", "IRS_Publication15T")
    candidate = DATA_DIR / f"{collection_name}.pdf"
    return candidate.name if candidate.exists() else f"{collection_name}.pdf"


def sqlite_tax_answer(query, db_path=DB_PATH):
    query_lower = query.lower()
    asks_tax_data = any(
        term in query_lower
        for term in ("tax bracket", "marginal rate", "standard deduction", "taxable income")
    )
    income = _income(query)
    status = _filing_status(query)
    asks_deduction = "standard deduction" in query_lower
    if not asks_tax_data or status is None or (income is None and not asks_deduction):
        return None

    year_match = re.search(r"\b(20\d{2})\b", query)
    tax_year = int(year_match.group(1)) if year_match else 2026
    if not db_path.exists():
        return None

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        table_name = _table_with_columns(
            connection,
            {
                "tax_year",
                "filing_status",
                "marginal_rate_percent",
                "standard_deduction_usd",
            },
        )
        if table_name is None:
            return None
        if income is None:
            row = connection.execute(
                f'SELECT * FROM "{table_name}" WHERE tax_year = ? AND filing_status = ? LIMIT 1',
                (tax_year, status),
            ).fetchone()
        else:
            row = connection.execute(
                f"""
                SELECT * FROM "{table_name}"
                WHERE tax_year = ?
                  AND filing_status = ?
                  AND (taxable_income_over_usd IS NULL OR taxable_income_over_usd <= ?)
                  AND (taxable_income_up_to_usd IS NULL OR ? < taxable_income_up_to_usd)
                ORDER BY taxable_income_over_usd IS NOT NULL DESC,
                         taxable_income_over_usd DESC
                LIMIT 1
                """,
                (tax_year, status, income, income),
            ).fetchone()

    if row is None:
        return None

    if income is None:
        answer = (
            f"For {tax_year} {status.replace('_', ' ')}, the standard deduction is "
            f"${row['standard_deduction_usd']:,.0f}."
        )
    else:
        upper = row["taxable_income_up_to_usd"]
        upper_text = f"up to ${upper:,.0f}" if upper is not None else "above the last threshold"
        answer = (
            f"For {tax_year} {status.replace('_', ' ')}, taxable income of ${income:,.0f} "
            f"falls in the {row['marginal_rate_percent']}% marginal tax bracket "
            f"({upper_text}). The standard deduction is ${row['standard_deduction_usd']:,.0f}."
        )
    return {
        "answer": answer,
        "source": "SQLite",
        "document_number": _source_file(table_name),
        "metadata": {"source_url": row["primary_source_url"], "tax_year": tax_year, "table": table_name},
    }


def sqlite_historical_answer(query, db_path=DB_PATH):
    year_match = re.search(r"\b(199\d|20(?:0\d|1\d|2[0-3]))\b", query)
    if year_match is None:
        return None

    query_lower = query.lower()
    history_terms = (
        "historical",
        "history",
        "all returns",
        "electronically filed",
        "income tax returns",
    )
    if not any(term in query_lower for term in history_terms):
        return None

    tax_year = int(year_match.group(1))
    if not db_path.exists():
        return None

    with sqlite3.connect(db_path) as connection:
        table_name = _table_with_columns(connection, {"table_a.__all_individual_income_tax_returns:_selected_income_and_tax_items_"})
        if table_name is None:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            table_name = next(
                (name for (name,) in tables if "historical" in name.lower()),
                None,
            )
        if table_name is None:
            return None
        rows = connection.execute(f'SELECT * FROM "{table_name}"').fetchall()

    if len(rows) < 3:
        return None

    year_row = rows[2]
    year_columns = {}
    for index, value in enumerate(year_row):
        if index == 0 or value is None:
            continue
        try:
            normalized_year = int(float(value))
        except (TypeError, ValueError):
            continue
        if 1990 <= normalized_year <= 2023:
            year_columns[normalized_year] = index
    year_index = year_columns.get(tax_year)
    if year_index is None:
        return None

    candidates = []
    query_tokens = set(tokenize(query_lower))
    for row in rows[5:]:
        if not row[0] or not isinstance(row[0], str):
            continue
        label = row[0].strip()
        label_tokens = set(tokenize(label))
        if label_tokens and label_tokens.issubset(query_tokens):
            candidates.append((len(label_tokens), label, row[year_index]))

    if not candidates and "income tax returns" in query_lower:
        for row in rows[5:]:
            if isinstance(row[0], str) and row[0].strip().lower() == "all returns":
                candidates.append((2, "All returns", row[year_index]))
                break
    if not candidates:
        return None

    _, label, value = max(candidates, key=lambda item: item[0])
    if value is None:
        return None
    try:
        numeric_value = float(str(value).replace(",", ""))
        value = int(numeric_value) if numeric_value.is_integer() else numeric_value
        value_text = f"{value:,}"
    except (TypeError, ValueError):
        value_text = str(value)

    answer = f"In {tax_year}, {label} was {value_text}."
    return {
        "answer": answer,
        "source": "SQLite",
        "document_number": _source_file(table_name),
        "metadata": {"table": table_name, "tax_year": tax_year, "item": label},
    }


def answer_question(query, store=None, top_k=5):
    structured = sqlite_tax_answer(query)
    if structured is not None:
        return [structured]

    historical = sqlite_historical_answer(query)
    if historical is not None:
        return [historical]

    store = store or get_store()
    results = hybrid_search(store, query, top_k=top_k)
    answers = []
    for result in results:
        metadata = result["metadata"]
        answers.append({
            "answer": result["text"],
            "source": "PDF",
            "document_number": metadata.get("source") or _pdf_document(store),
            "chunk_index": metadata.get("chunk_index", "unknown"),
            "metadata": metadata,
        })
    return answers


def print_answers(query, answers):
    print(f"\nQuestion: {query}")
    if not answers:
        print("No answer found in SQLite or the IRS PDF.")
        return
    for index, result in enumerate(answers, start=1):
        print(f"\n[{index}] Answer: {result['answer'][:1000].replace(chr(10), ' ')}")
        print(f"    Source: {result['source']}")
        print(f"    Document: {result['document_number']}")
        if "chunk_index" in result:
            print(f"    Chunk index: {result['chunk_index']}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if question:
        print_answers(question, answer_question(question))
    else:
        while True:
            question = input("\nAsk a tax question (q to quit): ").strip()
            if question.lower() in {"q", "quit", "quite"}:
                print("Goodbye!")
                break
            if not question:
                continue
            print_answers(question, answer_question(question))
