"""
ingest.py — GitLab Handbook md dosyalarini oku, chunk'la, embed et, Chroma'ya yaz.
Calistir: python ingest.py
"""
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DB_PATH = "chroma_db"
COLLECTION = "handbook"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # app.py ile AYNI olmali
CHUNK_SIZE = 1500   # karakter
OVERLAP = 200
MIN_LEN = 80        # bundan kisa parcalari at


def clean(text: str) -> str:
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.S)   # Hugo front matter
    text = re.sub(r"\{\{[<%].*?[>%]\}\}", "", text, flags=re.S)  # Hugo shortcodes
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str) -> list[str]:
    # Once basliklara gore bol, uzun bolumleri overlap ile parcala
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    for s in sections:
        s = s.strip()
        if len(s) < MIN_LEN:
            continue
        step = CHUNK_SIZE - OVERLAP
        for i in range(0, len(s), step):
            part = s[i:i + CHUNK_SIZE]
            if len(part) >= MIN_LEN:
                chunks.append(part)
    return chunks


def main():
    files = list(DATA_DIR.rglob("*.md"))
    print(f"{len(files)} md dosyasi bulundu")

    ids, docs, metas = [], [], []
    for f in files:
        rel = f.relative_to(DATA_DIR).as_posix()
        dept = rel.split("/")[0]            # people-group / support
        text = clean(f.read_text(encoding="utf-8", errors="ignore"))
        for i, c in enumerate(chunk(text)):
            ids.append(f"{rel}::{i}")
            docs.append(c)
            metas.append({"source": rel, "dept": dept})

    print(f"{len(docs)} chunk olustu, embed ediliyor...")
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(docs, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    B = 5000  # Chroma batch limiti
    for i in range(0, len(ids), B):
        col.upsert(ids=ids[i:i+B], documents=docs[i:i+B],
                   embeddings=embs[i:i+B], metadatas=metas[i:i+B])
    print(f"Tamam. Collection'da {col.count()} chunk var.")

    # Hizli test
    q = "How many days of paid time off do employees get?"
    q_emb = model.encode([q], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=3)
    print(f"\nTest sorusu: {q}")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"- [{meta['source']}] (benzerlik {1 - dist:.2f}) {doc[:120]!r}")


if __name__ == "__main__":
    main()