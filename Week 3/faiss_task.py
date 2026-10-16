import faiss   
from sentence_transformers import SentenceTransformer

docs = [
    "Refunds are processed within 5 business days.",
    "Our office is closed on public holidays.",
    "Reset your password from the account settings page.",
    "Shipping to India takes 7-10 days.",
    "LLM",
    "llm model for ai"
]

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(docs, normalize_embeddings=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

THRESHOLD = 0.3
q = model.encode(["Groq ai"], normalize_embeddings=True).astype("float32")
scores, ids = index.search(q, k=32)
for score, i in zip(scores[0], ids[0]):
    if i == -1 or score < THRESHOLD:            # skip empty slots
        continue
    print(f">>>>>>>>>>>>>>> \n{score:.3f}  {docs[i]}")