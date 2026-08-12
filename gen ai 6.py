from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline


# ==========================================================
# 1. KNOWLEDGE BASE
# ==========================================================

documents = [
    "The Eiffel Tower is located in Paris, France and was completed in 1889.",
    "Retrieval-Augmented Generation combines document retrieval with text generation.",
    "Python is a popular high-level programming language used in AI development.",
    "Vector databases store embeddings and support fast similarity search."
]


# ==========================================================
# 2. EMBED DOCUMENTS
# ==========================================================

print("Loading embedding model...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embed_model.encode(
    documents,
    convert_to_numpy=True
)


# ==========================================================
# 3. BUILD FAISS INDEX
# ==========================================================

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(doc_embeddings.astype("float32"))

print("FAISS index created successfully.")


# ==========================================================
# 4. QUERY AND RETRIEVE TOP-2 RELEVANT DOCUMENTS
# ==========================================================

query = "What is RAG in AI?"

query_embedding = embed_model.encode(
    [query],
    convert_to_numpy=True
)

D, I = index.search(
    query_embedding.astype("float32"),
    k=2
)

retrieved_chunks = [
    documents[i]
    for i in I[0]
]


# ==========================================================
# 5. BUILD AUGMENTED PROMPT
# ==========================================================

context = " ".join(retrieved_chunks)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""


# ==========================================================
# 6. GENERATE ANSWER
# ==========================================================

print("Loading text generation model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

answer = generator(
    prompt,
    max_new_tokens=60,
    do_sample=False
)


# ==========================================================
# 7. DISPLAY RESULTS
# ==========================================================

print("\n===== RAG RESULTS =====")

print("\nRetrieved Context:")
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"{i}. {chunk}")

print("\nQuestion:")
print(query)

print("\nAnswer:")
print(answer[0]["generated_text"])