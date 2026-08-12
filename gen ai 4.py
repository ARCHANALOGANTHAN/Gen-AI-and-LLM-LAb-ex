from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch

# ==========================================================
# TEXT SUMMARIZATION
# ==========================================================

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

article = """
Generative AI refers to a class of artificial intelligence models
capable of producing new content such as text, images, audio, and
video. Large Language Models (LLMs) such as GPT and LLaMA are trained
on massive text corpora and can perform a wide range of natural
language tasks including translation, summarization, and question
answering. These models are increasingly being deployed in industry
applications ranging from customer support to software development,
transforming how humans interact with machines.
"""

# Tokenize article
inputs = tokenizer(
    article,
    return_tensors="pt",
    truncation=True,
    max_length=1024
)

# Generate summary
with torch.no_grad():
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=45,
        min_new_tokens=20,
        num_beams=4,
        do_sample=False
    )

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("===== TEXT SUMMARIZATION =====")
print("Summary:")
print(summary)


# ==========================================================
# QUESTION ANSWERING
# ==========================================================

qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

question = "What are Large Language Models trained on?"

answer = qa(
    question=question,
    context=article
)

print("\n===== QUESTION ANSWERING =====")
print("Question:", question)
print("Answer:", answer["answer"])
print("Confidence:", round(answer["score"], 3))