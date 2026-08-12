from transformers import pipeline

# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================

sentiment_analyzer = pipeline(
    "sentiment-analysis"
)

reviews = [
    "The new smartphone has an amazing camera and battery life!",
    "The delivery was late and the packaging was damaged."
]

print("===== SENTIMENT ANALYSIS =====")

for review in reviews:
    result = sentiment_analyzer(review)[0]

    print(
        f"Review: {review}\n"
        f"-> {result['label']} ({round(result['score'], 3)})\n"
    )


# ==========================================================
# DOCUMENT CLASSIFICATION - ZERO SHOT
# ==========================================================

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

document = (
    "The central bank raised interest rates "
    "to control rising inflation."
)

candidate_labels = [
    "Politics",
    "Economy",
    "Sports",
    "Technology"
]

classification = classifier(
    document,
    candidate_labels
)

print("===== DOCUMENT CLASSIFICATION =====")
print("Document:", document)
print()

for label, score in zip(
    classification["labels"],
    classification["scores"]
):
    print(f"{label}: {round(score, 3)}")