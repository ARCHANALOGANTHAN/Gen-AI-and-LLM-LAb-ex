from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score


# ==========================================================
# 1. LOAD DOMAIN-SPECIFIC DATASET
# ==========================================================

print("Loading IMDB dataset...")

dataset = load_dataset("imdb")

# Select a smaller dataset for faster training
small_train = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(2000))
)

small_test = (
    dataset["test"]
    .shuffle(seed=42)
    .select(range(500))
)

print("Training samples:", len(small_train))
print("Testing samples:", len(small_test))


# ==========================================================
# 2. TOKENIZATION
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_ds = small_train.map(
    tokenize,
    batched=True
)

test_ds = small_test.map(
    tokenize,
    batched=True
)

# Remove unnecessary text column
train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

# Set PyTorch format
train_ds.set_format("torch")
test_ds.set_format("torch")


# ==========================================================
# 3. LOAD PRE-TRAINED MODEL
# ==========================================================

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# ==========================================================
# 4. TRAINING ARGUMENTS
# ==========================================================

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)


# ==========================================================
# 5. EVALUATION METRICS
# ==========================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    preds = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        preds
    )

    return {
        "accuracy": accuracy
    }


# ==========================================================
# 6. TRAIN MODEL
# ==========================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

print("\nStarting training...")

trainer.train()


# ==========================================================
# 7. EVALUATE MODEL
# ==========================================================

metrics = trainer.evaluate()

print("\n===== EVALUATION METRICS =====")
print(metrics)


# ==========================================================
# 8. SAVE FINE-TUNED MODEL
# ==========================================================

save_path = "./fine_tuned_distilbert_imdb"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("\nFine-tuned model saved to:")
print(save_path)