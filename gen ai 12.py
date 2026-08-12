import gradio as gr
import evaluate
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ==========================================================
# 1. LOAD SUMMARIZATION MODEL
# ==========================================================

model_name = "facebook/bart-large-cnn"

print("Loading summarization model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")


# ==========================================================
# 2. SUMMARIZATION FUNCTION
# ==========================================================

def summarize_text(input_text):

    if not input_text.strip():
        return "Please enter some text to summarize."

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        summary_ids = model.generate(
            **inputs,
            max_new_tokens=45,
            min_new_tokens=15,
            num_beams=4,
            do_sample=False
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# ==========================================================
# 3. BUILD GRADIO APPLICATION
# ==========================================================

demo = gr.Interface(
    fn=summarize_text,

    inputs=gr.Textbox(
        lines=8,
        label="Enter text to summarize",
        placeholder="Enter your text here..."
    ),

    outputs=gr.Textbox(
        label="Generated Summary"
    ),

    title="GenAI Text Summarizer",

    description=(
        "A Generative AI text summarization application "
        "built with Gradio."
    )
)


# ==========================================================
# 4. LAUNCH APPLICATION
# ==========================================================

demo.launch(share=True)


# ==========================================================
# 5. EVALUATE GENERATED OUTPUT USING ROUGE
# ==========================================================

rouge = evaluate.load("rouge")

generated_summaries = [
    "AI models generate new content such as text and images."
]

reference_summaries = [
    "Generative AI models are capable of producing new content "
    "including text and images."
]

scores = rouge.compute(
    predictions=generated_summaries,
    references=reference_summaries
)

print("\n===== ROUGE EVALUATION SCORES =====")

for metric, score in scores.items():
    print(f"{metric}: {score:.4f}")