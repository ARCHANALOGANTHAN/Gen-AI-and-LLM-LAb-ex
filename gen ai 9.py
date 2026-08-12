from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

from PIL import Image
import requests
from io import BytesIO


# ==========================================================
# LOAD IMAGE
# ==========================================================

image_url = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"

response = requests.get(image_url)
response.raise_for_status()

raw_image = Image.open(
    BytesIO(response.content)
).convert("RGB")

print("Image loaded successfully!")


# ==========================================================
# IMAGE CAPTIONING
# ==========================================================

print("\nLoading image captioning model...")

cap_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

cap_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Prepare image
inputs = cap_processor(
    raw_image,
    return_tensors="pt"
)

# Generate caption
caption_ids = cap_model.generate(
    **inputs,
    max_new_tokens=30
)

caption = cap_processor.decode(
    caption_ids[0],
    skip_special_tokens=True
)

print("\n===== IMAGE CAPTIONING =====")
print("Generated Caption:", caption)


# ==========================================================
# VISUAL QUESTION ANSWERING
# ==========================================================

print("\nLoading Visual Question Answering model...")

vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

# Ask a question about the image
question = "What animal is in the picture?"

vqa_inputs = vqa_processor(
    raw_image,
    question,
    return_tensors="pt"
)

# Generate answer
answer_ids = vqa_model.generate(
    **vqa_inputs,
    max_new_tokens=10
)

answer = vqa_processor.decode(
    answer_ids[0],
    skip_special_tokens=True
)

print("\n===== VISUAL QUESTION ANSWERING =====")
print("Question:", question)
print("Answer:", answer)