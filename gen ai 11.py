from transformers import pipeline
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import torch


# ==========================================================
# TOPIC
# ==========================================================

topic = "The benefits of renewable energy"


# ==========================================================
# 1. TEXT GENERATION
# ==========================================================

print("Generating text...")

text_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

text_prompt = (
    f"Write a short, engaging paragraph about: {topic}"
)

generated_text = text_generator(
    text_prompt,
    max_new_tokens=80,
    do_sample=False
)[0]["generated_text"]

print("\n===== GENERATED TEXT =====")
print(generated_text)


# ==========================================================
# 2. IMAGE GENERATION
# ==========================================================

print("\nLoading Stable Diffusion...")

image_prompt = (
    f"An illustration representing {topic}, "
    "digital art, highly detailed"
)

# Check whether CUDA/GPU is available
if torch.cuda.is_available():

    print("Using NVIDIA GPU...")

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )

    sd_pipe = sd_pipe.to("cuda")

else:

    print("CUDA GPU not available. Using CPU...")

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )

    sd_pipe = sd_pipe.to("cpu")


# Generate image
image = sd_pipe(
    image_prompt,
    num_inference_steps=25
).images[0]

# Save image
image.save("content_image.png")

print("Image saved as content_image.png")


# ==========================================================
# 3. TEXT-TO-SPEECH
# ==========================================================

print("\nGenerating audio...")

tts = gTTS(
    text=generated_text,
    lang="en"
)

tts.save("content_audio.mp3")

print("Audio saved as content_audio.mp3")


# ==========================================================
# FINAL RESULT
# ==========================================================

print("\n===== CONTENT GENERATION COMPLETED =====")
print("Text  : generated successfully")
print("Image : content_image.png")
print("Audio : content_audio.mp3")