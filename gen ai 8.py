from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# Use GPU if CUDA is available
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("Using GPU")
else:
    pipe = pipe.to("cpu")
    print("Using CPU")

# Text prompt
prompt = "A futuristic city skyline at sunset, digital art, highly detailed"

# Generate image
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# Save image
image.save("generated_city.png")

print("Image generated and saved as generated_city.png")