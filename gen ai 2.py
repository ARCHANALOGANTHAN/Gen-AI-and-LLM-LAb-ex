 from transformers import pipeline

# Load GPT-2 text generation model
generator = pipeline("text-generation", model="gpt2")


# 1. Zero-shot prompt
zero_shot_prompt = (
    "Classify the sentiment of this review as Positive or Negative: "
    "'The product quality is excellent!'\n"
    "Sentiment:"
)


# 2. Few-shot prompt
few_shot_prompt = """Review: 'I loved this movie, it was fantastic.'
Sentiment: Positive

Review: 'The service was slow and disappointing.'
Sentiment: Negative

Review: 'The product quality is excellent!'
Sentiment:"""


# 3. Chain-of-Thought prompt
cot_prompt = """Q: A shop had 15 apples. It sold 6 and then received 10 more.
How many apples now?

A: Let's think step by step.
15 - 6 = 9.
9 + 10 = 19.
The answer is 19.

Q: A library had 120 books. It lent out 45 and bought 30 new books.
How many books now?

A: Let's think step by step."""


# Generate responses
prompts = [
    ("Zero-shot", zero_shot_prompt),
    ("Few-shot", few_shot_prompt),
    ("Chain-of-Thought", cot_prompt)
]


for name, prompt in prompts:

    output = generator(
        prompt,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=False
    )

    print(f"=== {name} ===")
    print(output[0]["generated_text"])
    print()