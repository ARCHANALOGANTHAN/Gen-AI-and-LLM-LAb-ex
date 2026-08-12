from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ==========================================================
# LOAD CODE GENERATION MODEL
# ==========================================================

model_name = "Salesforce/codegen-350M-mono"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded successfully!")


# ==========================================================
# CODE GENERATION FUNCTION
# ==========================================================

def generate_code(prompt, max_new_tokens=80):
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


# ==========================================================
# 1. CODE GENERATION FROM NATURAL LANGUAGE
# ==========================================================

prompt1 = """
# Write a Python function to check if a number is prime
def is_prime(n):
"""

print("\n===== GENERATED FUNCTION =====")
print(generate_code(prompt1))


# ==========================================================
# 2. DEBUGGING A FAULTY CODE SNIPPET
# ==========================================================

buggy_code = """
# The following function should return the factorial of n,
# but has a bug. Fix it.

def factorial(n):
    result = 0
    for i in range(1, n + 1):
        result = result * i
    return result

# Corrected function:
def factorial_fixed(n):
"""

print("\n===== DEBUG SUGGESTION =====")
print(generate_code(buggy_code, max_new_tokens=60))