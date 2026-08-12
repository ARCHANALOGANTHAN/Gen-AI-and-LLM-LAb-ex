from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Store conversation history
chat_history_ids = None

print("Chatbot ready! Type 'quit' to exit.")

# Chat loop
for step in range(5):
    user_input = input(">> User: ")

    if user_input.lower() == "quit":
        print("Chatbot ended.")
        break

    # Encode user input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Add new input to conversation history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat(
            [chat_history_ids, new_input_ids],
            dim=-1
        )
    else:
        bot_input_ids = new_input_ids

    # Generate chatbot response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )

    # Decode only the new response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print(f"Bot: {response}")