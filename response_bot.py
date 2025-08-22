from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained("./hotel_chatbot_gpt2")
model = GPT2LMHeadModel.from_pretrained("./hotel_chatbot_gpt2")

# Set device
device = torch.device("cpu")  # Ensuring the model runs on CPU
model.to(device)  # Moving the model to CPU



# Display welcome message
print("\n🌟 Welcome to the Hotel Chatbot! 🌟")

# List of questions to test
test_questions = [
    "What services does your hotel offer?",
    "Do you have free Wi-Fi in all rooms?",
    "What is the check-in and check-out time?",
    "Are pets allowed in the hotel?",
    "Do you provide airport shuttle services?",
    "Can I request a late checkout?"
]

# Run model generation for each question
for question in test_questions:
    input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)  # Ensures proper handling of padding

    # Generate response
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_length=100,
        num_return_sequences=1,  # Ensure only one response is generated
        do_sample=True,  # Introduce some randomness for natural responses
        top_k=50,  # Limit to more likely token selections
        top_p=0.95  # Nucleus sampling for more fluid responses
    )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-processing to remove repetition of the question
    if response_text.lower().startswith(question.lower()):
        response_text = response_text[len(question):].strip()

    print(f"🛎 Question: {question}")
    print(f"💬 Chatbot Response: {response_text}\n")