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


while True:
    # Ask the user for input
    user_question = input("Ask me(or type 'exit' to quit): ")

    # Exit condition
    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    # Tokenize and generate response
    input_ids = tokenizer.encode(user_question, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)  # Ensures proper handling of padding

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
    if response_text.lower().startswith(user_question.lower()):
        response_text = response_text[len(user_question):].strip()

    print(f"💬 Chatbot Response: {response_text}\n")