import pandas as pd
import json
from transformers import GPT2Tokenizer

# Load CSV file
csv_file = "Hotel_Dataset.csv"  # Corrected filename
df = pd.read_csv(csv_file)

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Assign a padding token (GPT-2 does not have one by default)
tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence token as padding

# Tokenize dataset
def tokenize_function(question, answer):
    tokenized = tokenizer(
        question + " " + answer,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

# Create tokenized dataset
tokenized_data = [tokenize_function(q, a) for q, a in zip(df["Question"], df["Answer"])]

# Convert to DataFrame and save as CSV
tokenized_df = pd.DataFrame(tokenized_data)
tokenized_df.to_csv("tokenized_hotel_chatbot.csv", index=False)

# Also save as JSON for easier compatibility
with open("tokenized_hotel_chatbot.json", "w") as f:
    json.dump(tokenized_data, f, indent=4)

print("Tokenization complete! Saved as 'tokenized_hotel_chatbot.csv' and 'tokenized_hotel_chatbot.json'.")