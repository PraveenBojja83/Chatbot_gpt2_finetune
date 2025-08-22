from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Tokenizer loaded successfully!")

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("GPT-2 model loaded successfully!")

# Load dataset
dataset = load_dataset("json", data_files="hotel_chatbot_data.json")
print(f"Dataset loaded successfully! Keys available: {dataset.keys()}")

# Print sample data
print(f"Sample dataset entry: {dataset['train'][0]}")