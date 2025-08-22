## Install dependencies
##pip install transformers torch datasets sklearn
##pip install scikit-learn

import torch
import json
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from sklearn.model_selection import train_test_split  # ✅ Import to fix train_test_split issue

# Fine-tune GPT-2 for Hotel Chatbot using Hugging Face Transformers
# ✅ Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ✅ Assign padding token (GPT-2 does not have one by default)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Move model to selected device
model.to(device)

# ✅ Load dataset from JSON file
with open("tokenized_hotel_chatbot.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ Convert raw JSON data into Hugging Face Dataset
dataset = Dataset.from_list(data)

# ✅ Split dataset for training & testing
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# ✅ Ensure dataset includes labels for loss computation
def format_dataset(example):
    if "input_ids" in example:
        example["labels"] = example["input_ids"].copy()
    return example

# ✅ Apply formatting to datasets
train_dataset = train_dataset.map(format_dataset)
eval_dataset = eval_dataset.map(format_dataset)

# ✅ Verify dataset split
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(eval_dataset)}")
print(f"First training sample: {train_dataset[0]}")

# ✅ Define training arguments optimized for CPU
training_args = TrainingArguments(
    output_dir="./gpt2_hotel_chatbot",
    per_device_train_batch_size=2,  # Adjust for CPU efficiency
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=200,  # Save model checkpoints frequently
    logging_steps=50,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    report_to="none"  # Disable reporting logs
)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ✅ Start Fine-Tuning
print("Starting fine-tuning process...")
trainer.train()

# ✅ Save Fine-Tuned Model
model.save_pretrained("./hotel_chatbot_gpt2")
tokenizer.save_pretrained("./hotel_chatbot_gpt2")

print("Fine-tuning complete! Model saved successfully.")