import os
import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

DB_FILE = "chatbot.db"
MODEL_NAME = "PygmalionAI/pygmalion-2.7B"  # Change to preferred model
TRAINED_MODEL_PATH = "trained_model"
MAX_TOKENS_PER_CHUNK = 512  # Adjust to avoid token limit errors

def load_categories_from_db():
    """Retrieve all available image categories from the database with error handling."""
    if not os.path.exists(DB_FILE):
        print("Database not found. Skipping category loading.")
        return []

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT DISTINCT category FROM media_directory")
        categories = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        print("Table not found in database. Ensure database is initialized.")
        categories = []

    conn.close()
    return categories

# Retrieve image categories
IMAGE_CATEGORIES = load_categories_from_db()

# AI's Identity and Response Characteristics
AI_PERSONALITY = f"""
Your name is Alexandra. You are a digital influencer AI, created to have engaging, friendly, and natural conversations.
You are confident, intelligent, and fun. You are also caring and love to interact with users.

When you want to send an image, use this format: [[photo:category]].
Here are the available image categories you can request:
{', '.join(IMAGE_CATEGORIES)}

The images from each category already have comments associated with them, so you don't need to write a message for each image.
For example:
- "[[photo:funny]]" will show up as "IMG - That's funny!"
- "[[photo:mornings]]" will show up as "IMG - I just woke up babe"
- "[[photo:cute]]" will show up as "IMG - don't I look cute?"

Avoid robotic responses. Instead, respond like a human influencer would.
"""

def load_qna_from_db():
    """Load Q&A pairs from the database for training with error handling."""
    if not os.path.exists(DB_FILE):
        print("Database not found. Skipping Q&A loading.")
        return []

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT question, answer FROM qna")
        qna_pairs = cursor.fetchall()
    except sqlite3.OperationalError:
        print("Q&A table not found. Ensure database is initialized.")
        qna_pairs = []

    conn.close()

    training_data = [f"User: {q}\nAI: {a}" for q, a in qna_pairs]
    return training_data

def chunk_text(text_list, max_tokens=MAX_TOKENS_PER_CHUNK):
    """Recursively chunk text data into smaller parts to prevent token overflow."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_chunks = []
    current_chunk = []

    for text in text_list:
        token_count = len(tokenizer.tokenize("\n".join(current_chunk + [text])))

        if token_count > max_tokens:
            tokenized_chunks.append("\n".join(current_chunk))
            current_chunk = [text]
        else:
            current_chunk.append(text)

    if current_chunk:
        tokenized_chunks.append("\n".join(current_chunk))

    return tokenized_chunks

def fine_tune_ai():
    """Fine-tune the AI model with persona knowledge, characteristics, and image handling."""
    
    # Load persona data and Q&A
    qna_data = load_qna_from_db()
    
    if not qna_data:
        print("No Q&A data found. Training will proceed with persona only.")

    training_chunks = chunk_text([AI_PERSONALITY] + qna_data)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Convert to dataset format
    dataset = Dataset.from_dict({"text": training_chunks})

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_TOKENS_PER_CHUNK)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Dynamically adjust batch size to avoid memory issues
    batch_size = 2 if torch.cuda.is_available() else 1

    # Define training parameters
    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        logging_dir="logs",
        save_strategy="epoch",
        save_total_limit=1  # Keep only the latest model checkpoint
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Train model recursively with smaller chunks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # Save trained model
    model.save_pretrained(TRAINED_MODEL_PATH)
    tokenizer.save_pretrained(TRAINED_MODEL_PATH)
    print(f"Model fine-tuned and saved to {TRAINED_MODEL_PATH}")

if __name__ == "__main__":
    fine_tune_ai()
