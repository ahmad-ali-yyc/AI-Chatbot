import os
import sqlite3
import torch
from transformers import pipeline
from PIL import Image

DB_FILE = "chatbot.db"
IMAGE_DIRECTORY = "ALL_IMAGES"  # Update directory

# Load AI Model for Image Classification & Captioning (Only Once)
print("Loading AI models...")
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
image_captioner = pipeline("image-captioning", model="nlpconnect/vit-gpt2-image-captioning")

def analyze_image(image_path):
    """Analyze image to get a category and generate a relevant comment."""
    try:
        image = Image.open(image_path).convert("RGB")

        # Categorize the image
        category_result = image_classifier(image)
        category = category_result[0]["label"] if category_result else "Unknown"

        # Generate a descriptive comment
        caption_result = image_captioner(image)
        comment = caption_result[0]["generated_text"] if caption_result else "No description available"

        return category, comment
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return "Unknown", "No description available"

def process_and_store_images():
    """Process all images in the directory and store them in the database."""
    if not os.path.exists(IMAGE_DIRECTORY):
        print(f"Directory '{IMAGE_DIRECTORY}' not found. Skipping image processing.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for file_name in os.listdir(IMAGE_DIRECTORY):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(IMAGE_DIRECTORY, file_name)

            # Check if this image is already in the database
            cursor.execute("SELECT name FROM media_directory WHERE name = ?", (file_name,))
            if cursor.fetchone():
                print(f"{file_name} already exists in the database. Skipping.")
                continue  # Skip if already processed

            # Analyze image
            category, comment = analyze_image(image_path)

            # Store in database with error handling
            try:
                cursor.execute("INSERT INTO media_directory (name, category, comments) VALUES (?, ?, ?)", 
                               (file_name, category, comment))
                print(f"Stored {file_name}: {category} - {comment}")
            except sqlite3.Error as db_error:
                print(f"Database error while inserting {file_name}: {db_error}")

    conn.commit()
    conn.close()
    print("Image processing completed and data stored.")

if __name__ == "__main__":
    process_and_store_images()
