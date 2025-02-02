from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random
import sqlite3
import torch
import re
from twilio.rest import Client
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

#Load secrets.env
load_dotenv() 
# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TWILIO_WHATSAPP_NUMBER = os.getenv("WHATSAPP_PHONE_NUMBER")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load AI Model
TRAINED_MODEL_PATH = "trained_model"
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(TRAINED_MODEL_PATH)

# Image Base URL or Directory
IMAGE_BASE_URL = "/ALL_IMAGES/"

# Database file
DB_FILE = "chatbot.db"

def get_random_image(category: str):
    """Fetch a random image and its comments from the database based on category."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT name, comments FROM media_directory WHERE category = ? ORDER BY RANDOM() LIMIT 1", (category,))
    result = cursor.fetchone()
    conn.close()

    if result:
        image_url = f"{IMAGE_BASE_URL}{result[0]}"  # Construct full image URL
        return {"image": image_url, "comments": result[1]}  # Return image & comments
    else:
        return {"image": None, "comments": "No images found for this category."}

def generate_response(user_input: str) -> dict:
    """Generate AI response while ensuring correct image handling."""
    input_text = f"User: {user_input}\nAI:"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )

    ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Extract category from AI response
    category_match = re.search(r"\[\[photo:(.*?)\]\]", ai_text)
    if category_match:
        category = category_match.group(1).strip()
        image_data = get_random_image(category)  # Fetch image & comments
        ai_text = ai_text.replace(f"[[photo:{category}]]", "").strip()
        return {"response": ai_text, "media": image_data["image"], "comments": image_data["comments"]}

    return {"response": ai_text}

@app.post("/chat")
async def chat(request: Request):
    """Handles chat messages from the web UI or API requests."""
    app.mount("/ALL_IMAGES", StaticFiles(directory="all_images"), name="all_images")

    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return JSONResponse(content={"response": "Please enter a message."})

    ai_result = generate_response(user_message)

    return JSONResponse(content={"response": ai_result["response"], "media": ai_result.get("media"), "comments": ai_result.get("comments")})

@app.post("/twilio_webhook")
async def twilio_webhook(
    Body: str = Form(...),
    From: str = Form(...)
):
    """Handles incoming messages from Twilio SMS & WhatsApp."""
    app.mount("/ALL_IMAGES", StaticFiles(directory="all_images"), name="all_images")
    
    user_message = Body.strip()
    user_phone = From.strip()

    if not user_message:
        return JSONResponse(content={"response": "No message received."})

    is_whatsapp = user_phone.startswith("whatsapp:")

    # Get user name & chat history
    user_name = get_user_name(user_phone)
    past_history = get_conversation_history(user_phone)

    # Generate AI response
    ai_result = generate_response(user_message)

    # Save conversation in DB
    save_conversation(user_phone, user_message, "user")
    save_conversation(user_phone, ai_result["response"], "ai")

    # Determine sender number (WhatsApp or SMS)
    sender_number = TWILIO_WHATSAPP_NUMBER if is_whatsapp else TWILIO_PHONE_NUMBER

    # Send AI response text
    client.messages.create(
        body=ai_result["response"],
        from_=sender_number,
        to=user_phone
    )

    # Send media if available
    if ai_result.get("media"):
        client.messages.create(
            media_url=[ai_result["media"]],
            from_=sender_number,
            to=user_phone
        )

    return JSONResponse(content={"response": ai_result["response"], "media": ai_result.get("media")})

def get_user_name(phone_number: str) -> str:
    """Retrieve user's name from database based on phone number."""
    if phone_number.startswith("whatsapp:"):
        phone_number = phone_number.replace("whatsapp:", "")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM contacts WHERE phone_number = ?", (phone_number,))
    result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else "Unknown User"

def get_conversation_history(phone_number: str) -> str:
    """Retrieve last 5 messages for context."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT message, sender FROM conversations 
        WHERE phone_number = ? 
        ORDER BY timestamp DESC 
        LIMIT 5
    """, (phone_number,))
    
    history = cursor.fetchall()
    conn.close()
    
    return "\n".join([f"{msg[1].capitalize()}: {msg[0]}" for msg in reversed(history)])

def save_conversation(phone_number: str, message: str, sender: str):
    """Save conversation history into the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO conversations (phone_number, message, sender) 
        VALUES (?, ?, ?)
    """, (phone_number, message, sender))
    
    conn.commit()
    conn.close()
