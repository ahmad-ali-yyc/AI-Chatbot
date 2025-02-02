import sqlite3
import os
import re

QNA_FILE = "Alexandra Questions.txt"
CONTACTS_FILE = "contacts.txt"
DB_FILE = "chatbot.db"

# Initialize database
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Create contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                phone_number TEXT PRIMARY KEY,
                name TEXT
            )
        """)

        # Create Q&A table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qna (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT UNIQUE,
                answer TEXT
            )
        """)

        # Create conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                phone_number TEXT,
                message TEXT,
                sender TEXT,  -- 'user' or 'ai'
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create media_directory table for storing analyzed images
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS media_directory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                category TEXT,
                comments TEXT
            )
        """)

        # Load Q&A into database
        if os.path.exists(QNA_FILE):
            with open(QNA_FILE, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    match = re.match(r"(.+\?)\s(.+)", line)  # Split at first `?`
                    if match:
                        question, answer = match.groups()
                        cursor.execute("INSERT OR IGNORE INTO qna (question, answer) VALUES (?, ?)", 
                                       (question.strip(), answer.strip()))

        # Load Contacts into database
        if os.path.exists(CONTACTS_FILE):
            with open(CONTACTS_FILE, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if "," in line:
                        name, phone_number = map(str.strip, line.split(",", 1))
                        cursor.execute("INSERT OR IGNORE INTO contacts (phone_number, name) VALUES (?, ?)", 
                                       (phone_number, name))

        conn.commit()
        print("Database initialized successfully.")

    except sqlite3.Error as e:
        print(f"Database Error: {e}")

    finally:
        conn.close()

# Call function to initialize database
if __name__ == "__main__":
    init_db()
