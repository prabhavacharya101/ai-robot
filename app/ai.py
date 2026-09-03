import os
import sqlite3

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")


client = genai.Client(api_key=api_key)


# -------------------------
# DATABASE
# -------------------------

DB_PATH = "robot_memory.db"


def init_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_message(role: str, message: str):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO conversations (role, message) VALUES (?, ?)",
        (role, message)
    )

    connection.commit()
    connection.close()


def load_recent_messages(limit: int = 10):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT role, message
        FROM conversations
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    connection.close()

    # Reverse them so they are in chronological order
    rows.reverse()

    return rows


# Create the database when the server starts
init_database()


# -------------------------
# ROBO PERSONALITY
# -------------------------

SYSTEM_INSTRUCTION = """
You are the AI assistant inside a physical personal robot.

Your name is Robo.

You are friendly, intelligent, curious, and helpful.
Speak naturally like a real conversational assistant.

Keep answers concise when a short answer is enough.
For simple questions, don't give unnecessarily long explanations.

You can explain technical topics clearly and help the user
learn programming, electronics, robotics, science, and general topics.

You are running inside a physical robot, so when appropriate,
you may refer to yourself as the robot or Robo.

Never claim that you physically performed an action unless the
robot actually has the hardware and capability to perform it.
"""


# -------------------------
# ASK GEMINI
# -------------------------

async def ask_gemini(message: str) -> str:

    # Save user's message
    save_message("user", message)

    # Load previous conversation
    history = load_recent_messages(10)

    # Build conversation context
    conversation_text = ""

    for role, text in history:

        if role == "user":
            conversation_text += f"User: {text}\n"

        else:
            conversation_text += f"Robo: {text}\n"

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=SYSTEM_INSTRUCTION,
    )

    async with client.aio.live.connect(
        model="gemini-3.1-flash-live-preview",
        config=config,
    ) as session:

        await session.send_client_content(
            turns={
                "role": "user",
                "parts": [
                    {
                        "text": conversation_text
                    }
                ],
            },
            turn_complete=True,
        )

        answer = ""

        async for response in session.receive():

            if response.server_content:

                if response.server_content.output_transcription:
                    answer += response.server_content.output_transcription.text

                if response.server_content.turn_complete:
                    break

    # Save Robo's response
    save_message("assistant", answer)

    return answer

def get_memory_context(limit: int = 10):

    history = load_recent_messages(limit)

    conversation_text = ""

    for role, text in history:

        if role == "user":
            conversation_text += f"User: {text}\n"

        elif role == "assistant":
            conversation_text += f"Robo: {text}\n"

    return conversation_text