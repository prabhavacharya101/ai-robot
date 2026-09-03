from fastapi import FastAPI, WebSocket

from app.ai import ask_gemini


app = FastAPI(title="AI Robot Backend")


@app.get("/")
def root():
    return {
        "robot": "AI Assistant",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat")
async def chat(message: dict):
    answer = await ask_gemini(message["message"])

    return {
        "response": answer
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    print("🔌 Robot connected!")

    try:

        while True:

            message = await websocket.receive_text()

            print("📡 ESP32:", message)

            await websocket.send_text(
                f"Robo received: {message}"
            )

    except Exception:

        print("🔌 Robot disconnected.")