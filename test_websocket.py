import asyncio
import websockets


async def test():

    uri = "ws://127.0.0.1:8000/ws"

    async with websockets.connect(uri) as websocket:

        print("✅ Connected to Robo Backend!")

        await websocket.send("Hello Robo!")

        response = await websocket.recv()

        print("🤖 Robo:", response)


asyncio.run(test())