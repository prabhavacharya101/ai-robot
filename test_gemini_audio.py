import asyncio
import wave

import sounddevice as sd

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

client = genai.Client()


async def main():

    print("🤖 Connecting to Gemini Live...")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    async with client.aio.live.connect(
        model="gemini-3.1-flash-live-preview",
        config=config,
    ) as session:

        print("✅ Connected!")

        # Read the recorded WAV file
        with wave.open("voice.wav", "rb") as audio_file:

            sample_rate = audio_file.getframerate()

            audio_data = audio_file.readframes(
                audio_file.getnframes()
            )

        print(f"🎵 Input sample rate: {sample_rate} Hz")
        print("🎤 Sending audio...")

        # Send the recording in chunks
        chunk_size = 3200

        for i in range(0, len(audio_data), chunk_size):

            chunk = audio_data[i:i + chunk_size]

            await session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type="audio/pcm;rate=16000",
                )
            )

            await asyncio.sleep(0.1)

        # Tell Gemini the user stopped speaking
        await session.send_realtime_input(
            audio_stream_end=True
        )

        print("🛑 Audio finished.")
        print("\n🤖 Robo:")

        # Open speaker stream
        speaker = sd.RawOutputStream(
            samplerate=24000,
            channels=1,
            dtype="int16",
        )

        speaker.start()

        try:

            async for response in session.receive():

                if not response.server_content:
                    continue

                # What Gemini heard
                if response.server_content.input_transcription:

                    print(
                        "\n🎤 You said:",
                        response.server_content.input_transcription.text
                    )

                # Robo's text transcription
                if response.server_content.output_transcription:

                    print(
                        response.server_content.output_transcription.text,
                        end="",
                        flush=True
                    )

                # Robo's actual audio
                if response.server_content.model_turn:

                    for part in response.server_content.model_turn.parts:

                        if part.inline_data:

                            audio_chunk = part.inline_data.data

                            print(
                                "\n🔊 Audio chunk:",
                                len(audio_chunk),
                                "bytes"
                            )

                            speaker.write(audio_chunk)

                if response.server_content.turn_complete:

                    print("\n\n✅ Robo finished speaking.")
                    break

        finally:

            speaker.stop()
            speaker.close()


asyncio.run(main())