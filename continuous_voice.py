import asyncio

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()

client = genai.Client()

MODEL = "gemini-3.1-flash-live-preview"

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

OUTPUT_SAMPLE_RATE = 24000

# VAD settings
CHUNK_DURATION = 0.1          # 100 ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 0.8
MAX_RECORDING_SECONDS = 15


def get_volume(audio):
    """Calculate the volume of an audio chunk."""
    audio_float = audio.astype(np.float32)
    return np.sqrt(np.mean(audio_float ** 2))


def record_until_silence():
    """Wait for speech, then record until the user stops speaking."""

    print("🎤 Waiting for speech...")

    chunks = []
    speech_started = False
    silent_chunks = 0

    max_chunks = int(MAX_RECORDING_SECONDS / CHUNK_DURATION)
    required_silent_chunks = int(SILENCE_DURATION / CHUNK_DURATION)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=CHUNK_SAMPLES,
    ) as stream:

        for _ in range(max_chunks):

            audio, _ = stream.read(CHUNK_SAMPLES)

            audio = np.asarray(audio).reshape(-1)

            volume = get_volume(audio)

            if volume > SILENCE_THRESHOLD:

                if not speech_started:
                    print("🗣️ Speech detected!")

                speech_started = True
                silent_chunks = 0
                chunks.append(audio.copy())

            elif speech_started:

                chunks.append(audio.copy())
                silent_chunks += 1

                if silent_chunks >= required_silent_chunks:
                    print("🛑 Speech finished.")
                    break

    if not chunks:
        return None

    return np.concatenate(chunks).astype(np.int16)


async def main():

    print("🤖 Robo is starting...")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    async with client.aio.live.connect(
        model=MODEL,
        config=config,
    ) as session:

        print("✅ Connected to Gemini!")
        print("🦾 Robo is ready!\n")

        speaker = sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )

        speaker.start()

        try:

            while True:

                # Wait for the user to speak
                audio = record_until_silence()

                if audio is None:
                    continue

                print("🧠 Sending to Gemini...")

                audio_data = audio.tobytes()

                chunk_size = 3200

                for i in range(0, len(audio_data), chunk_size):

                    chunk = audio_data[i:i + chunk_size]

                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=chunk,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )

                    await asyncio.sleep(0.01)

                await session.send_realtime_input(
                    audio_stream_end=True
                )

                print("🤖 Robo:")

                async for response in session.receive():

                    if not response.server_content:
                        continue

                    server_content = response.server_content

                    # What Gemini heard
                    if server_content.input_transcription:

                        print(
                            "\n🎤 You:",
                            server_content.input_transcription.text
                        )

                    # Robo's text transcription
                    if server_content.output_transcription:

                        print(
                            server_content.output_transcription.text,
                            end="",
                            flush=True
                        )

                    # Robo's actual voice
                    if server_content.model_turn:

                        for part in server_content.model_turn.parts:

                            if part.inline_data:

                                audio_chunk = part.inline_data.data

                                speaker.write(audio_chunk)

                    if server_content.turn_complete:

                        print("\n")
                        break

        except KeyboardInterrupt:

            print("\n🛑 Robo shutting down...")

        finally:

            speaker.stop()
            speaker.close()


asyncio.run(main())