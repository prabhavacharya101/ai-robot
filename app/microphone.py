import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
RECORD_SECONDS = 5


def record_audio(filename="voice.wav"):

    print("\n🎤 Listening... Speak now!")

    recording = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    sf.write(
        filename,
        recording,
        SAMPLE_RATE
    )

    print("✅ Recording finished.")

    return filename