import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, AsyncGenerator

from pyaudio import PyAudio, paInt16
import audioop
from openai import AsyncOpenAI


async def stream_mic(audio: PyAudio, threshold: int = 500, chunk: int = 512, speech_end_delay: float = 2.5) -> AsyncGenerator[bytes, Any]:
    info = audio.get_default_input_device_info()
    rate = int(info["defaultSampleRate"])
    stream = audio.open(
            format=paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
    )
    
    frames = []
    state = "not_speaking"
    ts = 0.0
    print("Listening...")
    while True:
        t2 = time.time()
        data = stream.read(chunk, exception_on_overflow=False)
        rms = audioop.rms(data, 2)  # sample width of 2 for format paInt16
        
        if state == "not_speaking":
            if rms >= threshold:
                state = "speaking"
                print("Speaking...")
                frames.append(data)
                ts = t2
        elif state == "speaking":
            if rms < threshold and t2 - ts > speech_end_delay:
                state = "not_speaking"
                print("...done")
                yield b"".join(frames)
                frames = []
                ts = 0.0
            else:
                if rms > threshold:
                    ts = t2  # reset speaking timer
                frames.append(data)
    
    # TODO add a way to stop the stream & generator
    stream.stop_stream()
    stream.close()
                
            
async def transcribe(client: AsyncOpenAI, audio: PyAudio, data: bytes, model="whisper-1") -> str:
    info = audio.get_default_input_device_info()
    rate = int(info["defaultSampleRate"])
    with TemporaryDirectory() as tmpdir:
        fp = str(Path(tmpdir) / "mic.wav")
        wav = wave.open(fp, "wb")
        wav.setnchannels(1)
        wav.setsampwidth(audio.get_sample_size(paInt16))
        wav.setframerate(rate)
        wav.writeframes(data)
        wav.close()

        with open(fp, "rb") as audio_file:
            text = await client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text"
            )
            return text
