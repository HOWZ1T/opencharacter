from pyaudio import PyAudio, paInt16

from openai import AsyncOpenAI


async def say(client: AsyncOpenAI, audio: PyAudio, msg: str, model: str = "tts-1", voice: str = "sage", response_format: str = "pcm") -> None:
    player_stream = audio.open(format=paInt16, channels=1, rate=24000, output=True)

    async with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        response_format=response_format,
        input=msg
    ) as stream:
        async for chunk in stream.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)
    
    player_stream.stop_stream()
    player_stream.close()
