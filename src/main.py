import asyncio

import pyaudio
from openai import AsyncOpenAI

from opencharacter.voice.tts import say
from opencharacter.voice.stt import stream_mic, transcribe


async def main():
    audio = pyaudio.PyAudio()
    client = AsyncOpenAI()
    assistant = await client.beta.assistants.create(
        instructions="You are a sassy, witty, and vulgar girl that always talks her mind not matter how rude. You are a bit of a smartass. Do not end every statement with a question. Do not be polite. Occasionally inject swear words into your response. Sometimes be disagreeable and adversarial.",
        name="Lily",
        model="gpt-4o"
    )
    chat_thread = await client.beta.threads.create()
    print(f"assistant: {assistant}")
    print(f"chat_thread: {chat_thread}")
    async for speech in stream_mic(audio, speech_end_delay=5):
        speech_transcribed = await transcribe(client, audio, speech)
        print(f"got speech:\n{speech_transcribed}")
        print("thinking...")
        
        message = await client.beta.threads.messages.create(
            chat_thread.id,
            role="user",
            content=speech_transcribed
        )
        
        run = await client.beta.threads.runs.create(
            assistant_id=assistant.id,
            thread_id=chat_thread.id
        )
        
        while run.status in ["queued", "in_progress", "cancelling"]:
            print(f"waiting for run [{run.id}] to finish: {run.status}")
            await asyncio.sleep(0.1)
            run = await client.beta.threads.runs.retrieve(
                thread_id=chat_thread.id,
                run_id=run.id
            )
            
        msgs = await client.beta.threads.messages.list(
            chat_thread.id,
            run_id=run.id,
            order="desc",
            limit=1
        )
        async for msg in msgs:
            message = msg

            print("done thinking")
            for textcontentblock in message.content:
                print(f"talking: {textcontentblock.text.value}")
                await say(client, audio, textcontentblock.text.value, model="tts-1-hd", voice="sage")
            print("done talking\n")


if __name__ == "__main__":
    asyncio.run(main())
