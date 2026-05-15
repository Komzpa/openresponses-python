import asyncio
from openresponses.client import AsyncOpenResponsesClient

# Assuming one of the examples is running on port 8001 (OpenRouter), 8002 (OpenAI), etc.
URL = "http://localhost:8001"


async def main():
    client = AsyncOpenResponsesClient(base_url=URL)

    print(f"\n{'=' * 10} Standard Request {'=' * 10}")
    try:
        response = await client.create(
            model="deepseek/deepseek-r1", input="Explain the Open Responses standard."
        )
        for item in response.output:
            if item.type == "reasoning":
                print(f"🧠 [Thinking]: {item.content[:50]}...")
            elif item.type == "message":
                print(f"🤖 [Answer]: {item.content[:50]}...")
    except Exception as e:
        print(f"Error: {e}")

    print(f"\n{'=' * 10} Streaming Request {'=' * 10}")
    try:
        stream = await client.create(
            model="deepseek/deepseek-r1", input="Stream this explanation.", stream=True
        )
        async for event in stream:
            if event.event == "response.reasoning.delta":
                print(f"🧠 {event.data['delta']}", end="", flush=True)
            elif event.event == "response.text.delta":
                print(f"🤖 {event.data['delta']}", end="", flush=True)
        print("\n✅ Done")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
