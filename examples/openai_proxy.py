import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openresponses import OpenResponsesRequest, OpenResponsesOutput, MessageItem
from openresponses.provider import OpenResponsesProvider

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="OpenAI Proxy")


@app.post("/v1/responses")
async def create_response(request: OpenResponsesRequest):
    messages = OpenResponsesProvider.map_request_to_messages(request)

    if request.stream:
        return StreamingResponse(
            stream_openai(request.model, messages), media_type="text/event-stream"
        )

    try:
        completion = await client.chat.completions.create(
            model=request.model or "gpt-4o", messages=messages, stream=False
        )

        output_items = [
            MessageItem(role="assistant", content=completion.choices[0].message.content)
        ]

        return OpenResponsesOutput(
            id=completion.id,
            created=completion.created,
            model=request.model,
            output=output_items,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_openai(model: str, messages: list):
    stream = await client.chat.completions.create(
        model=model or "gpt-4o", messages=messages, stream=True
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield OpenResponsesProvider.create_text_delta(content)
    yield OpenResponsesProvider.create_done_event()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
