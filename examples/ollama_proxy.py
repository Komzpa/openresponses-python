from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openresponses import OpenResponsesRequest, MessageItem, OpenResponsesOutput
from openresponses.provider import OpenResponsesProvider

# Ollama typically runs on localhost:11434
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Not required but compliant info
)

app = FastAPI(title="Ollama Proxy")


@app.post("/v1/responses")
async def create_response(request: OpenResponsesRequest):
    messages = OpenResponsesProvider.map_request_to_messages(request)
    model = request.model or "llama3"

    if request.stream:
        return StreamingResponse(
            stream_ollama(model, messages), media_type="text/event-stream"
        )

    try:
        completion = await client.chat.completions.create(
            model=model, messages=messages, stream=False
        )
        # Check for deepseek-r1 reasoning in Ollama?
        # Often it comes as <think> tags in content or handled inside content.
        # We process simple content for now.
        return OpenResponsesOutput(
            id=completion.id,
            created=completion.created,
            model=model,
            output=[
                MessageItem(
                    role="assistant", content=completion.choices[0].message.content
                )
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_ollama(model: str, messages: list):
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, stream=True
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield OpenResponsesProvider.create_text_delta(content)
        yield OpenResponsesProvider.create_done_event()
    except Exception as e:
        yield OpenResponsesProvider.create_sse_event("error", {"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
