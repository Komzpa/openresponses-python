from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openresponses import OpenResponsesRequest, MessageItem, OpenResponsesOutput
from openresponses.provider import OpenResponsesProvider

# LM Studio typically runs on localhost:1234
client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

app = FastAPI(title="LM Studio Proxy")


@app.post("/v1/responses")
async def create_response(request: OpenResponsesRequest):
    messages = OpenResponsesProvider.map_request_to_messages(request)
    # LM Studio usually ignores model name if only one model is loaded,
    # but we pass it anyway.
    model = request.model or "local-model"

    if request.stream:
        return StreamingResponse(
            stream_lmstudio(model, messages), media_type="text/event-stream"
        )

    try:
        completion = await client.chat.completions.create(
            model=model, messages=messages, stream=False
        )
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


async def stream_lmstudio(model: str, messages: list):
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

    uvicorn.run(app, host="0.0.0.0", port=8004)
