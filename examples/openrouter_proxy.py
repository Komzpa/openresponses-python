import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openresponses import (
    OpenResponsesRequest,
    OpenResponsesOutput,
    ReasoningItem,
    MessageItem,
)
from openresponses.provider import OpenResponsesProvider

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found. Please set it in .env")

# Initialize OpenRouter Client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

app = FastAPI(title="OpenRouter Proxy")


@app.post("/v1/responses")
async def create_response(request: OpenResponsesRequest):
    """
    Open Responses Provider backed by OpenRouter.
    """
    messages = OpenResponsesProvider.map_request_to_messages(request)

    # Enable reasoning in request if supported by model/provider
    # For now, we trust the model parameter e.g. "deepseek/deepseek-r1"

    if request.stream:
        return StreamingResponse(
            stream_openrouter(request.model, messages), media_type="text/event-stream"
        )

    try:
        completion = await client.chat.completions.create(
            model=request.model, messages=messages, stream=False
        )
        return map_completion_to_output(completion, request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_openrouter(model: str, messages: list):
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta

            # Extract reasoning (provider specific)
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                yield OpenResponsesProvider.create_reasoning_delta(reasoning)

            if delta.content:
                yield OpenResponsesProvider.create_text_delta(delta.content)

        yield OpenResponsesProvider.create_done_event()

    except Exception as e:
        yield OpenResponsesProvider.create_sse_event("error", {"error": str(e)})


def map_completion_to_output(completion, model):
    output_items = []
    choice = completion.choices[0]

    # Attempt to extract reasoning
    reasoning = getattr(choice.message, "reasoning", None)
    if reasoning:
        output_items.append(ReasoningItem(content=reasoning))

    if choice.message.content:
        output_items.append(
            MessageItem(role="assistant", content=choice.message.content)
        )

    return OpenResponsesOutput(
        id=completion.id, created=completion.created, model=model, output=output_items
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
