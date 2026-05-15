import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openresponses import OpenResponsesRequest, MessageItem, OpenResponsesOutput
from openresponses.provider import OpenResponsesProvider

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://api-inference.huggingface.co/v1/")

# Use OpenAI compatible client for HF Inference Endpoints / TGI
client = AsyncOpenAI(base_url=HF_BASE_URL, api_key=HF_API_KEY or "hf_token")

app = FastAPI(title="HuggingFace Proxy")


@app.post("/v1/responses")
async def create_response(request: OpenResponsesRequest):
    messages = OpenResponsesProvider.map_request_to_messages(request)
    # For HF Inference API, model is often part of URL, but sometimes passed as param
    model = request.model or "tgi"

    if request.stream:
        return StreamingResponse(
            stream_hf(model, messages), media_type="text/event-stream"
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


async def stream_hf(model: str, messages: list):
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

    uvicorn.run(app, host="0.0.0.0", port=8005)
