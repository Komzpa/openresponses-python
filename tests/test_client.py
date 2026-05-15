from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from openresponses import (
    APIConnectionError,
    AuthenticationError,
    AsyncOpenResponsesClient,
    BadRequestError,
    ModelError,
    OpenResponsesClient,
    RateLimitError,
)


def _response_payload() -> dict[str, object]:
    return {
        "id": "resp_123",
        "object": "response",
        "created": 1,
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": "hello",
            }
        ],
    }


def _sync_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _async_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_sync_client_uses_injected_httpx_client_and_request_body_shape():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == "https://example.test/v1/responses"
        assert request.headers["authorization"] == "Bearer secret"
        assert json.loads(request.content) == {
            "model": "test-model",
            "input": "hello",
            "stream": False,
        }
        return httpx.Response(200, json=_response_payload())

    http_client = _sync_client(handler)
    client = OpenResponsesClient(
        "https://example.test/",
        api_key="secret",
        http_client=http_client,
    )

    response = client.create("test-model", "hello")
    client.close()

    assert response.id == "resp_123"
    assert len(requests) == 1
    assert not http_client.is_closed


def test_client_accepts_api_base_url_with_v1_suffix():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/v1/responses"
        return httpx.Response(200, json=_response_payload())

    client = OpenResponsesClient(
        "https://example.test/v1/",
        http_client=_sync_client(handler),
    )

    client.create("test-model", "hello")


def test_request_body_preserves_open_responses_options():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "hello",
                }
            ],
            "stream": False,
            "max_output_tokens": 16,
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "name": "turn_on",
                    "parameters": {"type": "object"},
                }
            ],
            "user": "conversation-1",
            "text": {"format": {"type": "json_object"}},
            "previous_response_id": "resp_previous",
        }
        return httpx.Response(200, json=_response_payload())

    client = OpenResponsesClient(
        "https://example.test",
        http_client=_sync_client(handler),
    )

    client.create(
        "test-model",
        [{"type": "message", "role": "user", "content": "hello"}],
        max_output_tokens=16,
        store=False,
        tools=[
            {
                "type": "function",
                "name": "turn_on",
                "parameters": {"type": "object"},
            }
        ],
        user="conversation-1",
        text={"format": {"type": "json_object"}},
        previous_response_id="resp_previous",
    )


def test_async_client_uses_injected_httpx_client_and_per_request_timeout():
    requests: list[httpx.Request] = []

    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            assert json.loads(request.content)["max_tool_calls"] == 2
            return httpx.Response(200, json=_response_payload())

        http_client = _async_client(handler)
        client = AsyncOpenResponsesClient(
            "https://example.test",
            http_client=http_client,
            timeout=30.0,
        )

        response = await client.create(
            "test-model",
            "hello",
            max_tool_calls=2,
            timeout=5.0,
        )
        await client.aclose()

        assert response.model == "test-model"
        assert not http_client.is_closed

    asyncio.run(run())
    assert len(requests) == 1


def test_streaming_sse_parsing_preserves_failure_events():
    stream = (
        "event: response.output_text.delta\n"
        'data: {"delta":"he"}\n\n'
        "event: response.failed\n"
        'data: {"response":{"status":"failed"},"error":{"message":"bad"}}\n\n'
        "event: response.incomplete\n"
        'data: {"response":{"incomplete_details":{"reason":"max_output_tokens"}}}\n\n'
        "event: error\n"
        'data: {"error":{"message":"transport failure"}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream,
        )

    client = OpenResponsesClient(
        "https://example.test",
        http_client=_sync_client(handler),
    )

    events = list(client.create("test-model", "hello", stream=True))

    assert [event.event for event in events] == [
        "response.output_text.delta",
        "response.failed",
        "response.incomplete",
        "error",
    ]
    assert events[1].data["error"]["message"] == "bad"
    assert events[3].data["error"]["message"] == "transport failure"


def test_streaming_status_errors_are_mapped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "slow down"}},
        )

    client = OpenResponsesClient(
        "https://example.test",
        http_client=_sync_client(handler),
    )

    with pytest.raises(RateLimitError) as err:
        list(client.create("test-model", "hello", stream=True))

    assert str(err.value) == "slow down"


def test_async_streaming_sse_parsing():
    async def run() -> list[str]:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content='event: response.output_text.done\ndata: {"text":"hello"}\n\n',
            )

        client = AsyncOpenResponsesClient(
            "https://example.test",
            http_client=_async_client(handler),
        )
        stream = await client.create("test-model", "hello", stream=True)
        events = [event.event async for event in stream]
        await client.aclose()
        return events

    assert asyncio.run(run()) == ["response.output_text.done"]


@pytest.mark.parametrize(
    ("status_code", "exception"),
    [
        (400, BadRequestError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, ModelError),
        (422, BadRequestError),
        (429, RateLimitError),
    ],
)
def test_status_errors_are_mapped(status_code: int, exception: type[Exception]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": "mapped"}},
        )

    client = OpenResponsesClient(
        "https://example.test",
        http_client=_sync_client(handler),
    )

    with pytest.raises(exception) as err:
        client.create("test-model", "hello")

    assert str(err.value) == "mapped"


def test_connection_errors_are_mapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("cannot connect")

    client = OpenResponsesClient(
        "https://example.test",
        http_client=_sync_client(handler),
    )

    with pytest.raises(APIConnectionError):
        client.create("test-model", "hello")
