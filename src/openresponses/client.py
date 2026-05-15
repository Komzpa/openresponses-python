from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator, Iterable
from typing import Any, Optional, Union

import httpx

from .exceptions import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ModelError,
    RateLimitError,
)
from .models import OpenResponsesRequest, OpenResponsesOutput, StreamEvent, MessageItem


DEFAULT_TIMEOUT = 60.0


def _normalize_base_url(base_url: str) -> str:
    """Return the provider root URL used before the Open Responses path."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def _request_json(request: OpenResponsesRequest) -> dict[str, Any]:
    """Return the compact request body sent to providers."""
    return request.model_dump(exclude_none=True)


def _extract_error_message(response: httpx.Response) -> tuple[str, Any | None]:
    try:
        body = response.json()
    except json.JSONDecodeError:
        text = response.text
        return text or response.reason_phrase, text or None

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            if message:
                return str(message), body
        if isinstance(error, str):
            return error, body
        message = body.get("message")
        if message:
            return str(message), body

    return response.reason_phrase, body


def _raise_mapped_status(response: httpx.Response) -> None:
    if response.is_success:
        return

    message, body = _extract_error_message(response)
    status_code = response.status_code
    error_kwargs = {"status_code": status_code, "response_body": body}

    if status_code in (401, 403):
        raise AuthenticationError(message, **error_kwargs)
    if status_code == 429:
        raise RateLimitError(message, **error_kwargs)
    if status_code in (400, 422):
        raise BadRequestError(message, **error_kwargs)
    if status_code == 404:
        raise ModelError(message, **error_kwargs)
    raise APIStatusError(message, **error_kwargs)


def _map_connection_error(error: httpx.HTTPError) -> APIConnectionError:
    mapped_error = APIConnectionError(str(error))
    mapped_error.__cause__ = error
    return mapped_error


def _parse_sse_lines(lines: Iterable[str]) -> Generator[StreamEvent, None, None]:
    event_type = "message"
    data_lines: list[str] = []

    for line in lines:
        if not line:
            if data_lines:
                data_str = "\n".join(data_lines)
                data_lines.clear()
                if data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                yield StreamEvent(event=event_type, data=data)
                event_type = "message"
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    if data_lines:
        data_str = "\n".join(data_lines)
        if data_str != "[DONE]":
            try:
                yield StreamEvent(event=event_type, data=json.loads(data_str))
            except json.JSONDecodeError:
                return


async def _parse_async_sse_lines(
    lines: AsyncGenerator[str, None],
) -> AsyncGenerator[StreamEvent, None]:
    event_type = "message"
    data_lines: list[str] = []

    async for line in lines:
        if not line:
            if data_lines:
                data_str = "\n".join(data_lines)
                data_lines.clear()
                if data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                yield StreamEvent(event=event_type, data=data)
                event_type = "message"
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    if data_lines:
        data_str = "\n".join(data_lines)
        if data_str != "[DONE]":
            try:
                yield StreamEvent(event=event_type, data=json.loads(data_str))
            except json.JSONDecodeError:
                return


class OpenResponsesClient:
    """
    Async/Sync Client for Open Responses API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        http_client: httpx.Client | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._client = http_client
        self._owns_client = http_client is None
        self.timeout = timeout

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()

    def create(
        self,
        model: str,
        input: Union[str, list[MessageItem], list[dict[str, Any]]],
        stream: bool = False,
        max_tool_calls: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        store: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        user: str | None = None,
        text: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        **extra_fields: Any,
    ) -> Union[OpenResponsesOutput, Generator[StreamEvent, None, None]]:
        """
        Synchronous request to create a response.
        """
        request = OpenResponsesRequest(
            model=model,
            input=input,
            stream=stream,
            max_tool_calls=max_tool_calls,
            max_output_tokens=max_output_tokens,
            store=store,
            tools=tools,
            user=user,
            text=text,
            **extra_fields,
        )

        url = f"{self.base_url}/v1/responses"

        if stream:
            return self._stream_request(url, request, timeout=timeout)

        try:
            resp = self.client.post(
                url,
                json=_request_json(request),
                headers=self.headers,
                timeout=self.timeout if timeout is None else timeout,
            )
        except httpx.HTTPError as err:
            raise _map_connection_error(err)

        _raise_mapped_status(resp)
        return OpenResponsesOutput(**resp.json())

    def _stream_request(
        self,
        url: str,
        request: OpenResponsesRequest,
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> Generator[StreamEvent, None, None]:
        try:
            with self.client.stream(
                "POST",
                url,
                json=_request_json(request),
                headers=self.headers,
                timeout=self.timeout if timeout is None else timeout,
            ) as resp:
                if not resp.is_success:
                    resp.read()
                _raise_mapped_status(resp)
                yield from _parse_sse_lines(resp.iter_lines())
        except httpx.HTTPError as err:
            raise _map_connection_error(err)

    def __enter__(self) -> OpenResponsesClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncOpenResponsesClient:
    """
    Async Client for Open Responses API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._client = http_client
        self._owns_client = http_client is None
        self.timeout = timeout

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    async def create(
        self,
        model: str,
        input: Union[str, list[MessageItem], list[dict[str, Any]]],
        stream: bool = False,
        max_tool_calls: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        store: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        user: str | None = None,
        text: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        **extra_fields: Any,
    ) -> Union[OpenResponsesOutput, AsyncGenerator[StreamEvent, None]]:
        request = OpenResponsesRequest(
            model=model,
            input=input,
            stream=stream,
            max_tool_calls=max_tool_calls,
            max_output_tokens=max_output_tokens,
            store=store,
            tools=tools,
            user=user,
            text=text,
            **extra_fields,
        )

        url = f"{self.base_url}/v1/responses"

        if stream:
            return self._stream_request(url, request, timeout=timeout)

        try:
            resp = await self.client.post(
                url,
                json=_request_json(request),
                headers=self.headers,
                timeout=self.timeout if timeout is None else timeout,
            )
        except httpx.HTTPError as err:
            raise _map_connection_error(err)

        _raise_mapped_status(resp)
        return OpenResponsesOutput(**resp.json())

    async def _stream_request(
        self,
        url: str,
        request: OpenResponsesRequest,
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        try:
            async with self.client.stream(
                "POST",
                url,
                json=_request_json(request),
                headers=self.headers,
                timeout=self.timeout if timeout is None else timeout,
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
                _raise_mapped_status(resp)
                async for event in _parse_async_sse_lines(resp.aiter_lines()):
                    yield event
        except httpx.HTTPError as err:
            raise _map_connection_error(err)

    async def __aenter__(self) -> AsyncOpenResponsesClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
