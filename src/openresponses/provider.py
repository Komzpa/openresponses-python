from typing import Any, Dict, List
import json
from .models import (
    OpenResponsesRequest,
)


class OpenResponsesProvider:
    """
    Utilities for building Open Responses Providers.
    """

    @staticmethod
    def map_request_to_messages(request: OpenResponsesRequest) -> List[Dict[str, str]]:
        """
        Maps an OpenResponsesRequest input to a standard list of messages (OpenAI-compatible).
        """
        messages = []
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for item in request.input:
                if item.type == "message":
                    # Simple mapping; complex items need more logic
                    content = item.content
                    if isinstance(content, list):
                        # Flatten list of InputText to string for simple backends
                        content = "".join(
                            [i.text for i in content if i.type == "input_text"]
                        )

                    messages.append({"role": item.role, "content": content})
        return messages

    @staticmethod
    def create_sse_event(event_type: str, data: Any) -> str:
        """
        Helper to format an SSE event string.
        """
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def create_reasoning_delta(content: str) -> str:
        return OpenResponsesProvider.create_sse_event(
            "response.reasoning.delta", {"delta": content}
        )

    @staticmethod
    def create_text_delta(content: str) -> str:
        return OpenResponsesProvider.create_sse_event(
            "response.text.delta", {"delta": content}
        )

    @staticmethod
    def create_done_event() -> str:
        return "event: response.done\ndata: {}\n\n"
