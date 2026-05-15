import os
from typing import Dict, List, Optional

try:
    import anthropic
except ImportError:
    raise ImportError("The 'anthropic' library is required. Please install it using 'pip install anthropic'.")

from outhad_contextkit.configs.llms.base import BaseLlmConfig
from outhad_contextkit.llms.base import LLMBase


class AnthropicLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "claude-3-5-sonnet-20240620"

        api_key = (
            self.config.api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("CLAUDE_API_KEY")
        )
        self.client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _translate_tools(tools):
        """Translate OpenAI-style ``tools`` definitions to Claude shape.

        OpenAI:
            {"type": "function", "function": {"name": "...", "parameters": {...}}}

        Claude 4.x:
            {"type": "custom", "name": "...", "input_schema": {...},
             "description": "..."}

        Pass-through when the input already looks like a Claude tool.
        """
        out = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                out.append(tool)
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                fn = tool["function"]
                out.append(
                    {
                        "type": "custom",
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            elif "type" in tool:
                out.append(tool)
            elif "name" in tool and "input_schema" in tool:
                out.append({"type": "custom", **tool})
            else:
                out.append(tool)
        return out

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using Anthropic.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        # Separate system message from other messages
        system_message = ""
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                filtered_messages.append(message)

        # Claude 4.x rejects requests that carry both ``temperature`` and
        # ``top_p`` — pick one. We default to ``temperature`` since that
        # is what every other ``LlmFactory`` provider already uses.
        params = {
            "model": self.config.model,
            "messages": filtered_messages,
            "system": system_message,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        # Only include ``top_p`` when the caller explicitly opted out of
        # ``temperature`` (set it to None). This keeps the older
        # Claude 3 path working unchanged for callers who depended on it.
        if self.config.temperature is None and getattr(self.config, "top_p", None) is not None:
            params["top_p"] = self.config.top_p
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = self._translate_tools(tools)
            # Claude 4.x expects ``tool_choice`` to be a dict
            # (``{"type": "auto"}``) — Claude 3 accepted bare strings.
            # Normalise so both call sites work.
            if isinstance(tool_choice, str):
                params["tool_choice"] = {"type": tool_choice}
            else:
                params["tool_choice"] = tool_choice

        response = self.client.messages.create(**params)

        # Match OpenAILLM._parse_response shape so downstream Memory +
        # graph_memory callers don't have to special-case providers:
        #   * tools  → ``{"content": <text>, "tool_calls": [{"name": ..., "arguments": dict}]}``
        #   * no tools → plain string (concatenated text blocks).
        text_chunks = []
        tool_calls = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "name": getattr(block, "name", ""),
                        "arguments": getattr(block, "input", {}) or {},
                    }
                )
        joined_text = "".join(text_chunks)

        if tools:
            return {"content": joined_text, "tool_calls": tool_calls}
        return joined_text
