"""OpenAI-compatible adapter for responses wrapped in otherwise valid JSON prose."""

import json
from collections.abc import Iterable
from typing import Any

from openai import APIConnectionError, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from pydantic import BaseModel

from browser_use.llm.exceptions import ModelOutputTruncatedError, ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion


class OpenAICompatibleChatOpenAI(ChatOpenAI):
    """Keep Browser Use's structured request, tolerating prose around valid JSON."""

    @staticmethod
    def _extract_valid_json(raw: str, output_format: type[BaseModel]) -> BaseModel:
        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"invalid JSON constant: {value}")

        decoder = json.JSONDecoder(
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
        candidates: list[BaseModel] = []
        for index, char in enumerate(raw):
            if char != "{":
                continue
            try:
                _, end = decoder.raw_decode(raw, index)
                candidate = output_format.model_validate_json(raw[index:end])
            except (ValueError, TypeError):
                continue
            candidates.append(candidate)

        if len(candidates) != 1:
            raise ValueError(f"expected one valid {output_format.__name__} JSON object, found {len(candidates)}")
        return candidates[0]

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[BaseModel] | None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[Any]:
        if output_format is None:
            return await super().ainvoke(messages, output_format=output_format, **kwargs)

        openai_messages = self._serialize_messages(messages)
        model_params: dict[str, Any] = {}
        if self.temperature is not None:
            model_params["temperature"] = self.temperature
        if self.frequency_penalty is not None:
            model_params["frequency_penalty"] = self.frequency_penalty
        if self.max_completion_tokens is not None:
            model_params["max_completion_tokens"] = self.max_completion_tokens
        if self.top_p is not None:
            model_params["top_p"] = self.top_p
        if self.seed is not None:
            model_params["seed"] = self.seed
        if self.service_tier is not None:
            model_params["service_tier"] = self.service_tier
        if self.reasoning_models and any(str(m).lower() in str(self.model).lower() for m in self.reasoning_models):
            model_params["reasoning_effort"] = self.reasoning_effort
            model_params.pop("temperature", None)
            model_params.pop("frequency_penalty", None)

        response_format: JSONSchema = {
            "name": "agent_output",
            "strict": True,
            "schema": SchemaOptimizer.create_optimized_json_schema(
                output_format,
                remove_min_items=self.remove_min_items_from_schema,
                remove_defaults=self.remove_defaults_from_schema,
            ),
        }
        if self.add_schema_to_system_prompt and openai_messages and openai_messages[0]["role"] == "system":
            schema_text = f"\n<json_schema>\n{response_format}\n</json_schema>"
            if isinstance(openai_messages[0]["content"], str):
                openai_messages[0]["content"] += schema_text
            elif isinstance(openai_messages[0]["content"], Iterable):
                openai_messages[0]["content"] = list(openai_messages[0]["content"]) + [
                    ChatCompletionContentPartTextParam(text=schema_text, type="text")
                ]

        try:
            response = await self.get_client().chat.completions.create(
                model=self.model,
                messages=openai_messages,
                response_format=ResponseFormatJSONSchema(json_schema=response_format, type="json_schema"),
                **model_params,
            )
            choice = response.choices[0] if response.choices else None
            if choice is None:
                raise ModelProviderError(message="Invalid OpenAI chat completion response: missing choices.", status_code=502, model=self.name)
            if choice.finish_reason == "length":
                raise ModelOutputTruncatedError(
                    message=f"Model output was truncated at max_completion_tokens={self.max_completion_tokens}; the structured output is incomplete.",
                    model=self.name,
                )
            if choice.message.content is None:
                raise ModelProviderError(message="Failed to parse structured output from model response", status_code=500, model=self.name)

            raw = choice.message.content
            try:
                parsed = output_format.model_validate_json(raw)
            except (ValueError, TypeError):
                parsed = self._extract_valid_json(raw, output_format)
            return ChatInvokeCompletion(completion=parsed, usage=self._get_usage(response), stop_reason=choice.finish_reason)
        except ModelProviderError:
            raise
        except RateLimitError as exc:
            raise ModelRateLimitError(message=exc.message, model=self.name) from exc
        except APIConnectionError as exc:
            raise ModelProviderError(message=str(exc), model=self.name) from exc
        except APIStatusError as exc:
            raise ModelProviderError(message=exc.message, status_code=exc.status_code, model=self.name) from exc
        except Exception as exc:
            raise ModelProviderError(message=str(exc), model=self.name) from exc

    @staticmethod
    def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        from browser_use.llm.openai.serializer import OpenAIMessageSerializer

        return OpenAIMessageSerializer.serialize_messages(messages)
