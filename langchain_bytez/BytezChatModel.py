from typing import (
    Optional,
    Any,
    List,
)

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.chat_models.base import BaseChatModel

from .BytezModelMixin import (
    BytezModelMixin,
    _convert_message_to_dict,
)

# TODO inspect this package is a great example of how to organize extending a BaseChatModel
# from langchain_openai.chat_models.base import _convert_dict_to_message


class BytezChatModel(BytezModelMixin, BaseChatModel):
    """
    Bytez LLM integration with LangChain. Supports: ['chat', 'image-text-to-text', 'audio-text-to-text', 'video-text-to-text'].
    """

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_type": "ChatModel",
            **super()._identifying_params,
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop=None,
        run_manager=None,
        response_headers={},
    ):

        formatted_messages = [
            _convert_message_to_dict(messsage) for messsage in messages
        ]

        return super()._stream(formatted_messages, stop, run_manager, response_headers)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop=None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        response_headers={},
    ):

        formatted_messages = [
            _convert_message_to_dict(messsage) for messsage in messages
        ]

        async for event in super()._astream(
            formatted_messages, stop, run_manager, response_headers
        ):
            yield event

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response_headers = {}

        if self.streaming:
            stream_iter = self._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                response_headers=response_headers,
                **kwargs,
            )

            tokens = []

            for generation_chunk in stream_iter:
                tokens.append(generation_chunk.text)

            text_result = "".join(tokens)
        else:
            formatted_messages = [
                _convert_message_to_dict(messsage) for messsage in messages
            ]

            result = self._call(
                formatted_messages,
                stop=stop,
                run_manager=run_manager,
                response_headers=response_headers,
                **kwargs,
            )

            try:
                # TODO schema formatting needs to be made consistent on the BE
                text_result = result["content"][0]["text"]
            except Exception:
                text_result = result["content"]

        message = AIMessage(
            content=text_result,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata=self.get_meta_data_headers(response_headers),
            # TODO we need to modify our container code to support this, because deep within the preprocessing logic this can be determined
            # it's also dependent on the type of model, so we'd want to add the header as a parent header e.g. "Task-Specific-Meta-Data"
            # usage_metadata={
            #     "input_tokens": ct_input_tokens,
            #     "output_tokens": ct_output_tokens,
            #     "total_tokens": ct_input_tokens + ct_output_tokens,
            # },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response_headers = {}

        if self.streaming:
            stream_iter = self._astream(
                messages,
                stop=stop,
                run_manager=run_manager,
                response_headers=response_headers,
                **kwargs,
            )

            tokens = []

            async for generation_chunk in stream_iter:
                tokens.append(generation_chunk.text)

            text_result = "".join(tokens)
        else:
            formatted_messages = [
                _convert_message_to_dict(messsage) for messsage in messages
            ]
            result = await self._acall(
                formatted_messages,
                stop=stop,
                run_manager=run_manager,
                response_headers=response_headers,
                **kwargs,
            )

            text_result = result["content"][0]["text"]

        message = AIMessage(
            content=text_result,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata=self.get_meta_data_headers(response_headers),
            # TODO we need to modify our container code to support this, because deep within the preprocessing logic this can be determined
            # it's also dependent on the type of model, so we'd want to add the header as a parent header e.g. "Task-Specific-Meta-Data"
            # usage_metadata={
            #     "input_tokens": ct_input_tokens,
            #     "output_tokens": ct_output_tokens,
            #     "total_tokens": ct_input_tokens + ct_output_tokens,
            # },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
