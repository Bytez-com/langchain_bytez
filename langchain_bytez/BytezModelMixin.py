import json
from typing import Optional, AsyncIterator, Any, Dict, Iterator, List, Union, Literal
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.outputs import (
    GenerationChunk,
    ChatGenerationChunk,
)
from langchain_core.messages import AIMessage, BaseMessage, AIMessageChunk
from langchain.schema import HumanMessage, SystemMessage

from langchain_core.messages import (
    ChatMessage,
    FunctionMessage,
    InvalidToolCall,
    ToolCall,
    ToolMessage,
)

from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel

# NOTE inspect this package for speedups surrounding converting langchain messages into our schema
# from langchain_openai.chat_models.base import ChatOpenAI


from pydantic import Field
import requests
import httpx
import asyncio
import threading

# TODO consider returning the response_headers dict with each call, right now it's handled in an impure fashion
# besides having the downsides of being impure, there are too many layers of indirection for the reader

# NOTE required to get around unsafe threading with "requests" and "httpx"
LOCK = threading.Lock()

# NOTE required to get around unsafe threading with "httpx"
ASYNC_IO_LOCKS = {}


# this allows you to determine the event loop a request is running in before acquiring a lock
def get_async_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    if loop not in ASYNC_IO_LOCKS:
        ASYNC_IO_LOCKS[loop] = asyncio.Lock()
    return ASYNC_IO_LOCKS[loop]


"""Base class for chat models.

Key imperative methods:
    Methods that actually call the underlying model.

    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | Method                    | Input                                                          | Output                                                              | Description                                                                                      |
    +===========================+================================================================+=====================================================================+==================================================================================================+
    | `invoke`                  | str | List[dict | tuple | BaseMessage] | PromptValue           | BaseMessage                                                         | A single chat model call.                                                                        |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `ainvoke`                 | '''                                                            | BaseMessage                                                         | Defaults to running invoke in an async executor.                                                 |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `stream`                  | '''                                                            | Iterator[BaseMessageChunk]                                          | Defaults to yielding output of invoke.                                                           |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `astream`                 | '''                                                            | AsyncIterator[BaseMessageChunk]                                     | Defaults to yielding output of ainvoke.                                                          |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `astream_events`          | '''                                                            | AsyncIterator[StreamEvent]                                          | Event types: 'on_chat_model_start', 'on_chat_model_stream', 'on_chat_model_end'.                 |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `batch`                   | List[''']                                                      | List[BaseMessage]                                                   | Defaults to running invoke in concurrent threads.                                                |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `abatch`                  | List[''']                                                      | List[BaseMessage]                                                   | Defaults to running ainvoke in concurrent threads.                                               |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `batch_as_completed`      | List[''']                                                      | Iterator[Tuple[int, Union[BaseMessage, Exception]]]                 | Defaults to running invoke in concurrent threads.                                                |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
    | `abatch_as_completed`     | List[''']                                                      | AsyncIterator[Tuple[int, Union[BaseMessage, Exception]]]            | Defaults to running ainvoke in concurrent threads.                                               |
    +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+

    This table provides a brief overview of the main imperative methods. Please see the base Runnable reference for full documentation.

Key declarative methods:
    Methods for creating another Runnable using the ChatModel.

    +----------------------------------+-----------------------------------------------------------------------------------------------------------+
    | Method                           | Description                                                                                               |
    +==================================+===========================================================================================================+
    | `bind_tools`                     | Create ChatModel that can call tools.                                                                     |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------+
    | `with_structured_output`         | Create wrapper that structures model output using schema.                                                 |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------+
    | `with_retry`                     | Create wrapper that retries model calls on failure.                                                       |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------+
    | `with_fallbacks`                 | Create wrapper that falls back to other models on failure.                                                |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------+
    | `configurable_fields`            | Specify init args of the model that can be configured at runtime via the RunnableConfig.                  |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------+
    | `configurable_alternatives`      | Specify alternative models which can be swapped in at runtime via the RunnableConfig.                     |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------+

    This table provides a brief overview of the main declarative methods. Please see the reference for each method for full documentation.

Creating custom chat model:
    Custom chat model implementations should inherit from this class.
    Please reference the table below for information about which
    methods and properties are required or optional for implementations.

    +----------------------------------+--------------------------------------------------------------------+-------------------+
    | Method/Property                  | Description                                                        | Required/Optional |
    +==================================+====================================================================+===================+
    | `_generate`                      | Use to generate a chat result from a prompt                        | Required          |
    +----------------------------------+--------------------------------------------------------------------+-------------------+
    | `_llm_type` (property)           | Used to uniquely identify the type of the model. Used for logging. | Required          |
    +----------------------------------+--------------------------------------------------------------------+-------------------+
    | `_identifying_params` (property) | Represent model parameterization for tracing purposes.             | Optional          |
    +----------------------------------+--------------------------------------------------------------------+-------------------+
    | `_stream`                        | Use to implement streaming                                         | Optional          |
    +----------------------------------+--------------------------------------------------------------------+-------------------+
    | `_agenerate`                     | Use to implement a native async method                             | Optional          |
    +----------------------------------+--------------------------------------------------------------------+-------------------+
    | `_astream`                       | Use to implement async version of `_stream`                        | Optional          |
    +----------------------------------+--------------------------------------------------------------------+-------------------+

    Follow the guide for more information on how to implement a custom Chat Model:
    [Guide](https://python.langchain.com/docs/how_to/custom_chat_model/).

"""


class BytezModelMixin:
    """
    Bytez Model Mixin, provides basic method overrides for the langchain LLM base and BaseChatModel.
    """

    model_id: str = Field(..., description="The unique model ID for the Bytez LLM.")
    api_key: str = Field(..., description="The API key for accessing the Bytez LLM.")
    capacity: dict = Field(
        default_factory=dict,
        description="Controls the scaling behavior, contains one or all keys 'desired': int, 'min': int, and 'max': int",
    )
    timeout: int = Field(
        None,
        description="Controls how many minutes to wait after the last inference to shutdown the cluster",
    )
    streaming: bool = Field(
        False, description="Enable streaming responses from the API."
    )
    params: dict = Field(
        default_factory=dict, description="Parameters passed to the Bytez API."
    )
    headers: dict = Field(
        default_factory=dict,
        description="Additional headers for the Bytez API. Matching keys override the defaults.",
    )
    http_timeout_s: float = Field(
        60 * 5.0,
        description="How long to wait in seconds for a response from the model before timing out",
    )
    debug: bool = Field(
        False,
        description="Control whether to use localhost for requests",
    )

    host: str = Field(
        "https://api.bytez.com",
        description="The host to send requests to",
    )
    url: str = Field(
        "",
        description="The full url to send requests to",
    )

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.headers = {"Authorization": f"Key {self.api_key}", **self.headers}
        self.host = "http://localhost:8080" if self.debug else "https://api.bytez.com"
        self.url = f"{self.host}/models/v2/{self.model_id}"

    @property
    def payload_input_key(self) -> Union[Literal["messages", "text"]]:
        # Is either 'messages' for a chat model, or 'text' for a LLM
        return "messages"

    @property
    def _identifying_params(self) -> dict:
        """
        Return the parameters that uniquely identify this LLM.
        """
        return {
            "model_name": self._llm_type,
            "model_id": self.model_id,
            "api_key": self.api_key,
            "capacity": self.capacity,
            "timeout": self.timeout,
            "streaming": self.streaming,
            "params": self.params,
            "headers": self.headers,
            "http_timeout_s": self.http_timeout_s,
            "debug": self.debug,
        }

    @property
    def _llm_type(self) -> str:
        """
        Return a unique identifier for this LLM type.
        """

        name = f"bytez_model:{self.model_id}"

        return name

    def contruct_model_payload(self, input, stream: bool = False):
        payload = {
            self.payload_input_key: input,
            "params": self.params,
            "stream": stream,
        }
        return payload

    def text_to_chunk(
        self,
        text: Union[str, bytes],
    ) -> Union[GenerationChunk, ChatGenerationChunk]:
        if isinstance(self, LLM):
            chunk = GenerationChunk(text=text)

        elif isinstance(self, BaseChatModel):
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=(
                        text
                        if isinstance(text, str)
                        else text.decode("utf-8", errors="replace")
                    ),
                ),
            )
        else:
            raise Exception(
                "The BytezModelMixin only suppports the base classes LLM and BaseChatModel, unable to handle iterator output."
            )

        return chunk

    def get_meta_data_headers(self, response_headers: dict):
        return {
            "ratelimit-limit": response_headers["ratelimit-limit"],
            "ratelimit-remaining": response_headers["ratelimit-remaining"],
            "ratelimit-reset": response_headers["ratelimit-reset"],
            "inference-meter": response_headers["inference-meter"],
            "inference-meter-price": response_headers["inference-meter-price"],
            "inference-time": response_headers["inference-time"],
            "inference-cost": response_headers["inference-cost"],
        }

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        response_headers={},
    ) -> Union[Iterator[GenerationChunk], Iterator[ChatGenerationChunk]]:

        payload = self.contruct_model_payload(prompt, stream=True)

        iterator = self._make_request_stream(
            method="post",
            payload=payload,
            response_headers=response_headers,
        )

        for chunk in iterator:
            chunk = self.text_to_chunk(chunk)

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    def collect_response_headers(self, response: Any, response_headers: dict):
        for key, value in response.headers.items():
            # NOTE this is because the async library httpx has a bug in one of its libraries that prevents preserving the casing from the server
            # so we need to enforce consistency for both types of calls
            response_headers[key.lower()] = value

    def _make_request(
        self,
        method: str,
        payload={},
        response_headers={},
        **kwargs,
    ):
        response = requests.request(
            url=self.url,
            method=method,
            headers=self.headers,
            json=payload,
            stream=False,
            timeout=self.http_timeout_s,
        )

        self.collect_response_headers(response, response_headers)

        # an error is always returned in the JSON if it's not a streaming request
        json = response.json()

        error = json["error"]

        # NOTE assign this here to view its output in the debugger
        output = json["output"]

        # NOTE, if there is a problem, this NEEDS to throw, do not swallow errors
        if error:
            raise Exception(error)

        return output

    def _make_request_stream(
        self,
        method: str,
        payload={},
        response_headers={},
        **kwargs,
    ):
        # NOTE there's still a problem with urllib3, which "requests" is built ontop of, that causes a deadlock when
        # this is executed concurrently, hence, we lock the request to prevent that malarky
        with LOCK:
            response = requests.request(
                url=self.url,
                method=method,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=self.http_timeout_s,
            )

        # we raise for status only for streaming, as it is currently the only way through which we can notify the client
        # that there is a problem with the request
        response.raise_for_status()

        for chunk in response.iter_content():
            yield chunk

        # transfer encoding: chunked supports the ability to update the headers as chunks are sent
        # we collect the last headers
        # TODO there is a bug on the BE where these are not updated properly
        self.collect_response_headers(response, response_headers)

    async def _make_async_request(
        self,
        method: str,
        payload: Optional[dict] = None,
        response_headers={},
    ):
        async with httpx.AsyncClient() as client:
            response = await client.request(
                url=self.url,
                method=method,
                headers=self.headers,
                json=payload,
                timeout=self.http_timeout_s,
            )

            self.collect_response_headers(response, response_headers)

            json = response.json()
            error = json.get("error")
            if error:
                raise Exception(error)

            output = json.get("output")

            return output

    async def _make_async_request_stream(
        self,
        method: str,
        payload: Optional[dict] = None,
        iterator_consumer: Optional[callable] = None,
        response_headers={},
    ):
        async with httpx.AsyncClient() as client:

            # NOTE we need to lock the critical section, which is calling client.stream()
            # there is a problem with it that leads to deadlocks if multiple coroutines call it at the same time
            # likely in the C code for httpcore, similar to urllib3
            lock = get_async_lock()

            response = None

            try:
                await lock.acquire()

                ######### critical section #########
                response_cm = client.stream(
                    method=method,
                    url=self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.http_timeout_s,
                )

                response = await response_cm.__aenter__()
                ######### end of critical section #########

                # free the lock
                lock.release()

                # should throw if something unexpected happens
                response.raise_for_status()

                # NOTE this allows us to provide a mechanism for returning an async iterator directly
                # this allows us to "plug in" to the async iterator
                # it can't be directly returned because the stream's with block will close the connection
                if iterator_consumer:
                    await iterator_consumer(response.aiter_text())

                    # transfer encoding: chunked supports the ability to update the headers as chunks are sent
                    # we collect the last headers
                    # NOTE there is a bug on the BE where these are not updated properly
                    self.collect_response_headers(response, response_headers)
                    return

            except Exception as exception:
                # only release on failure, if it was successful, it was already unlocked
                lock.release()
                raise exception
            finally:
                if response:
                    # explicitly close the request so we can get the updated headers, this uses chunked transfer encoding
                    await response.aclose()

                    for key, value in response.headers.items():
                        response_headers[key] = value

                await response_cm.__aexit__(None, None, None)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        response_headers={},
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:

        payload = self.contruct_model_payload(prompt, stream=True)

        queue = asyncio.Queue()

        async def iterator_consumer(async_iterator):
            async for chunk in async_iterator:
                await queue.put(chunk)
            await queue.put(None)  # Signal end of stream

        async def deplete_queue():
            while True:
                chunk = await queue.get()
                if chunk is None:  # End of stream
                    break

                chunk = self.text_to_chunk(chunk)

                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

                yield chunk

        make_request_task = asyncio.create_task(
            self._make_async_request_stream(
                method="post",
                payload=payload,
                iterator_consumer=iterator_consumer,
                response_headers=response_headers,
            )
        )

        async for result in deplete_queue():
            yield result

        await make_request_task

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        response_headers: dict = {},
        **kwargs,
    ) -> str:
        """
        Call the custom LLM API with the given prompt and return the response.
        """

        if self.streaming:
            chunks = []
            for chunk in self._stream(
                prompt,
                stop=stop,
                run_manager=run_manager,
                response_headers=response_headers,
            ):

                chunks.append(chunk.text)

            text = "".join(chunks)
            return text

        payload = self.contruct_model_payload(prompt, stream=self.streaming)

        text = self._make_request(
            method="post",
            payload=payload,
            response_headers=response_headers,
        )

        return text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        response_headers={},
        **kwargs,
    ) -> str:
        """
        Asynchronous call with streaming support.
        """

        if self.streaming:
            chunks = []
            async for chunk in self._astream(
                prompt,
                stop=stop,
                run_manager=run_manager,
                response_headers=response_headers,
            ):

                chunks.append(chunk.text)

            text = "".join(chunks)
            return text

        payload = self.contruct_model_payload(prompt, stream=self.streaming)

        text = await self._make_async_request(
            method="post",
            payload=payload,
            response_headers=response_headers,
        )

        return text

    def start_cluster(self):
        payload = {"capacity": self.capacity, "timeout": self.timeout}
        return self._make_request(method="put", payload=payload)

    def get_cluster(self):
        return self._make_request(method="get")

    def update_cluster(self):
        payload = {"capacity": self.capacity, "timeout": self.timeout}
        return self._make_request(method="patch", payload=payload)

    def shutdown_cluster(self):
        return self._make_request(method="delete")


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {"content": _format_message_content(message.content)}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        if "audio" in message.additional_kwargs:
            # openai doesn't support passing the data back - only the id
            # https://platform.openai.com/docs/guides/audio/multi-turn-conversations
            raw_audio = message.additional_kwargs["audio"]
            audio = (
                {"id": message.additional_kwargs["audio"]["id"]}
                if "id" in raw_audio
                else raw_audio
            )
            message_dict["audio"] = audio
    elif isinstance(message, SystemMessage):
        message_dict["role"] = message.additional_kwargs.get(
            "__openai_role__", "system"
        )
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        # Remove unexpected block types
        formatted_content = []
        for block in content:
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] == "tool_use"
            ):
                continue
            # NOTE this may need to be reenabled
            # if isinstance(block, str):
            #     formatted_content.append({"type": "text", "text": content})
            else:
                formatted_content.append(block)

    # NOTE this may need to be reenabled
    # elif content and isinstance(content, str):
    #     formatted_content = [{"type": "text", "text": content}]
    else:
        formatted_content = content

    return formatted_content


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }
