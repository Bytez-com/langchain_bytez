import os
from typing import List, Union
import time
import asyncio

from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.utils import print_text
from langchain_core.messages import AIMessage, BaseMessageChunk

from langchain_bytez import BytezLLM
from langchain_bytez import BytezChatModel
from langchain_bytez import BytezStdOutCallbackHandler


# sudo cat /var/log/cloud-init-output.log


async def run_async_test():
    API_KEY = os.environ.get("API_KEY")

    # Example usage of BytezLLM

    cluster_params = {
        "capacity": {
            #
            "min": 1,
            "desired": 2,
            "max": 10,
        },
        "timeout": 60,
    }

    model_params = {
        "params": {"max_new_tokens": 3},
    }

    callbacks = [
        #
        StreamingStdOutCallbackHandler(),
        BytezStdOutCallbackHandler(color="green"),
    ]

    bytez_chat_model_phi = BytezChatModel(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        api_key=API_KEY,
        **cluster_params,
        **model_params,
        callbacks=callbacks,
        # debug=True, # send request to localhost:8080
        # other kwargs supported by langchain can then go here and will be set on the class instance as attr's
    )

    # only run when streaming is enabled
    bytez_llm = BytezLLM(
        model_id="microsoft/phi-2",
        api_key=API_KEY,
        **cluster_params,
        **model_params,
        callbacks=callbacks,
        # debug=True, # send request to localhost:8080
        # other kwargs supported by langchain can then go here and will be set on the class instance as attr's
    )

    bytez_chat_model_llama = BytezChatModel(
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        api_key=API_KEY,
        **cluster_params,
        **model_params,
        callbacks=callbacks,
        # debug=True, # send request to localhost:8080
        # other kwargs supported by langchain can then go here and will be set on the class instance as attr's
    )

    models: List[Union[BytezLLM, BytezChatModel]] = [
        bytez_chat_model_phi,
        bytez_llm,
        bytez_chat_model_llama,
    ]

    def shutdown_models():
        for model in models:
            try:
                model.shutdown_cluster()
            except Exception as exception:
                print(f"{model.model_id}", exception)

    def start_models():
        for model in models:
            try:
                model.start_cluster()
            except Exception as exception:
                print(f"{model.model_id}", exception)

            result = model.get_cluster()
            status = result["status"]

            while result["status"] == "loading":
                result = model.get_cluster()
                status = result["status"]

                print(f"{model.model_id} status: {status}")

                time.sleep(5)

            if status == "on":
                continue

            raise Exception(
                f"{model.model_id} - Cluster failed to create with status: {status}"
            )

    start_models()

    # shutdown_models()

    tests = [
        #
        # interface tests
        #
        lambda: print_text(f"Testing messages interface: {streaming}", color="pink"),
        lambda: test_chat_with_string_messages(bytez_chat_model=bytez_chat_model_phi),
        #
        # LLM tests
        #
        lambda: print_text(
            f"Testing LLM (not chat) with streaming: {streaming}", color="pink"
        ),
        lambda: test_llm_sync(bytez_llm=bytez_llm),
        lambda: test_llm_async(bytez_llm=bytez_llm),
        #
        # Chat tests
        #
        lambda: print_text(
            f"Testing Chat model with streaming: {streaming}", color="pink"
        ),
        lambda: test_chat_sync(bytez_chat_model=bytez_chat_model_llama),
        lambda: test_chat_async(bytez_chat_model=bytez_chat_model_llama),
    ]

    # shutdown_models()

    streaming_values = [
        #
        False,
        True,
    ]

    for streaming in streaming_values:
        for model in models:
            model.streaming = streaming

        for test in tests:
            result = test()
            if asyncio.iscoroutine(result):
                await result

        # NOTE this can be sped up by doing this, but it's easier to understand what's happening if run serially
        # await asyncio.gather(*(test() for test in tests))

    print_text("\n\n>>>>>>>>>> Tests pass <<<<<<<<<<<", color="blue")

    # shutdown_models()
    pass


test_string_input = (
    "Write a poem about Kanye West, make sure there's a yeezy reference."
)

test_batch_input = [
    "Write a poem about the stars.",
    "Write a poem about cats.",
    "Write a poem about pirates.",
]


async def test_llm_sync(bytez_llm: BytezLLM):
    response = bytez_llm.invoke(test_string_input)

    assert isinstance(response, str), "BytezLLM .invoke() must return a string"

    iterator = bytez_llm.stream(test_string_input)

    # NOTE consume the iterator so tokens can be logged via the logging callbacks
    for chunk in iterator:
        assert isinstance(
            chunk, str
        ), "BytezLLM .stream() must return an iterator that contains strings"

    # runs all prompts at the same time while blocking
    batch_response = bytez_llm.batch(test_batch_input)

    for response in batch_response:
        assert isinstance(
            response, str
        ), "BytezLLM .batch() must return an iterable that contains strings"

    # runs all prompts at the same time while blocking and yields results
    iterator = bytez_llm.batch_as_completed(test_batch_input)

    for index, output in iterator:
        assert isinstance(
            index, int
        ), "BytezLLM .batch_as_completed() must return an iterator of tuples that contain (int, str)"
        assert isinstance(
            output, str
        ), "BytezLLM .batch_as_completed() must return an iterator of tuples that contain (int, str)"

    pass


async def test_llm_async(bytez_llm: BytezLLM):
    response = await bytez_llm.ainvoke(test_string_input)

    assert isinstance(response, str), "BytezLLM .ainvoke() must return a string"

    async_iterator = bytez_llm.astream(test_string_input)

    async for chunk in async_iterator:
        assert isinstance(
            chunk, str
        ), "BytezLLM .astream() must return an async iterator containing strings"

    # runs all prompts at the same time
    batch_response = await bytez_llm.abatch(test_batch_input)

    for response in batch_response:
        assert isinstance(
            response, str
        ), "BytezLLM .abatch() must return an iterable containing strings"

    # NOTE runs all the prompts at the same time and streams out the results
    async_iterator = bytez_llm.abatch_as_completed(test_batch_input)

    async for index, output in async_iterator:
        assert isinstance(
            index, int
        ), "BytezLLM .abatch() must return an async iterator of tuples (int, str)"
        assert isinstance(
            output, str
        ), "BytezLLM .abatch() must return an async iterator of tuples (int, str)"

    pass


system_message = SystemMessage(
    content=[
        {
            "type": "text",
            "text": "You are a helpful assistant that answers questions clearly and concisely.",
        }
    ]
)


human_message = HumanMessage(
    content=[
        {"type": "text", "text": "What is this image?"},
        {
            "type": "image",
            "url": "https://hips.hearstapps.com/hmg-prod/images/how-to-keep-ducks-call-ducks-1615457181.jpg?crop=0.670xw:1.00xh;0.157xw,0&resize=980:*",
        },
    ]
)

messages = [system_message, human_message]

batch_prompts = [messages, messages, messages]


async def test_chat_sync(bytez_chat_model: BytezChatModel):
    response = bytez_chat_model.invoke(messages)

    assert isinstance(
        response, AIMessage
    ), "BytezChatModel .invoke() must return an AIMessage"

    iterator = bytez_chat_model.stream(messages)

    for chunk in iterator:
        assert isinstance(
            chunk, BaseMessageChunk
        ), "BytezChatModel .invoke() must return an iterator of BaseMessageChunk's"

    batch_response = bytez_chat_model.batch(batch_prompts)

    for output in batch_response:
        assert isinstance(
            output, AIMessage
        ), "BytezChatModel .batch() must return an iterable of AIMessage's"

    iterator = bytez_chat_model.batch_as_completed(batch_prompts)

    for index, output in iterator:
        assert isinstance(
            index, int
        ), "BytezChatModel .batch_as_completed() must return an iterator of tuples (int, AIMessage)"
        assert isinstance(
            output, AIMessage
        ), "BytezChatModel .batch_as_completed() must return an iterator of tuples (int, AIMessage)"
        pass

    pass


async def test_chat_async(bytez_chat_model: BytezChatModel):
    response = await bytez_chat_model.ainvoke(messages)

    assert isinstance(
        response, AIMessage
    ), "BytezChatModel .ainvoke() must return an AIMessage"

    async_iterator = bytez_chat_model.astream(messages)

    async for chunk in async_iterator:
        assert isinstance(
            chunk, BaseMessageChunk
        ), "BytezChatModel .astream() must return an async iterator of AIMessage's"

    async_iterator = bytez_chat_model.astream_events(messages, version="v2")

    async for event in async_iterator:
        assert isinstance(
            event, dict
        ), "BytezChatModel .astream_events() must return an async iterator of dict's"

    batch_response = await bytez_chat_model.abatch(batch_prompts)

    for response in batch_response:
        assert isinstance(
            response, AIMessage
        ), "BytezChatModel .abatch() must return an iterable of AIMessage's"

    async_iterator = bytez_chat_model.abatch_as_completed(batch_prompts)

    async for index, output in async_iterator:
        assert isinstance(
            index, int
        ), "BytezChatModel .abatch_as_completed() must return an async iterator of tuples (int, AIMessage)"
        assert isinstance(
            output, AIMessage
        ), "BytezChatModel .abatch_as_completed() must return an async iterator of tuples (int, AIMessage)"

    pass


# NOTE this is to make sure that strings are correctly converted to our schema {"type": "text", "text": "Hi there!"}
# there are inconsistencies in the schema for chat models and image-text-to-text, so we need to modify it on the BE
async def test_chat_with_string_messages(bytez_chat_model: BytezChatModel):
    system_message_as_string = SystemMessage(
        content="You are a helpful assistant that answers questions clearly and concisely."
    )

    human_message_as_string = HumanMessage(content="My name is: ")

    string_messages = [system_message_as_string, human_message_as_string]

    # response = bytez_chat_model.invoke(messages)
    response = bytez_chat_model.invoke(string_messages)

    assert isinstance(
        response, AIMessage
    ), "BytezChatModel .invoke() must return an AIMessage"


if __name__ == "__main__":
    result = asyncio.run(run_async_test())
