# langchain_bytez

This package allows you to use the Bytez API in langchain. Note, only text-generation, chat, image-text-to-text, video-text-to-text, and audio-text-to-text are currently supported.

Fully supports streaming + native async!

Curious about what else Bytez has to offer? You can check out Bytez [here](https://bytez.com).

Want to know more about our API? Check out the [docs](https://docs.bytez.com)!

# Chat Example

```py
import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

from langchain_bytez import BytezChatModel

API_KEY = os.environ.get("API_KEY")


bytez_chat_model_phi = BytezChatModel(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    api_key=API_KEY,
    capacity={
        "min": 1,
        "max": 1,  # up to 10 instances
    },
    params={"max_new_tokens": 64},
    timeout=10,  # minutes before expiring
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers questions clearly and concisely."
    ),
    HumanMessage(content="List the phylums in the biological taxonomy"),
]

results = bytez_chat_model_phi.invoke(messages)
```

# Text generation (LLM) example

```python
import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

from langchain_bytez import BytezLLM

API_KEY = os.environ.get("API_KEY")


bytez_chat_model_phi = BytezLLM(
    model_id="microsoft/phi-2",
    api_key=API_KEY,
    capacity={
        "min": 1,
        "max": 1,  # up to 10 instances
    },
    params={"max_new_tokens": 64},
    timeout=10,  # minutes before expiring
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers questions clearly and concisely."
    ),
    HumanMessage(content="List the phylums in the biological taxonomy"),
]

results = bytez_chat_model_phi.invoke(messages)
```

# Extending callback handlers for better observability

_NOTE_ this is experimental and we're working to enhance it. In the meantime it will help bootstrap you in doing whatever you need to do with a model's "run" lifecycle.

```py
import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

from langchain_bytez import BytezChatModel, BytezStdOutCallbackHandler

API_KEY = os.environ.get("API_KEY")


bytez_chat_model_phi = BytezChatModel(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    api_key=API_KEY,
    capacity={
        "min": 1,
        "max": 1,  # up to 10 instances
    },
    params={"max_new_tokens": 64},
    timeout=10,  # minutes before expiring
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler(), BytezStdOutCallbackHandler()],
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers questions clearly and concisely."
    ),
    HumanMessage(content="List the phylums in the biological taxonomy"),
]

results = bytez_chat_model_phi.invoke(messages)

```

# Shutdown your cluster

```py
bytez_chat_model_phi.shutdown_cluster()
```

# Update your cluster

```py
bytez_chat_model_phi.capacity = {
    "min": 2,  # we've increased the minimum number of instances
    "max": 3,  # up to 10 instances
}

bytez_chat_model_phi.update_cluster()
```

# kwargs for BytezChatModel and BytezLLM

```py
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
```

## API Playground

Explore our API endpoints in the documentation [here](https://docs.bytez.com/model-api/playground/overview).

## Status

Check out the status of our [API](https://status.bytez.com)

## Resources

Get to know our story, our mission, and our roadmap [here](https://docs.bytez.com/company/about).

## Feedback

We’re committed to building the best developer experience for AI builders. Have feedback? Let us know on [Discord](https://discord.com/invite/Z723PfCFWf) or open an issue on [GitHub](https://github.com/Bytez-com/docs/issues).
