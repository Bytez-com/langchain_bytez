from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.utils import print_text
from langchain_core.outputs import LLMResult


# NOTE this is experimental and helps serve as a bootstrapper for your own code
# extend this however you see fit
class BytezStdOutCallbackHandler(StdOutCallbackHandler):
    def print(self, *args, color: str = None, **kwargs):
        text = " ".join([str(arg) for arg in args])
        print_text(f"\n{text}", color or self.color, **kwargs)

    def _on_start(self, serialized, prompts, *args, **kwargs):
        model_type = kwargs["invocation_params"]["model_type"]
        model_name = kwargs["invocation_params"]["model_name"]

        self.print(f"Model started: {model_name} ({model_type})", color="blue")
        self.print("Prompt:", prompts, color="yellow")

    def on_llm_start(self, *args, **kwargs):
        self._on_start(*args, **kwargs)

    def on_llm_end(self, response: LLMResult, *args, **kwargs):
        self.print("Model responded with: ", response, "\n")

    def on_chat_model_start(self, *args, **kwargs):
        self._on_start(*args, **kwargs)
