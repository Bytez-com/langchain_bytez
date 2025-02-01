from langchain.llms.base import LLM

from .BytezModelMixin import BytezModelMixin


class BytezLLM(BytezModelMixin, LLM):
    """
    Bytez LLM integration with LangChain. Supports only text-generation models.
    """

    @property
    def payload_input_key(self) -> str:
        return "text"

    @property
    def _identifying_params(self) -> dict:
        """
        Return the parameters that uniquely identify this LLM.
        """
        return {
            "model_type": "LLM",
            **super()._identifying_params,
        }
