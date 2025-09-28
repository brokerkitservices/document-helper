from langchain_core.callbacks import BaseCallbackHandler
from typing import Any
from langchain_core.outputs import LLMResult


class AgentLoggerCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print("LLM Starting...")
        print(f"Prompt sent to LLM:\n{prompts[0]}")
        print("---------")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        # Extract just the generated text from the response
        generated_text = response.generations[0][0].text
        print(f"LLM Response: {generated_text}")
        print("*********")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> Any:
        """Run when LLM errors."""
