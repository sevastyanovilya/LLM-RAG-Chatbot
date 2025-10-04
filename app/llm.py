"""LLM provider abstraction supporting local and hosted models."""
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt."""
        pass


class LocalLLM(LLMProvider):
    """Local LLM using llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        try:
            from llama_cpp import Llama

            logger.info(f"Loading local model from {model_path}")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=4,
                n_gpu_layers=0,  # CPU-only
                verbose=False,
            )
            logger.info("Local model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using local model."""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "\n\n\n"],
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"


class HostedLLM(LLMProvider):
    """Hosted LLM using OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Using hosted model: {model}")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"Error generating response: {str(e)}"


def get_llm(mode: str, local_path: Optional[str] = None, api_key: Optional[str] = None) -> LLMProvider:
    """Factory function to get appropriate LLM provider."""
    if mode == "local":
        if not local_path:
            raise ValueError("local_path required for local mode")
        return LocalLLM(model_path=local_path)
    elif mode == "hosted":
        if not api_key:
            raise ValueError("api_key required for hosted mode")
        return HostedLLM(api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM mode: {mode}")