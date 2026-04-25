"""Groq LLM integration for TalentScout."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqLLM:
    """Thin wrapper around the Groq chat completion API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 700,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        configured_model = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
        # Map old deprecated default to a supported model.
        if configured_model == "llama3-8b-8192":
            configured_model = "openai/gpt-oss-120b"

        self.model = configured_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_error: Optional[str] = None
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    @property
    def is_available(self) -> bool:
        """Return whether the client is configured with a valid API key."""
        return self.client is not None

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from Groq using the configured LLaMA model."""
        if not self.client:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Please add it to your .env file before running the app."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def create_completion(model_name: str):
            return self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        try:
            completion = create_completion(self.model)
        except Exception as exc:
            error_text = str(exc)
            if "decommissioned" in error_text.lower() or "model_decommissioned" in error_text:
                fallback_model = os.getenv("GROQ_MODEL_FALLBACK") or "openai/gpt-oss-120b"
                if fallback_model != self.model:
                    self.model = fallback_model
                    completion = create_completion(self.model)
                else:
                    raise RuntimeError(
                        "Configured Groq model is decommissioned. "
                        "Set GROQ_MODEL in .env to an active model (example: openai/gpt-oss-120b)."
                    ) from exc
            else:
                raise

        content = completion.choices[0].message.content
        self.last_error = None
        return (content or "").strip()

    def check_connection(self) -> tuple[bool, str]:
        """Validate whether the configured API key can make a real Groq request."""
        if not self.client:
            return False, "GROQ_API_KEY not found. Add it to your .env file."

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with OK"}],
                temperature=0,
                max_tokens=5,
            )
            text = (completion.choices[0].message.content or "").strip().upper()
            self.last_error = None
            if "OK" in text:
                return True, f"Connected to Groq model {self.model}."
            return True, f"Connected to Groq model {self.model}."
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if "decommissioned" in str(exc).lower() or "model_decommissioned" in str(exc):
                self.last_error += (
                    " | Set GROQ_MODEL in .env to a supported model "
                    "(for example: openai/gpt-oss-120b)."
                )
            return False, self.last_error

    def safe_generate_response(
        self,
        prompt: str,
        default_response: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text and fall back to a default response on any API error."""
        try:
            response = self.generate_response(prompt=prompt, system_prompt=system_prompt)
            return response or default_response
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return default_response
