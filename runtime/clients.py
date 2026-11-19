"""HTTP clients for local services and configurable translation/TTS APIs."""

import os
import time

import requests


def api_session(api_key_env, use_environment=False):
    session = requests.Session()
    session.trust_env = use_environment
    token = os.environ.get(api_key_env) if api_key_env else None
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


class Translator:
    def __init__(self, url, model, api_key_env="LIGHTDUBBER_LLM_API_KEY", backend="lightllm", options=None):
        self.url, self.model, self.backend = url.rstrip("/"), model, backend
        self.session = api_session(api_key_env, use_environment=backend == "openai_compatible")
        self.options = {}
        if backend in ("lightllm", "vllm"):
            self.options.update(temperature=0, max_tokens=1024,
                                chat_template_kwargs={"enable_thinking": False})
        self.options.update(options or {})

    def health(self):
        response = self.session.get(f"{self.url}/models", timeout=(5, 10))
        response.raise_for_status()
        if self.model not in {item["id"] for item in response.json()["data"]}:
            raise RuntimeError(f"Translation service does not serve {self.model}")

    def __call__(self, prompt, *,
                 system="Translate faithfully. Output only the translated spoken words, without commentary.",
                 allow_partial=False):
        payload = {
            **self.options,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = self.session.post(f"{self.url}/chat/completions", json=payload, timeout=(10, 300))
        response.raise_for_status()
        choice = response.json()["choices"][0]
        text = choice["message"]["content"]
        if allow_partial and choice["finish_reason"] == "length":
            return (text or "").strip()
        if choice["finish_reason"] != "stop" or not text or not text.strip():
            raise RuntimeError(f"Translation did not finish: {choice['finish_reason']}")
        return text.strip()

    def close(self):
        self.session.close()


class SpeechSynthesizer:
    """OpenAI-compatible audio/speech endpoint with a configured preset voice."""

    def __init__(self, url, model, voice, api_key_env="LIGHTDUBBER_TTS_API_KEY"):
        self.url, self.model, self.voice = url.rstrip("/"), model, voice
        self.session = api_session(api_key_env, use_environment=True)

    def __call__(self, text):
        response = self.session.post(f"{self.url}/audio/speech", json={
            "model": self.model, "input": text, "voice": self.voice, "response_format": "wav",
        }, timeout=(10, 300))
        response.raise_for_status()
        return response.content

    def close(self):
        self.session.close()


class LipSync:
    def __init__(self, url, timeout=1800):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self):
        response = self.session.get(f"{self.url}/health", timeout=(5, 10))
        response.raise_for_status()

    def submit(self, payload):
        # POST is deliberately not retried: its task may already be queued.
        response = self.session.post(f"{self.url}/v1/tasks/video/", json=payload, timeout=(10, 30))
        response.raise_for_status()
        return response.json()

    def wait(self, task_id):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            response = self.session.get(f"{self.url}/v1/tasks/{task_id}/status", timeout=(10, 30))
            response.raise_for_status()
            state = response.json()
            if state["status"] == "completed":
                return state
            if state["status"] in ("failed", "cancelled", "canceled", "stopped"):
                raise RuntimeError(f"TBDub task {task_id}: {state}")
            time.sleep(2)
        raise TimeoutError(f"TBDub task {task_id} is still pending; inspect it before resubmitting")

    def close(self):
        self.session.close()
