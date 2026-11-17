"""HTTP clients for the dedicated translation and lip-sync GPUs."""

import time

import requests


class Translator:
    def __init__(self, url, model):
        self.url, self.model = url.rstrip("/"), model
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self):
        response = self.session.get(f"{self.url}/models", timeout=(5, 10))
        response.raise_for_status()
        if self.model not in {item["id"] for item in response.json()["data"]}:
            raise RuntimeError(f"vLLM does not serve {self.model}")

    def __call__(self, prompt):
        response = self.session.post(f"{self.url}/chat/completions", json={
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Translate faithfully. Output only the translated spoken words, without commentary."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=(10, 300))
        response.raise_for_status()
        choice = response.json()["choices"][0]
        text = choice["message"]["content"]
        if choice["finish_reason"] != "stop" or not text or not text.strip():
            raise RuntimeError(f"Translation did not finish: {choice['finish_reason']}")
        return text.strip()

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
