"""The single path for LLM calls (docs/design-1.1.0.md, section 4.3).

Every step asks for JSON matching a schema through `complete_json` or
`complete_model`. They use the client activated by `karaage.setup()`, which
applies the profile: model id for the backend, temperature, seed and how
reasoning is turned off.

The client is live unless the test suite makes it record every response in a
recordings directory, or replay those responses instead of calling the backend
(docs/design-1.1.0.md, section 7.1). Only the test suite sets that mode, through
`karaage._setup()`; `karaage.setup()` and `karaage run` are always live.
Responses are grouped in one file per recording, e.g. one per test article.

We call client.chat.completions.create() rather than the SDK's parse helpers,
because those return nothing when a model mixes reasoning tokens with its JSON
(an LM Studio quirk); the JSON is extracted and parsed here instead.

An error raised on an answer that the model returned carries it, for the output to keep
(`answer_of`). Errors keep their type and message: outputs and evals read them as they are.
"""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from openai import APIStatusError, OpenAI
from openai.types.chat import ChatCompletion

from karaage.config import Settings, connection
from karaage.profile import Profile

MAX_RETRIES = 3          # the SDK retries connection errors, timeouts, 429 and 5xx, with exponential backoff
TIMEOUT_SECONDS = 600    # a local model can take minutes on a long prompt

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

LLM_MODES = ("live", "record", "replay")


class ReplayMissError(RuntimeError):
    """Replay mode found no recorded response for a request."""


class RecordedError(RuntimeError):
    """An HTTP error recorded from the backend, raised again in replay mode."""


class LLMClient:
    """An OpenAI-compatible client set up for one profile, on the backend it names."""

    def __init__(self, settings: Settings, profile: Profile, *, mode: str = "live", recordings=None):
        """`mode` and `recordings` are for the test suite: `record` or `replay`, with the recordings'
        root directory. Recordings are kept in its subdirectory named after the profile."""
        if mode not in LLM_MODES:
            raise ValueError(f"The LLM mode must be one of: {', '.join(LLM_MODES)}.")
        if mode != "live" and recordings is None:
            raise ValueError(f"The {mode} mode needs a recordings directory.")
        self.backend = profile.backend
        self.base_url, api_key = connection(settings, self.backend)
        self.profile = profile
        self.backend_model = profile.backend_model
        self.model = self.backend_model.model
        self.reasoning_off = self.backend_model.reasoning_off
        self.client = OpenAI(base_url=self.base_url, api_key=api_key,
                             max_retries=MAX_RETRIES, timeout=TIMEOUT_SECONDS)
        self.model_reported = None   # model id given by the server in its latest response
        self.model_build = None      # LM Studio only: the build that is loaded (lmstudio_build)
        self.providers = set()       # OpenRouter only: providers that served requests, reset per document
        self.mode = mode
        self.recordings = Path(recordings) / profile.label.replace(":", "-") if recordings is not None else None
        self._recording = None       # record mode: name and responses of the file being written
        self._recording_name = None  # replay mode: the recording looked up first
        self._replayed = None        # replay mode: every recording's responses, by recording name and request key
        if self.mode == "record":
            self.use_recording("default")

    def use_recording(self, name: str) -> None:
        """Record the next responses in `<recordings>/<name>.json`, written from scratch.

        In replay mode, responses are looked up in that recording first, then in all the
        others. So when two recordings hold the same request, as an article run with and
        without paraphrasing does, each replays its own response.
        """
        self._recording_name = name
        if self.mode == "record":
            self._recording = (self.recordings / f"{name}.json", {})
            self._recording[0].unlink(missing_ok=True)

    def start_document(self) -> None:
        """Forget the model id and the providers reported for the previous document."""
        self.model_reported = None
        self.providers.clear()

    def check(self) -> None:
        """Check that the backend serves the profile's model, and on LM Studio record and check its build.

        On LM Studio, a build other than the profile's is refused: the outputs would carry
        the profile's name, but answers depend on the build (docs/design-1.1.0.md, section 4.2).
        """
        if self.mode == "replay":
            return
        available = {m.id for m in self.client.models.list().data}
        if self.model not in available:
            raise ValueError(
                f"Model {self.model!r} of profile {self.profile.label!r} is not available "
                f"on {self.backend} at {self.base_url}."
            )
        if self.backend != "lmstudio":
            return
        self.model_build = lmstudio_build(self.base_url, self.model)
        expected = (self.backend_model.format, self.backend_model.quantization)
        if self.model_build is None:
            print(f"⚠️  Could not read the build of {self.model} from LM Studio's API: it will not be recorded, "
                  f"nor checked against the {' '.join(expected)} build of profile {self.profile.label!r}.")
            return
        found = (self.model_build["format"], self.model_build["quantization"])
        if found != expected:
            raise ValueError(
                f"LM Studio has {self.model} as {' '.join(map(str, found))}, but profile {self.profile.label!r} "
                f"is for the {' '.join(expected)} build. Download that build, or use a profile made for this one."
            )
        if not self.model_build["loaded"]:
            print(f"⚠️  {self.model} is not loaded in LM Studio: it would load with LM Studio's default settings. "
                  f"Load it with a context length of at least {self.profile.context_tokens}.")
        elif self.model_build["context_length"] < self.profile.context_tokens:
            print(f"⚠️  {self.model} is loaded with a context length of {self.model_build['context_length']}, "
                  f"below the {self.profile.context_tokens} tokens of profile {self.profile.label!r}.")

    def request(self, system: str, user: str, schema: dict, *, name: str,
                strict: bool = True, max_tokens: int | None = None):
        """Send one chat request asking for JSON that matches `schema`, and return the raw response."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        extra_body = {}
        if self.reasoning_off == "no-think-tag":
            messages[0]["content"] = "/no_think\n" + system
        elif self.reasoning_off == "prefill-empty-think":
            messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
        elif self.reasoning_off == "request-parameter":
            # LM Studio honours only the top-level parameter; OpenRouter documents the object.
            if self.backend == "openrouter":
                extra_body["reasoning"] = {"effort": "none"}
            else:
                extra_body["reasoning_effort"] = "none"
        if self.backend == "openrouter":
            provider = {"require_parameters": True, "data_collection": "deny"}
            if self.backend_model.quantizations:   # the profile fixes the precision (docs/design-1.1.0.md, section 4.2)
                provider["quantizations"] = list(self.backend_model.quantizations)
            if self.backend_model.providers:
                provider["only"] = list(self.backend_model.providers)
            extra_body["provider"] = provider

        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.profile.temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": strict},
            },
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self.profile.seed is not None:
            kwargs["seed"] = self.profile.seed
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self._send(kwargs)
        self.model_reported = response.model
        if getattr(response, "provider", None):   # OpenRouter names the provider that served the request
            self.providers.add(response.provider)
        return response

    def _send(self, kwargs: dict) -> ChatCompletion:
        """Call the backend, or record or replay its response, depending on the mode."""
        if self.mode == "live":
            return self.client.chat.completions.create(**kwargs)
        key = hashlib.sha256(
            json.dumps({"backend": self.backend, **kwargs}, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if self.mode == "replay":
            if self._replayed is None:
                self._replayed = {path.stem: json.loads(path.read_text(encoding="utf-8"))
                                  for path in sorted(self.recordings.glob("*.json"))}
            current = self._replayed.get(self._recording_name, {})
            recorded = current.get(key) or next((r[key] for r in self._replayed.values() if key in r), None)
            if recorded is None:
                raise ReplayMissError(
                    f"No recorded response for this request in {self.recordings}. Re-record when prompts, "
                    "schemas or the profile change (docs/design-1.1.0.md, section 7.1)."
                )
            if "error" in recorded:
                # Raise it under its original class name, so that step errors read as in the recording.
                error = recorded["error"]
                raise type(error["type"], (RecordedError,), {})(error["message"])
            return ChatCompletion.model_validate(recorded)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            # An HTTP error from the backend (e.g. a prompt longer than the context) is part of
            # the recorded behaviour; connection errors and timeouts are not recorded.
            self._record(key, {"error": {"type": type(e).__name__, "message": str(e)}})
            raise
        self._record(key, response.model_dump(mode="json"))
        return response

    def _record(self, key: str, recorded: dict) -> None:
        path, responses = self._recording
        responses[key] = recorded
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(responses, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    def complete_json(self, system: str, user: str, schema: dict, *, name: str,
                      strict: bool = True, max_tokens: int | None = None, parse=None):
        """Ask for JSON that matches `schema` and return the JSON text, or what `parse` makes of it.

        An error raised because the answer is cut off or empty, or by `parse`, carries the answer.
        """
        response = self.request(system, user, schema, name=name, strict=strict, max_tokens=max_tokens)
        choice = response.choices[0]
        # With reasoning on, LM Studio may put the JSON in reasoning_content and leave content empty.
        raw = choice.message.content or getattr(choice.message, "reasoning_content", None) or ""
        content = _THINK_BLOCK.sub("", raw).strip()
        try:
            if choice.finish_reason == "length":
                raise ValueError(f"The answer was cut off by the token limit (max_tokens={max_tokens}).")
            if not content:
                raise ValueError(f"The model returned no content. Full message: {choice.message!r}")
            return parse(content) if parse else content
        except Exception as e:
            e.answer = _answer(response, raw)
            raise

    def complete_model(self, system: str, user: str, model_cls, *,
                       strict: bool = True, max_tokens: int | None = None):
        """Ask for JSON matching a pydantic model and return an instance of it."""
        return self.complete_json(system, user, model_cls.model_json_schema(), name=model_cls.__name__,
                                  strict=strict, max_tokens=max_tokens, parse=model_cls.model_validate_json)

    def provenance(self) -> dict:
        """What the output records about the model (docs/design-1.1.0.md, section 6.3)."""
        return {
            "profile": self.profile.label,
            "profile_settings": self.profile.to_dict(),
            "backend": self.backend,
            "model": self.model,
            "model_reported": self.model_reported,
            # OpenRouter may route each request to a provider running another precision.
            "model_build": self.model_build if self.backend != "openrouter"
                           else {"providers": sorted(self.providers)},
        }


def _answer(response, content: str) -> dict:
    """What the output keeps of an answer that an error was raised on (docs/design-1.1.0.md, section 6.3)."""
    return {
        "content": content,
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump(mode="json", exclude_none=True) if response.usage else None,
        "provider": getattr(response, "provider", None),   # OpenRouter names the provider that served the request
    }


def answer_of(error: Exception) -> dict | None:
    """The answer that `error` was raised on; None for other errors, such as an HTTP error from the backend."""
    return getattr(error, "answer", None)


def lmstudio_build(base_url: str, model: str) -> dict | None:
    """Ask LM Studio's own API which build of `model` it has, and how it is loaded.

    Returns None when that API is not available (older LM Studio, or another server).
    """
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/v1/models", timeout=10) as response:
            models = json.load(response)["models"]
    except Exception:
        return None
    for entry in models:
        if entry.get("key") == model:
            instances = entry.get("loaded_instances") or []
            return {
                "publisher": entry.get("publisher"),
                "format": entry.get("format"),
                "quantization": (entry.get("quantization") or {}).get("name"),
                "loaded": bool(instances),
                "context_length": instances[0].get("config", {}).get("context_length") if instances else None,
            }
    return None


_active: LLMClient | None = None


def activate(client: LLMClient) -> None:
    global _active
    _active = client


def active() -> LLMClient:
    if _active is None:
        raise RuntimeError("No LLM client is set up: call karaage.setup() first.")
    return _active


def complete_json(system: str, user: str, schema: dict, **kwargs) -> str:
    return active().complete_json(system, user, schema, **kwargs)


def complete_model(system: str, user: str, model_cls, **kwargs):
    return active().complete_model(system, user, model_cls, **kwargs)


def context_tokens() -> int:
    """Context size of the active profile, for token budgets that depend on it."""
    return active().profile.context_tokens
