"""Backend settings, profiles, request construction and record/replay (no LLM needed)."""

import types

import pytest
from openai.types.chat import ChatCompletion

from karaage.config import load_settings
from karaage.llm import LLMClient, ReplayMissError
from karaage.profile import load_profile

CUSTOM_PROFILE = """\
name = "my-test"
context_tokens = 16000
temperature = 0
seed = 7
embedding_model = "x"

[backends.openai]
model = "m"
reasoning_off = "none"
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No KARAAGE_* variables and no .env file: each test sets what it needs."""
    for name in ("BACKEND", "BASE_URL", "API_KEY", "PROFILE", "LLM_MODE", "RECORDINGS"):
        # setenv first, so that teardown restores the original state even if load_dotenv sets the variable.
        monkeypatch.setenv(f"KARAAGE_{name}", "")
        monkeypatch.delenv(f"KARAAGE_{name}")
    monkeypatch.chdir(tmp_path)


def write_profile(tmp_path, text):
    path = tmp_path / "profile.toml"
    path.write_text(text)
    return str(path)


def fake_backend(client, response):
    """Replace the client's backend: capture each request and answer with `response`."""
    sent = {}
    def create(**kwargs):
        sent.update(kwargs)
        return response
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return sent


def sent_request(settings, profile):
    client = LLMClient(settings, profile)
    sent = fake_backend(client, types.SimpleNamespace(model="m", choices=[]))
    client.request("SYS", "USER", {"type": "object"}, name="X", max_tokens=5)
    return sent


def test_default_profile():
    profile = load_profile("qwen3.5-9b")
    assert (profile.label, profile.context_tokens, profile.temperature, profile.seed) == ("qwen3.5-9b", 8000, 0.0, None)
    assert set(profile.backends) == {"lmstudio", "openrouter"}


def test_custom_profiles(tmp_path):
    assert load_profile(write_profile(tmp_path, CUSTOM_PROFILE)).label == "custom:my-test"
    with pytest.raises(ValueError, match="reserved"):
        load_profile(write_profile(tmp_path, CUSTOM_PROFILE.replace("my-test", "qwen3.5-9b")))
    with pytest.raises(ValueError, match="Unknown profile keys: sead"):
        load_profile(write_profile(tmp_path, CUSTOM_PROFILE.replace("seed", "sead")))
    with pytest.raises(ValueError, match="No profile 'nope'"):
        load_profile("nope")
    with pytest.raises(ValueError, match="no section for backend 'openai'"):
        load_profile("qwen3.5-9b").for_backend("openai")


def test_settings(monkeypatch):
    settings = load_settings()
    assert (settings.backend, settings.base_url, settings.api_key, settings.profile) == (
        "lmstudio", "http://127.0.0.1:1234/v1", "lm-studio", "qwen3.5-9b")
    assert "api_key" not in repr(settings)
    with pytest.raises(ValueError, match="needs an API key"):
        load_settings(backend="openrouter")
    with pytest.raises(ValueError, match="needs a base URL"):
        load_settings(backend="openai", api_key="k")
    with pytest.raises(TypeError, match="Unknown settings"):
        load_settings(model="x")

    monkeypatch.setenv("KARAAGE_BACKEND", "openrouter")
    monkeypatch.setenv("KARAAGE_API_KEY", "from-env")
    settings = load_settings()
    assert (settings.backend, settings.base_url, settings.api_key) == ("openrouter", "https://openrouter.ai/api/v1", "from-env")
    assert load_settings(backend="lmstudio").backend == "lmstudio"   # arguments override the environment


def test_dotenv(tmp_path):
    (tmp_path / ".env").write_text("KARAAGE_PROFILE=from-dotenv\n")
    assert load_settings().profile == "from-dotenv"


def test_requests_per_backend():
    profile = load_profile("qwen3.5-9b")
    sent = sent_request(load_settings(backend="lmstudio"), profile)
    assert sent["model"] == "qwen/qwen3.5-9b" and sent["temperature"] == 0.0 and sent["max_tokens"] == 5
    assert sent["extra_body"] == {"reasoning_effort": "none"} and "seed" not in sent
    sent = sent_request(load_settings(backend="openrouter", api_key="k"), profile)
    assert sent["extra_body"] == {"reasoning": {"effort": "none"},
                                  "provider": {"require_parameters": True, "data_collection": "deny"}}


@pytest.mark.parametrize("strategy, check", [
    ("no-think-tag", lambda sent: sent["messages"][0]["content"].startswith("/no_think\n")),
    ("prefill-empty-think", lambda sent: sent["messages"][-1] == {"role": "assistant", "content": "<think>\n\n</think>\n\n"}),
    ("none", lambda sent: "extra_body" not in sent and len(sent["messages"]) == 2),
])
def test_reasoning_off_strategies(tmp_path, strategy, check):
    profile = load_profile(write_profile(tmp_path, CUSTOM_PROFILE.replace('"none"', f'"{strategy}"')))
    sent = sent_request(load_settings(backend="openai", base_url="http://x/v1", api_key="k"), profile)
    assert check(sent) and sent["seed"] == 7


def test_record_and_replay(monkeypatch, tmp_path):
    monkeypatch.setenv("KARAAGE_RECORDINGS", str(tmp_path / "recordings"))
    schema = {"type": "object"}
    answer = ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"a": 1}'}}],
    })

    monkeypatch.setenv("KARAAGE_LLM_MODE", "record")
    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"))
    fake_backend(client, answer)
    assert client.complete_json("S", "U", schema, name="X") == '{"a": 1}'
    assert len(list((tmp_path / "recordings" / "qwen3.5-9b").glob("*.json"))) == 1

    monkeypatch.setenv("KARAAGE_LLM_MODE", "replay")
    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"))   # its backend is never called
    client.check()
    assert client.complete_json("S", "U", schema, name="X") == '{"a": 1}'
    with pytest.raises(ReplayMissError):
        client.complete_json("S", "another prompt", schema, name="X")
