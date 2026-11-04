"""Settings, profiles, request construction and record/replay (no LLM needed)."""

import json
import types

import httpx
import pytest
from openai import InternalServerError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from karaage import llm
from karaage.config import connection, load_settings
from karaage.llm import LLMClient, RecordedError, ReplayMissError
from karaage.profile import load_profile, shipped_profiles

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
CUSTOM_HEADER = CUSTOM_PROFILE.split("[backends")[0]   # without its backend section

LMSTUDIO_SECTION = """\
[backends.lmstudio]
model = "m"
reasoning_off = "none"
format = "mlx"
quantization = "4bit"
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


def test_shipped_profiles():
    assert shipped_profiles() == ["qwen3.5-9b", "qwen3.5-9b-openrouter"]
    local = load_profile("qwen3.5-9b")
    assert (local.label, local.context_tokens, local.temperature, local.seed) == ("qwen3.5-9b", 8000, 0.0, None)
    assert local.to_dict()["backends"] == {"lmstudio": {
        "model": "qwen/qwen3.5-9b", "reasoning_off": "request-parameter", "format": "mlx", "quantization": "4bit"}}
    remote = load_profile("qwen3.5-9b-openrouter")
    assert remote.to_dict()["backends"] == {"openrouter": {
        "model": "qwen/qwen3.5-9b", "reasoning_off": "request-parameter", "quantizations": ["bf16"]}}
    # Otherwise the same, so that the two differ only in where, and in which build, the model runs.
    def rest(profile):
        return {k: v for k, v in profile.to_dict().items() if k not in ("name", "description", "backends")}
    assert rest(remote) == rest(local)


def test_custom_profiles(tmp_path):
    profile = load_profile(write_profile(tmp_path, CUSTOM_PROFILE))
    assert (profile.label, profile.backend, profile.backend_model.model) == ("custom:my-test", "openai", "m")
    with pytest.raises(ValueError, match="reserved"):
        load_profile(write_profile(tmp_path, CUSTOM_PROFILE.replace("my-test", "qwen3.5-9b")))
    with pytest.raises(ValueError, match="Unknown profile keys: sead"):
        load_profile(write_profile(tmp_path, CUSTOM_PROFILE.replace("seed", "sead")))
    with pytest.raises(ValueError, match="No profile 'nope'"):
        load_profile("nope")


@pytest.mark.parametrize("sections, error", [
    ("", "exactly one"),
    (LMSTUDIO_SECTION + '\n[backends.openai]\nmodel = "m"\nreasoning_off = "none"\n', "exactly one"),
    ('[backends.ollama]\nmodel = "m"\nreasoning_off = "none"\n', "Unknown backend"),
    (LMSTUDIO_SECTION.replace('quantization = "4bit"\n', ""), "needs the build's format and quantization"),
    (LMSTUDIO_SECTION + 'quantizations = ["bf16"]\n', r"Unknown keys in \[backends.lmstudio\]: quantizations"),
    ('[backends.openrouter]\nmodel = "m"\nreasoning_off = "none"\n', "needs quantizations or providers"),
])
def test_backend_sections(tmp_path, sections, error):
    with pytest.raises(ValueError, match=error):
        load_profile(write_profile(tmp_path, CUSTOM_HEADER + sections))


def test_settings(monkeypatch):
    settings = load_settings()
    assert (settings.profile, settings.base_url, settings.api_key) == ("qwen3.5-9b", None, None)
    assert "api_key" not in repr(settings)
    assert connection(settings, "lmstudio") == ("http://127.0.0.1:1234/v1", "lm-studio")
    with pytest.raises(ValueError, match="needs an API key"):
        connection(settings, "openrouter")
    with pytest.raises(ValueError, match="needs a base URL"):
        connection(load_settings(api_key="k"), "openai")
    with pytest.raises(TypeError, match="Unknown settings: backend"):
        load_settings(backend="openrouter")

    monkeypatch.setenv("KARAAGE_PROFILE", "qwen3.5-9b-openrouter")
    monkeypatch.setenv("KARAAGE_API_KEY", "from-env")
    settings = load_settings()
    assert settings.profile == "qwen3.5-9b-openrouter"
    assert connection(settings, "openrouter") == ("https://openrouter.ai/api/v1", "from-env")
    assert load_settings(profile="qwen3.5-9b").profile == "qwen3.5-9b"   # arguments override the environment


@pytest.mark.parametrize("source", ["environment", ".env"])
def test_backend_setting_is_refused(source, monkeypatch, tmp_path):
    """Since 1.2.0 the profile sets the backend: an old KARAAGE_BACKEND is an error, not ignored."""
    if source == ".env":
        (tmp_path / ".env").write_text("KARAAGE_BACKEND=openrouter\n")
    else:
        monkeypatch.setenv("KARAAGE_BACKEND", "openrouter")
    with pytest.raises(ValueError, match="the profile sets the backend"):
        load_settings()


def test_dotenv(tmp_path):
    (tmp_path / ".env").write_text("KARAAGE_PROFILE=from-dotenv\n")
    assert load_settings().profile == "from-dotenv"


def test_requests_per_backend(tmp_path):
    sent = sent_request(load_settings(), load_profile("qwen3.5-9b"))
    assert sent["model"] == "qwen/qwen3.5-9b" and sent["temperature"] == 0.0 and sent["max_tokens"] == 5
    assert sent["extra_body"] == {"reasoning_effort": "none"} and "seed" not in sent
    sent = sent_request(load_settings(api_key="k"), load_profile("qwen3.5-9b-openrouter"))
    assert sent["extra_body"] == {"reasoning": {"effort": "none"}, "provider": {
        "require_parameters": True, "data_collection": "deny", "quantizations": ["bf16"]}}

    section = '[backends.openrouter]\nmodel = "m"\nreasoning_off = "none"\nproviders = ["deepinfra"]\n'
    sent = sent_request(load_settings(api_key="k"), load_profile(write_profile(tmp_path, CUSTOM_HEADER + section)))
    assert sent["extra_body"]["provider"] == {"require_parameters": True, "data_collection": "deny", "only": ["deepinfra"]}


@pytest.mark.parametrize("strategy, check", [
    ("no-think-tag", lambda sent: sent["messages"][0]["content"].startswith("/no_think\n")),
    ("prefill-empty-think", lambda sent: sent["messages"][-1] == {"role": "assistant", "content": "<think>\n\n</think>\n\n"}),
    ("none", lambda sent: "extra_body" not in sent and len(sent["messages"]) == 2),
])
def test_reasoning_off_strategies(tmp_path, strategy, check):
    profile = load_profile(write_profile(tmp_path, CUSTOM_PROFILE.replace('"none"', f'"{strategy}"')))
    sent = sent_request(load_settings(base_url="http://x/v1", api_key="k"), profile)
    assert check(sent) and sent["seed"] == 7


def fake_lmstudio(monkeypatch, client, build):
    """Make LM Studio serve the client's model, as `build` (None: its own API can't be read)."""
    models = types.SimpleNamespace(data=[types.SimpleNamespace(id=client.model)])
    client.client = types.SimpleNamespace(models=types.SimpleNamespace(list=lambda: models))
    monkeypatch.setattr(llm, "lmstudio_build", lambda base_url, model: build)


def test_lmstudio_build_is_checked(monkeypatch, capsys):
    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"))
    loaded = {"publisher": "qwen", "format": "mlx", "quantization": "4bit", "loaded": True, "context_length": 8192}
    fake_lmstudio(monkeypatch, client, loaded)
    client.check()
    assert client.model_build == loaded and capsys.readouterr().out == ""

    fake_lmstudio(monkeypatch, client, {**loaded, "format": "gguf", "quantization": "Q4_K_M"})
    with pytest.raises(ValueError, match="as gguf Q4_K_M, but profile 'qwen3.5-9b' is for the mlx 4bit build"):
        client.check()

    fake_lmstudio(monkeypatch, client, None)
    client.check()
    assert "nor checked against the mlx 4bit build" in capsys.readouterr().out


USAGE = {"prompt_tokens": 10, "completion_tokens": 8000, "total_tokens": 8010}


def invented_answer(content, finish_reason="stop", **extra):
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
        "usage": USAGE,
        **extra,
    })


def client_answering(response):
    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"))
    fake_backend(client, response)
    return client


def test_an_answer_cut_off_is_kept_with_its_error():
    client = client_answering(invented_answer('{"a": 1, "b', finish_reason="length"))
    with pytest.raises(ValueError) as raised:
        client.complete_json("S", "U", {"type": "object"}, name="X", max_tokens=8000)
    assert f"{type(raised.value).__name__}: {raised.value}" == "ValueError: The answer was cut off by the token limit (max_tokens=8000)."
    assert llm.answer_of(raised.value) == {"content": '{"a": 1, "b', "finish_reason": "length", "usage": USAGE, "provider": None}


def test_an_empty_answer_is_kept_with_its_error():
    client = client_answering(invented_answer(None))
    with pytest.raises(ValueError, match="The model returned no content") as raised:
        client.complete_json("S", "U", {"type": "object"}, name="X")
    assert llm.answer_of(raised.value)["content"] == ""


def test_an_answer_that_does_not_match_the_schema_is_kept_with_its_error():
    class Result(BaseModel):
        a: int

    client = client_answering(invented_answer('{"a": "not a number"}'))
    with pytest.raises(ValidationError) as raised:
        client.complete_model("S", "U", Result)
    assert f"{type(raised.value).__name__}: {raised.value}".startswith("ValidationError: 1 validation error for Result")
    assert llm.answer_of(raised.value)["content"] == '{"a": "not a number"}'

    # So is one that a step parses itself, through complete_json.
    with pytest.raises(json.JSONDecodeError) as raised:
        client_answering(invented_answer('{"a": ')).complete_json("S", "U", {"type": "object"}, name="X", parse=json.loads)
    assert llm.answer_of(raised.value)["content"] == '{"a": '


def test_an_answer_is_kept_as_returned():
    client = client_answering(invented_answer('<think>Hmm.</think> {"a": ', finish_reason="length"))
    with pytest.raises(ValueError) as raised:
        client.complete_json("S", "U", {"type": "object"}, name="X")
    assert llm.answer_of(raised.value)["content"] == '<think>Hmm.</think> {"a": '


def test_an_openrouter_answer_names_its_provider():
    client = client_answering(invented_answer('{"a"', finish_reason="length", provider="InventedCloud"))
    with pytest.raises(ValueError) as raised:
        client.complete_json("S", "U", {"type": "object"}, name="X")
    assert llm.answer_of(raised.value)["provider"] == "InventedCloud"


def test_an_error_with_no_answer_keeps_none():
    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"))
    def http_error(**kwargs):
        raise InternalServerError("backend down", response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
                                  body=None)
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=http_error)))
    with pytest.raises(InternalServerError) as raised:
        client.complete_json("S", "U", {"type": "object"}, name="X")
    assert llm.answer_of(raised.value) is None
    assert llm.answer_of(RuntimeError("anything else")) is None


def test_an_answer_that_works_is_returned_as_before():
    client = client_answering(invented_answer('<think>Hmm.</think> {"a": 1} '))
    assert client.complete_json("S", "U", {"type": "object"}, name="X") == '{"a": 1}'
    assert client.complete_json("S", "U", {"type": "object"}, name="X", parse=json.loads) == {"a": 1}


def test_record_and_replay(tmp_path):
    recordings = tmp_path / "recordings"
    schema = {"type": "object"}
    answer = ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"a": 1}'}}],
    })

    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"), mode="record", recordings=recordings)
    fake_backend(client, answer)
    assert client.complete_json("S", "U", schema, name="X") == '{"a": 1}'
    assert len(list((recordings / "qwen3.5-9b").glob("*.json"))) == 1

    # Its backend is never called.
    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"), mode="replay", recordings=recordings)
    client.check()
    assert client.complete_json("S", "U", schema, name="X") == '{"a": 1}'
    with pytest.raises(ReplayMissError):
        client.complete_json("S", "another prompt", schema, name="X")

    with pytest.raises(ValueError, match="needs a recordings directory"):
        LLMClient(load_settings(), load_profile("qwen3.5-9b"), mode="replay")


def test_recorded_http_error_is_replayed(tmp_path):
    import httpx
    from openai import BadRequestError
    recordings = tmp_path / "recordings"
    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"), mode="record", recordings=recordings)
    response = httpx.Response(400, request=httpx.Request("POST", "http://x/v1/chat/completions"))
    def create(**kwargs):
        raise BadRequestError("Error code: 400 - context too long", response=response, body=None)
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    with pytest.raises(BadRequestError):
        client.complete_json("S", "U", {"type": "object"}, name="X")

    client = LLMClient(load_settings(), load_profile("qwen3.5-9b"), mode="replay", recordings=recordings)
    with pytest.raises(RecordedError) as replayed:
        client.complete_json("S", "U", {"type": "object"}, name="X")
    assert f"{type(replayed.value).__name__}: {replayed.value}" == "BadRequestError: Error code: 400 - context too long"


@pytest.mark.parametrize("source", ["environment", ".env"])
def test_setup_is_live_whatever_the_environment_says(source, monkeypatch, tmp_path):
    """Only the test suite records or replays (tests/conftest.py): KARAAGE_LLM_MODE doesn't reach setup()."""
    import karaage
    from karaage import llm, segmentation
    monkeypatch.setattr(LLMClient, "check", lambda self: None)   # no backend to reach
    monkeypatch.setattr(karaage, "load_embedding_model", lambda name: None)
    monkeypatch.setattr(segmentation, "load_nlp_pipeline", lambda: None)
    monkeypatch.setattr(llm, "_active", llm._active)   # setup() activates its client; restore the previous one
    if source == ".env":
        (tmp_path / ".env").write_text("KARAAGE_LLM_MODE=replay\nKARAAGE_RECORDINGS=recordings\n")
    else:
        monkeypatch.setenv("KARAAGE_LLM_MODE", "record")
        monkeypatch.setenv("KARAAGE_RECORDINGS", str(tmp_path / "recordings"))
    client = karaage.setup().client
    assert client.mode == "live" and client.recordings is None
    assert not (tmp_path / "recordings").exists()
