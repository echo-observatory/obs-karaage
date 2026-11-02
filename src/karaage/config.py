"""Backend settings: where the LLM runs (docs/design-1.1.0.md, section 4.1).

Settings come from KARAAGE_* environment variables, or from a .env file in the
current directory or one of its parents. Keyword arguments override both.
"""

import os
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv

DEFAULT_PROFILE = "qwen3.5-9b"

# Default base URL per backend; `openai` stands for any other OpenAI-compatible server.
BACKENDS = {
    "lmstudio": "http://127.0.0.1:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": None,
}


@dataclass(frozen=True)
class Settings:
    backend: str
    base_url: str
    api_key: str = field(repr=False)
    profile: str  # name of a shipped profile, or path to a profile file


def _from_env(name: str) -> str | None:
    return os.environ.get(f"KARAAGE_{name.upper()}") or None


def load_settings(**overrides) -> Settings:
    """Read the backend settings. Overrides: backend, base_url, api_key, profile."""
    unknown = set(overrides) - {"backend", "base_url", "api_key", "profile"}
    if unknown:
        raise TypeError(f"Unknown settings: {', '.join(sorted(unknown))}")
    # usecwd: look for .env from the working directory, not from where the package is installed.
    load_dotenv(find_dotenv(usecwd=True))
    values = {name: _from_env(name) for name in ("backend", "base_url", "api_key", "profile")}
    values.update({name: value for name, value in overrides.items() if value is not None})

    backend = values["backend"] or "lmstudio"
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of: {', '.join(BACKENDS)}.")
    base_url = values["base_url"] or BACKENDS[backend]
    if not base_url:
        raise ValueError(f"Backend {backend!r} needs a base URL: set KARAAGE_BASE_URL.")
    api_key = values["api_key"] or ("lm-studio" if backend == "lmstudio" else None)
    if not api_key:
        raise ValueError(f"Backend {backend!r} needs an API key: set KARAAGE_API_KEY, preferably in a .env file.")
    return Settings(backend=backend, base_url=base_url, api_key=api_key,
                    profile=values["profile"] or DEFAULT_PROFILE)


def profile_from_env() -> str:
    """The profile named in the environment or .env, or the default one."""
    load_dotenv(find_dotenv(usecwd=True))
    return _from_env("profile") or DEFAULT_PROFILE
