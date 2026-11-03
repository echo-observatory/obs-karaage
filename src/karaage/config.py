"""Local settings: which profile to run, and how to reach its backend (docs/design-1.1.0.md, section 4.1).

The profile names the backend; the base URL and the API key are local. Settings
come from KARAAGE_* environment variables, or from a .env file in the current
directory or one of its parents. Keyword arguments override both.
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
    profile: str                          # name of a shipped profile, or path to a profile file
    base_url: str | None                  # None: the default of the profile's backend
    api_key: str | None = field(repr=False)


def _from_env(name: str) -> str | None:
    return os.environ.get(f"KARAAGE_{name.upper()}") or None


def load_settings(**overrides) -> Settings:
    """Read the local settings. Overrides: profile, base_url, api_key."""
    unknown = set(overrides) - {"profile", "base_url", "api_key"}
    if unknown:
        raise TypeError(f"Unknown settings: {', '.join(sorted(unknown))}")
    # usecwd: look for .env from the working directory, not from where the package is installed.
    load_dotenv(find_dotenv(usecwd=True))
    if _from_env("backend"):
        # Since 1.2.0; refused rather than ignored, so that a run can't go to another backend than intended.
        raise ValueError("KARAAGE_BACKEND is no longer used: the profile sets the backend. "
                         "Remove it from the environment or the .env file, and choose a profile instead.")
    values = {name: _from_env(name) for name in ("profile", "base_url", "api_key")}
    values.update({name: value for name, value in overrides.items() if value is not None})
    return Settings(profile=values["profile"] or DEFAULT_PROFILE, base_url=values["base_url"], api_key=values["api_key"])


def connection(settings: Settings, backend: str) -> tuple[str, str]:
    """The base URL and API key to reach `backend`: the settings, or the backend's defaults."""
    base_url = settings.base_url or BACKENDS[backend]
    if not base_url:
        raise ValueError(f"Backend {backend!r} needs a base URL: set KARAAGE_BASE_URL.")
    api_key = settings.api_key or ("lm-studio" if backend == "lmstudio" else None)
    if not api_key:
        raise ValueError(f"Backend {backend!r} needs an API key: set KARAAGE_API_KEY, preferably in a .env file.")
    return base_url, api_key


def profile_from_env() -> str:
    """The profile named in the environment or .env, or the default one."""
    load_dotenv(find_dotenv(usecwd=True))
    return _from_env("profile") or DEFAULT_PROFILE
