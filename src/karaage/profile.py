"""Model profiles: which model, and how it is driven (docs/design-1.1.0.md, section 4.2).

Shipped profiles live in karaage/profiles/ and their names are reserved. A
profile file passed by path is a custom profile; outputs label it `custom:<name>`.
"""

import tomllib
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

# How to turn the model's reasoning off; `request-parameter` is translated per backend in llm.py.
REASONING_OFF = ("none", "no-think-tag", "prefill-empty-think", "request-parameter")

_TOP_KEYS = {"name", "description", "context_tokens", "temperature", "seed", "embedding_model", "backends"}
_BACKEND_KEYS = {"model", "reasoning_off"}


@dataclass(frozen=True)
class BackendModel:
    model: str           # the model id on that backend
    reasoning_off: str   # one of REASONING_OFF


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    context_tokens: int
    temperature: float
    embedding_model: str
    backends: dict[str, BackendModel]
    seed: int | None = None
    custom: bool = False

    @property
    def label(self) -> str:
        """The name recorded in outputs."""
        return f"custom:{self.name}" if self.custom else self.name

    def for_backend(self, backend: str) -> BackendModel:
        if backend not in self.backends:
            raise ValueError(
                f"Profile {self.label!r} has no section for backend {backend!r} "
                f"(it supports: {', '.join(self.backends)})."
            )
        return self.backends[backend]

    def to_dict(self) -> dict:
        return asdict(self)


def shipped_profiles() -> list[str]:
    return sorted(p.name.removesuffix(".toml") for p in files("karaage").joinpath("profiles").iterdir()
                  if p.name.endswith(".toml"))


def load_profile(name_or_path: str | Path) -> Profile:
    """Load a shipped profile by name, or a custom profile from a TOML file."""
    shipped = shipped_profiles()
    if str(name_or_path) in shipped:
        text = files("karaage").joinpath("profiles", f"{name_or_path}.toml").read_text(encoding="utf-8")
        custom = False
    elif Path(name_or_path).is_file():
        text = Path(name_or_path).read_text(encoding="utf-8")
        custom = True
    else:
        raise ValueError(
            f"No profile {str(name_or_path)!r}: not a shipped profile ({', '.join(shipped)}) nor a file."
        )
    profile = _parse(tomllib.loads(text), custom=custom)
    if custom and profile.name in shipped:
        raise ValueError(f"Profile name {profile.name!r} is reserved for the shipped profile; rename the custom one.")
    return profile


def _parse(data: dict, *, custom: bool) -> Profile:
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown profile keys: {', '.join(sorted(unknown))}")
    backends = {}
    for backend, section in data.get("backends", {}).items():
        unknown = set(section) - _BACKEND_KEYS
        if unknown:
            raise ValueError(f"Unknown keys in [backends.{backend}]: {', '.join(sorted(unknown))}")
        if section.get("reasoning_off") not in REASONING_OFF:
            raise ValueError(f"[backends.{backend}] reasoning_off must be one of: {', '.join(REASONING_OFF)}")
        backends[backend] = BackendModel(model=section["model"], reasoning_off=section["reasoning_off"])
    if not backends:
        raise ValueError("A profile needs at least one [backends.<name>] section.")
    return Profile(
        name=data["name"],
        description=data.get("description", ""),
        context_tokens=int(data["context_tokens"]),
        temperature=float(data["temperature"]),
        embedding_model=data["embedding_model"],
        backends=backends,
        seed=data.get("seed"),
        custom=custom,
    )
