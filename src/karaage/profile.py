"""Model profiles: which model, on which backend, and how it is driven (docs/design-1.1.0.md, section 4.2).

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

# The keys of a profile's backend section, per backend. Besides the model, a section fixes
# the build the backend serves where it can: the build changes the answers (section 4.1).
_BACKEND_KEYS = {
    "lmstudio": {"model", "reasoning_off", "format", "quantization"},
    "openrouter": {"model", "reasoning_off", "quantizations", "providers"},
    "openai": {"model", "reasoning_off"},   # any other OpenAI-compatible server; its build can't be read
}


@dataclass(frozen=True)
class BackendModel:
    model: str                              # the model id on that backend
    reasoning_off: str                      # one of REASONING_OFF
    format: str | None = None               # lmstudio: the build's format, as LM Studio reports it (e.g. "mlx")
    quantization: str | None = None         # lmstudio: the build's quantisation, as LM Studio reports it (e.g. "4bit")
    quantizations: tuple[str, ...] | None = None   # openrouter: the precisions providers may serve (e.g. "bf16")
    providers: tuple[str, ...] | None = None       # openrouter: the only providers allowed, by slug

    def to_dict(self) -> dict:
        """The section as in the profile file, without the keys it doesn't set."""
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    context_tokens: int
    temperature: float
    embedding_model: str
    backends: dict[str, BackendModel]   # exactly one: the backend the profile runs on
    seed: int | None = None
    custom: bool = False

    @property
    def label(self) -> str:
        """The name recorded in outputs."""
        return f"custom:{self.name}" if self.custom else self.name

    @property
    def backend(self) -> str:
        """The backend the profile runs on: the name of its only backend section."""
        return next(iter(self.backends))

    @property
    def backend_model(self) -> BackendModel:
        return self.backends[self.backend]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["backends"] = {name: section.to_dict() for name, section in self.backends.items()}
        return data


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
    sections = data.get("backends", {})
    if len(sections) != 1:
        raise ValueError(f"A profile needs exactly one [backends.<name>] section, the backend it runs on "
                         f"({', '.join(_BACKEND_KEYS)}); this one has {len(sections)}.")
    (backend, section), = sections.items()
    if backend not in _BACKEND_KEYS:
        raise ValueError(f"Unknown backend [backends.{backend}]; expected one of: {', '.join(_BACKEND_KEYS)}.")
    unknown = set(section) - _BACKEND_KEYS[backend]
    if unknown:
        raise ValueError(f"Unknown keys in [backends.{backend}]: {', '.join(sorted(unknown))}")
    if section.get("reasoning_off") not in REASONING_OFF:
        raise ValueError(f"[backends.{backend}] reasoning_off must be one of: {', '.join(REASONING_OFF)}")
    if backend == "lmstudio" and not (section.get("format") and section.get("quantization")):
        raise ValueError("[backends.lmstudio] needs the build's format and quantization, as LM Studio reports them.")
    if backend == "openrouter" and not (section.get("quantizations") or section.get("providers")):
        raise ValueError("[backends.openrouter] needs quantizations or providers, "
                         "so that the precision the model runs at is fixed.")
    lists = {key: tuple(section[key]) for key in ("quantizations", "providers") if key in section}
    backend_model = BackendModel(model=section["model"], reasoning_off=section["reasoning_off"],
                                 format=section.get("format"), quantization=section.get("quantization"), **lists)
    return Profile(
        name=data["name"],
        description=data.get("description", ""),
        context_tokens=int(data["context_tokens"]),
        temperature=float(data["temperature"]),
        embedding_model=data["embedding_model"],
        backends={backend: backend_model},
        seed=data.get("seed"),
        custom=custom,
    )
