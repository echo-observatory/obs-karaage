from __future__ import annotations

_client = None


def set_client(client) -> None:
    global _client
    _client = client


def get_client():
    if _client is None:
        raise RuntimeError(
            "No LLM client configured. "
            "Call karaage.create_llm_client(...) or karaage.set_client(...) first."
        )
    return _client


def llm_create(*args, max_retries: int = 2, **kwargs):
    """Call client.chat.completions.create() with automatic retry on failure.

    Retries up to max_retries times on any exception. A warning is printed on
    each retry because failures are unexpected and the user should know.
    """
    client = get_client()
    last_exc = None
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        try:
            return client.chat.completions.create(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                print(f"⚠️  LLM call failed (attempt {attempt + 1}/{total_attempts}): {exc}. Retrying…")
    raise last_exc
