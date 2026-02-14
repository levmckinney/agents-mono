"""Configuration utilities for inspect_ai settings."""

from inspect_ai.model import GenerateConfig

_max_connections: int = 10


def set_max_connections(n: int) -> None:
    """Set the global default max concurrent connections for inspect_ai model APIs.

    Args:
        n: Maximum number of concurrent connections to allow.
    """
    global _max_connections
    _max_connections = n


def get_max_connections() -> int:
    """Get the current default max concurrent connections setting."""
    return _max_connections


def get_generate_config(**kwargs) -> GenerateConfig:
    """Get a GenerateConfig with the global max_connections setting applied.

    Any additional kwargs are passed to GenerateConfig and can override max_connections.
    """
    config_kwargs = {"max_connections": _max_connections}
    config_kwargs.update(kwargs)
    return GenerateConfig(**config_kwargs)
