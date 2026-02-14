"""
On-disk LRU cache decorator that persists across process runs.

The cache invalidates automatically when:
- Function arguments change
- Function implementation changes
"""

import functools
import hashlib
import inspect
import json
import pickle
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional, TypedDict, cast


class CacheMetadata(TypedDict):
    lru_order: list[str]
    function_versions: dict[str, str]


class DiskLRUCache:
    """Thread-safe on-disk LRU cache implementation."""

    def __init__(self, cache_dir: Path, max_size: int = 128):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.metadata_file = self.cache_dir / "_metadata.json"
        self.lock = Lock()

        # Load or initialize metadata
        self.metadata: CacheMetadata = self._load_metadata()

    def _load_metadata(self) -> CacheMetadata:
        """Load cache metadata (LRU order, function versions)."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return cast(CacheMetadata, json.load(f))
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "lru_order": [],
            "function_versions": {},
        }

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _compute_key(self, func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Compute cache key from function name and arguments."""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs,
        }
        # Use pickle for consistent serialization, then hash
        key_bytes = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_bytes).hexdigest()

    def _compute_function_hash(self, func: Callable[..., Any]) -> str:
        """Compute hash of function source code."""
        try:
            source = inspect.getsource(func)
            return hashlib.sha256(source.encode()).hexdigest()
        except (OSError, TypeError):
            # If source is not available, use function's __code__ attributes
            code = func.__code__
            code_data = (
                code.co_code,
                code.co_consts,
                code.co_names,
                code.co_varnames,
            )
            return hashlib.sha256(pickle.dumps(code_data)).hexdigest()

    def _update_lru(self, cache_key: str) -> None:
        """Update LRU order - move key to end (most recently used)."""
        lru_order = self.metadata["lru_order"]
        if cache_key in lru_order:
            lru_order.remove(cache_key)
        lru_order.append(cache_key)
        self.metadata["lru_order"] = lru_order

    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache is full."""
        while len(self.metadata["lru_order"]) > self.max_size:
            # Remove least recently used (first in list)
            lru_key = self.metadata["lru_order"].pop(0)
            cache_path = self._get_cache_path(lru_key)
            if cache_path.exists():
                cache_path.unlink()

    def _check_function_version(self, func_name: str, current_hash: str) -> bool:
        """Check if function implementation has changed."""
        stored_hash = self.metadata["function_versions"].get(func_name)
        if stored_hash != current_hash:
            # Function changed - invalidate all cache entries for this function
            self._invalidate_function_cache(func_name)
            self.metadata["function_versions"][func_name] = current_hash
            return False
        return True

    def _invalidate_function_cache(self, func_name: str) -> None:
        """Remove all cache entries for a function."""
        # We can't easily map back from cache keys to functions,
        # so we'll just clear all cache when any function changes
        # A more sophisticated approach would store func_name in metadata per key
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except OSError:
                pass
        self.metadata["lru_order"] = []

    def get(self, func_name: str, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[bool, Any]:
        """
        Get cached value if available and valid.

        Returns:
            (hit, value) where hit is True if cache hit, False otherwise
        """
        with self.lock:
            # Check if function implementation changed
            func_hash = self._compute_function_hash(func)
            self._check_function_version(func_name, func_hash)

            # Compute cache key
            cache_key = self._compute_key(func_name, args, kwargs)
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        value = pickle.load(f)
                    self._update_lru(cache_key)
                    self._save_metadata()
                    return True, value
                except (pickle.PickleError, IOError, EOFError):
                    # Corrupted cache file
                    if cache_path.exists():
                        cache_path.unlink()
                    if cache_key in self.metadata["lru_order"]:
                        self.metadata["lru_order"].remove(cache_key)

            return False, None

    def set(self, func_name: str, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], value: Any) -> None:
        """Store value in cache."""
        with self.lock:
            # Update function version when setting cache
            func_hash = self._compute_function_hash(func)
            if self.metadata["function_versions"].get(func_name) != func_hash:
                self.metadata["function_versions"][func_name] = func_hash

            cache_key = self._compute_key(func_name, args, kwargs)
            cache_path = self._get_cache_path(cache_key)

            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

                self._update_lru(cache_key)
                self._evict_if_needed()
                self._save_metadata()
            except (pickle.PickleError, IOError):
                # Failed to cache - not critical, just continue
                pass

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
            self.metadata = {"lru_order": [], "function_versions": {}}
            self._save_metadata()


def disk_lru_cache(cache_dir: Optional[str] = None, max_size: int = 128) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for on-disk LRU caching with automatic invalidation.

    The cache persists across process runs and automatically invalidates when:
    - Function arguments change
    - Function implementation changes

    Args:
        cache_dir: Directory to store cache files. Defaults to ./.cache/<func_name>
        max_size: Maximum number of cached entries (LRU eviction)

    Example:
        @disk_lru_cache(max_size=100)
        def expensive_function(x, y):
            # Some expensive computation
            return x ** y

        # First call - executes function
        result = expensive_function(2, 10)

        # Second call - returns cached result
        result = expensive_function(2, 10)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Determine cache directory
        nonlocal cache_dir
        if cache_dir is None:
            cache_dir = f".cache/{func.__name__}"

        # Create cache instance for this function
        cache = DiskLRUCache(Path(cache_dir), max_size)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__

            # Try to get from cache
            hit, value = cache.get(func_name, func, args, kwargs)
            if hit:
                return value

            # Cache miss - execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(func_name, func, args, kwargs, result)

            return result

        # Add cache management methods to wrapper
        wrapper.cache_clear = cache.clear  # type: ignore
        wrapper.cache_info = lambda: {  # type: ignore
            "cache_dir": str(cache.cache_dir),
            "max_size": cache.max_size,
            "current_size": len(cache.metadata["lru_order"]),
        }

        return wrapper

    return decorator
