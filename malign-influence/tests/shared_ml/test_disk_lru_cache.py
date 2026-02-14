"""Tests for disk_lru_cache decorator."""

import shutil
import tempfile
from pathlib import Path

from src.shared_ml.disk_lru_cache import DiskLRUCache, disk_lru_cache


class TestDiskLRUCache:
    """Test the DiskLRUCache class."""

    def test_basic_caching(self) -> None:
        """Test basic cache functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskLRUCache(Path(tmpdir), max_size=10)

            # Define a simple function
            def add(x: int, y: int) -> int:
                return x + y

            # First call - cache miss
            hit, value = cache.get("add", add, (1, 2), {})
            assert not hit
            assert value is None

            # Store value
            cache.set("add", add, (1, 2), {}, 3)

            # Second call - cache hit
            hit, value = cache.get("add", add, (1, 2), {})
            assert hit
            assert value == 3

    def test_lru_eviction(self) -> None:
        """Test that LRU eviction works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskLRUCache(Path(tmpdir), max_size=3)

            def dummy(x: int) -> int:
                return x

            # Add 3 items (fill cache)
            for i in range(3):
                cache.set("dummy", dummy, (i,), {}, i * 10)

            # Verify all 3 are cached
            for i in range(3):
                hit, value = cache.get("dummy", dummy, (i,), {})
                assert hit
                assert value == i * 10

            # Add one more item - should evict the least recently used (0)
            cache.set("dummy", dummy, (3,), {}, 30)

            # Item 0 should be evicted
            hit, _ = cache.get("dummy", dummy, (0,), {})
            assert not hit

            # Items 1, 2, 3 should still be cached
            for i in range(1, 4):
                hit, value = cache.get("dummy", dummy, (i,), {})
                assert hit
                assert value == i * 10

    def test_lru_update_on_access(self) -> None:
        """Test that accessing an item updates its position in LRU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskLRUCache(Path(tmpdir), max_size=3)

            def dummy(x: int) -> int:
                return x

            # Add 3 items
            for i in range(3):
                cache.set("dummy", dummy, (i,), {}, i * 10)

            # Access item 0 (making it most recently used)
            cache.get("dummy", dummy, (0,), {})

            # Add item 3 (should evict item 1, not 0)
            cache.set("dummy", dummy, (3,), {}, 30)

            # Item 0 should still be cached (was accessed recently)
            hit, value = cache.get("dummy", dummy, (0,), {})
            assert hit
            assert value == 0

            # Item 1 should be evicted
            hit, _ = cache.get("dummy", dummy, (1,), {})
            assert not hit

    def test_function_version_tracking(self) -> None:
        """Test that function implementation changes invalidate cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskLRUCache(Path(tmpdir), max_size=10)

            # Original function
            def func(x: int) -> int:  # type: ignore[no-redef]
                return x * 2

            # Cache a value
            cache.set("func", func, (5,), {}, 10)
            hit, value = cache.get("func", func, (5,), {})
            assert hit
            assert value == 10

            # "Modify" the function by creating a new one with same name
            def func(x: int) -> int:  # noqa: F811
                return x * 3  # Different implementation

            # Cache should be invalidated
            hit, _ = cache.get("func", func, (5,), {})
            assert not hit

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskLRUCache(Path(tmpdir), max_size=10)

            def dummy(x: int) -> int:
                return x

            # Add some items
            for i in range(5):
                cache.set("dummy", dummy, (i,), {}, i * 10)

            # Clear cache
            cache.clear()

            # Metadata should be reset immediately after clear
            assert len(cache.metadata["lru_order"]) == 0
            assert len(cache.metadata["function_versions"]) == 0

            # All items should be gone
            for i in range(5):
                hit, _ = cache.get("dummy", dummy, (i,), {})
                assert not hit

    def test_persistence_across_instances(self) -> None:
        """Test that cache persists across cache instances."""
        with tempfile.TemporaryDirectory() as tmpdir:

            def func(x: int) -> int:
                return x * 2

            # First cache instance
            cache1 = DiskLRUCache(Path(tmpdir), max_size=10)
            cache1.set("func", func, (5,), {}, 10)

            # Create new cache instance with same directory
            cache2 = DiskLRUCache(Path(tmpdir), max_size=10)

            # Should be able to retrieve cached value
            hit, value = cache2.get("func", func, (5,), {})
            assert hit
            assert value == 10

    def test_different_argument_types(self) -> None:
        """Test caching with different argument types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskLRUCache(Path(tmpdir), max_size=10)

            def func(*args: int, **kwargs: int) -> int:
                return sum(args) + sum(kwargs.values())

            # Test with positional args
            cache.set("func", func, (1, 2, 3), {}, 6)
            hit, value = cache.get("func", func, (1, 2, 3), {})
            assert hit
            assert value == 6

            # Test with keyword args
            cache.set("func", func, (), {"a": 1, "b": 2}, 3)
            hit, value = cache.get("func", func, (), {"a": 1, "b": 2})
            assert hit
            assert value == 3

            # Test with mixed args
            cache.set("func", func, (1, 2), {"c": 3}, 6)
            hit, value = cache.get("func", func, (1, 2), {"c": 3})
            assert hit
            assert value == 6


class TestDiskLRUCacheDecorator:
    """Test the disk_lru_cache decorator."""

    def test_basic_decorator_usage(self) -> None:
        """Test basic decorator functionality."""
        call_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def expensive_func(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            # First call - executes function
            result1 = expensive_func(5)
            assert result1 == 10
            assert call_count == 1

            # Second call - uses cache
            result2 = expensive_func(5)
            assert result2 == 10
            assert call_count == 1  # Function not called again

            # Different argument - executes function
            result3 = expensive_func(10)
            assert result3 == 20
            assert call_count == 2

    def test_decorator_with_kwargs(self) -> None:
        """Test decorator with keyword arguments."""
        call_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int, y: int = 10) -> int:
                nonlocal call_count
                call_count += 1
                return x + y

            # Call with default kwarg
            result1 = func(5)
            assert result1 == 15
            assert call_count == 1

            # Same call - should use cache
            result2 = func(5)
            assert result2 == 15
            assert call_count == 1

            # Call with explicit kwarg
            result3 = func(5, y=20)
            assert result3 == 25
            assert call_count == 2

            # Same call - should use cache
            result4 = func(5, y=20)
            assert result4 == 25
            assert call_count == 2

    def test_decorator_cache_clear(self) -> None:
        """Test cache_clear method on decorated function."""
        call_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            # First call
            func(5)
            assert call_count == 1

            # Second call - cached
            func(5)
            assert call_count == 1

            # Clear cache
            func.cache_clear()  # type: ignore

            # Call again - should execute function
            func(5)
            assert call_count == 2

    def test_decorator_cache_info(self) -> None:
        """Test cache_info method on decorated function."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int) -> int:
                return x * 2

            info = func.cache_info()  # type: ignore
            assert info["max_size"] == 10
            assert info["current_size"] == 0
            assert tmpdir in info["cache_dir"]

            # Add some cached values
            func(1)
            func(2)
            func(3)

            info = func.cache_info()  # type: ignore
            assert info["current_size"] == 3

    def test_decorator_persistence(self) -> None:
        """Test that decorated function cache persists across processes."""
        call_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int) -> int:  # type: ignore[no-redef]
                nonlocal call_count
                call_count += 1
                return x * 2

            # First call
            result1 = func(5)
            assert result1 == 10
            assert call_count == 1

            # Store the cached value for later verification
            first_call_count = call_count

            # Simulate new process by creating new decorated function with IDENTICAL source
            # Note: Must have identical source including no comments for hash to match
            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            # Should use cached value since function source is identical
            result2 = func(5)
            assert result2 == 10
            assert call_count == first_call_count  # Function not called again

    def test_decorator_invalidation_on_change(self) -> None:
        """Test that cache invalidates when function implementation changes."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int) -> int:  # type: ignore[no-redef]
                return x * 2

            # Cache a value
            result1 = func(5)
            assert result1 == 10

            # "Change" the function by redefining it
            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int) -> int:  # noqa: F811
                return x * 3  # Different implementation

            # Should compute new value, not use cached
            result2 = func(5)
            assert result2 == 15

    def test_decorator_lru_eviction(self) -> None:
        """Test LRU eviction in decorator."""
        call_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=3)
            def func(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            # Fill cache
            func(1)
            func(2)
            func(3)
            assert call_count == 3

            # Access first item (make it recently used)
            func(1)
            assert call_count == 3  # Cached

            # Add new item - should evict item 2
            func(4)
            assert call_count == 4

            # Item 1 should still be cached
            func(1)
            assert call_count == 4  # Cached

            # Item 2 should be evicted
            func(2)
            assert call_count == 5  # Not cached

    def test_decorator_default_cache_dir(self) -> None:
        """Test decorator with default cache directory."""
        cache_dir = Path(".cache/test_func")

        # Clean up if exists
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        try:

            @disk_lru_cache(max_size=10)
            def test_func(x: int) -> int:
                return x * 2

            # Should create default cache directory
            test_func(5)

            assert cache_dir.exists()
            assert any(cache_dir.glob("*.pkl"))

        finally:
            # Clean up
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

    def test_decorator_with_complex_return_types(self) -> None:
        """Test caching functions that return complex types."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def func(x: int) -> dict[str, int]:
                return {"value": x, "squared": x**2, "cubed": x**3}

            result1 = func(5)
            assert result1 == {"value": 5, "squared": 25, "cubed": 125}

            # Should return cached dict
            result2 = func(5)
            assert result2 == {"value": 5, "squared": 25, "cubed": 125}
            assert result2 is not result1  # Different object (unpickled)

    def test_functools_wraps_preservation(self) -> None:
        """Test that decorator preserves function metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @disk_lru_cache(cache_dir=tmpdir, max_size=10)
            def example_func(x: int) -> int:
                """This is an example function."""
                return x * 2

            assert example_func.__name__ == "example_func"
            assert example_func.__doc__ == "This is an example function."
