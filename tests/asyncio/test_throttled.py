from collections.abc import Awaitable, Callable, Coroutine
from functools import partial
from typing import Any, cast

import pytest
from throttled.asyncio import (
    RateLimiterType,
    Throttled,
    exceptions,
    per_sec,
    rate_limiter,
    store,
    types,
    utils,
)
from throttled.asyncio.hooks import Hook
from throttled.asyncio.rate_limiter import GCRARateLimiter
from throttled.rate_limiter import RateLimitResult


@pytest.fixture
def decorated_demo() -> Callable[[int, int], Coroutine[Any, Any, int]]:
    @Throttled(
        key="/api/product",
        using=RateLimiterType.FIXED_WINDOW.value,
        quota=rate_limiter.per_min(1),
        store=store.MemoryStore(),
    )
    async def demo(left: int, right: int) -> int:
        return left + right

    return cast("Callable[[int, int], Coroutine[Any, Any, int]]", demo)


EXPECTED_RESULT = 3
EXPECTED_HOOK_CALL_COUNT = 2


@pytest.mark.asyncio
class TestThrottled:
    @classmethod
    async def test_demo(
        cls, decorated_demo: Callable[[int, int], Coroutine[Any, Any, int]]
    ) -> None:
        assert await decorated_demo(1, 2) == EXPECTED_RESULT
        with pytest.raises(exceptions.LimitedError):
            await decorated_demo(2, 3)

    @classmethod
    async def test_limit__timeout(cls) -> None:
        throttle: Throttled = Throttled(timeout=1, quota=per_sec(1))
        assert (await throttle.limit("key")).limited is False

        def _callback(
            left: float,
            right: float,
            elapsed: types.TimeLikeValueT,
            *args: object,
            **kwargs: object,
        ) -> None:
            assert left <= elapsed < right

        async with utils.Timer(callback=partial(_callback, 1, 2)):
            assert (await throttle.limit("key")).limited is False

        # case: retry_after > timeout
        async with utils.Timer(callback=partial(_callback, 0, 0.1)):
            assert (await throttle.limit("key", cost=2)).limited

        # case: timeout < retry_after
        async with utils.Timer(callback=partial(_callback, 0, 0.1)):
            assert (await throttle.limit("key", timeout=0.5)).limited

    @classmethod
    async def test_enter(cls) -> None:
        construct_kwargs: dict[str, Any] = {
            "key": "key",
            "quota": per_sec(1),
            "store": store.MemoryStore(),
        }
        throttle: Throttled = Throttled(**construct_kwargs)
        async with throttle as rate_limit_result:
            assert rate_limit_result.limited is False

        with pytest.raises(exceptions.LimitedError) as exc_info:
            async with throttle:
                pass
        raised_result = exc_info.value.rate_limit_result
        assert raised_result is not None
        assert raised_result.limited
        assert raised_result.state.remaining == 0
        assert raised_result.state.reset_after == 1
        assert raised_result.state.retry_after == 1

        async with Throttled(**construct_kwargs, timeout=1) as rate_limit_result:
            assert not rate_limit_result.limited

    @classmethod
    async def test_constructor__string_quota(cls) -> None:
        throttle: Throttled = Throttled(
            key="quota-str",
            quota="1/s",
            store=store.MemoryStore(),
        )
        assert not (await throttle.limit()).limited

    @classmethod
    async def test_constructor__reject_multi_rules_string(cls) -> None:
        with pytest.raises(exceptions.DataError, match="multiple quota rules"):
            Throttled(
                key="quota-multi",
                quota="1/s; 10/m",
                store=store.MemoryStore(),
            )

    @classmethod
    async def test_hook__execution_order(cls) -> None:
        """Throttled should execute hooks in correct middleware order.

        hooks=[A, B] should execute: A_before → B_before → limit → B_after → A_after
        """
        order: list[str] = []

        class OrderHook(Hook):
            def __init__(self, name: str) -> None:
                self.name = name

            async def on_limit(self, *args: object, **kwargs: object) -> RateLimitResult:
                call_next = cast("Callable[[], Awaitable[RateLimitResult]]", args[0])
                order.append(f"{self.name}_before")
                result: RateLimitResult = await call_next()
                order.append(f"{self.name}_after")
                return result

        throttle: Throttled = Throttled(
            key="test",
            quota=per_sec(10),
            hooks=[OrderHook("A"), OrderHook("B")],
        )

        await throttle.limit()

        assert order == ["A_before", "B_before", "B_after", "A_after"]

    @classmethod
    async def test_hook__called_once_per_limit_even_with_retry(cls) -> None:
        """Hook should be called once per limit() call, not per retry attempt.

        This test ensures that when timeout/retry is enabled, the hook wraps
        the entire limit() operation (including retries), not each individual
        retry attempt.
        """
        call_count: int = 0

        class CountingHook(Hook):
            async def on_limit(self, *args: object, **kwargs: object) -> RateLimitResult:
                nonlocal call_count
                call_count += 1
                call_next = cast("Callable[[], Awaitable[RateLimitResult]]", args[0])
                return await call_next()

        throttle: Throttled = Throttled(
            key="test",
            quota=per_sec(1),
            timeout=2,  # Enable retry
            hooks=[CountingHook()],
        )

        # First call: allowed immediately
        await throttle.limit()
        # Second call: denied initially, then retries and succeeds after ~1s
        await throttle.limit()

        # Hook should be called exactly 2 times (once per limit() call)
        # NOT more times due to internal retry attempts
        assert call_count == EXPECTED_HOOK_CALL_COUNT


@pytest.mark.asyncio
class TestKeyPrefix:
    @classmethod
    async def test_limit__default_key_prefix(cls) -> None:
        """Keys are stored under ``throttled:v1:<rate limiter type>:`` by default."""
        mem_store: store.MemoryStore = store.MemoryStore()
        throttle: Throttled = Throttled(
            key="key",
            using=RateLimiterType.GCRA.value,
            quota=per_sec(1),
            store=mem_store,
        )
        await throttle.limit()
        assert await mem_store.exists("throttled:v1:gcra:key")

    @classmethod
    async def test_limit__custom_key_prefix(cls) -> None:
        """A provided key_prefix replaces the default throttled: namespace."""
        mem_store: store.MemoryStore = store.MemoryStore()
        throttle: Throttled = Throttled(
            key="key",
            using=RateLimiterType.GCRA.value,
            quota=per_sec(1),
            store=mem_store,
            key_prefix="my-app:rate-limit",
        )
        await throttle.limit()
        assert await mem_store.exists("my-app:rate-limit:v1:gcra:key")
        assert not await mem_store.exists("throttled:v1:gcra:key")

    @classmethod
    async def test_peek__custom_key_prefix(cls) -> None:
        """peek reads the key that limit wrote; a mismatch would look fresh."""
        throttle: Throttled = Throttled(
            key="key",
            using=RateLimiterType.GCRA.value,
            quota=per_sec(1),
            store=store.MemoryStore(),
            key_prefix="my-app:rate-limit",
        )
        await throttle.limit()
        assert (await throttle.peek("key")).remaining == 0

    @classmethod
    @pytest.mark.parametrize("key_prefix", ["", "  ", ":", ":my-app", "my-app:"])
    async def test_constructor__invalid_key_prefix(cls, key_prefix: str) -> None:
        """A blank or colon-edged namespace would produce malformed storage keys."""
        with pytest.raises(exceptions.DataError, match="Invalid key_prefix"):
            Throttled(key_prefix=key_prefix)

    @classmethod
    async def test_limit__subclass_key_prefix_override(cls) -> None:
        """A subclass overriding the legacy ``KEY_PREFIX`` keeps its keys."""

        class _OverridingRateLimiter(GCRARateLimiter):
            KEY_PREFIX = "acme:v1:"

            class Meta:
                type: str = "gcra_prefix_override"

        mem_store: store.MemoryStore = store.MemoryStore()
        throttle: Throttled = Throttled(
            key="key",
            using="gcra_prefix_override",
            quota=per_sec(1),
            store=mem_store,
        )
        await throttle.limit()
        assert await mem_store.exists("acme:v1:gcra_prefix_override:key")

    @classmethod
    async def test_limit__legacy_constructor_signature(cls) -> None:
        """A registered limiter predating key_prefix is still constructible."""

        class _LegacyRateLimiter(GCRARateLimiter):
            class Meta:
                type: str = "gcra_legacy_init"

            def __init__(
                self, quota: rate_limiter.Quota, store: store.BaseStore
            ) -> None:
                super().__init__(quota, store)

        throttle: Throttled = Throttled(
            key="key",
            using="gcra_legacy_init",
            quota=per_sec(1),
            store=store.MemoryStore(),
        )
        assert (await throttle.limit()).limited is False

    @classmethod
    async def test_construct_rate_limiter_directly__invalid_key_prefix(cls) -> None:
        """Validation also covers limiters built without going via Throttled."""
        with pytest.raises(exceptions.DataError, match="Invalid key_prefix"):
            GCRARateLimiter(per_sec(1), store.MemoryStore(), key_prefix="  ")
