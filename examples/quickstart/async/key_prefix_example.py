import asyncio

from throttled.asyncio import RateLimiterType, Throttled, store

mem_store = store.MemoryStore()

# 🌟 Replace the default `throttled` key namespace with your own.
throttle = Throttled(
    key="/api/products",
    using=RateLimiterType.GCRA.value,
    quota="60/m",
    store=mem_store,
    key_prefix="my-app:rate-limit",
)


async def demo() -> None:
    await throttle.limit()
    # The storage schema version and rate limiter type are still appended
    # after the namespace.
    # >> True
    print(await mem_store.exists("my-app:rate-limit:v1:gcra:/api/products"))


if __name__ == "__main__":
    asyncio.run(demo())
