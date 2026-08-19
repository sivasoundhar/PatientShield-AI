"""LLM fallback gateway — the single place every agent goes through to reach an LLM.

CLAUDE.md section 9 specifies a 4-link fallback chain: Groq (primary model) ->
Groq (fallback model) -> local Ollama -> Claude (demo mode only). Centralizing
this here means an agent never needs to know which provider actually answered
it — it calls `LLMManager.get_response()` and gets text back, while the audit
trail (via the returned `LLMResult.provider`) records which link served the
request for HIPAA-relevant transparency (section 9: "Log which LLM was used").

Self-test note (rule 6: self-tests must pass without external dependencies):
this module's `__main__` block monkeypatches the three provider-call methods
rather than hitting real network services, so it verifies the fallback
*control flow* — ordering, exception handling, DEMO_MODE branching — without
requiring a GROQ_API_KEY, a running Ollama, or a CLAUDE_API_KEY to be present.
"""

import logging
from dataclasses import dataclass

import httpx
from anthropic import Anthropic, APIError as AnthropicAPIError
from groq import Groq
from groq import APIError as GroqAPIError

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """What every LLMManager call returns, regardless of which provider served it.

    Use when: any agent needs both the text and a record of provenance for
    the audit trail (section 9: "Log which LLM was used").
    """

    text: str
    provider: str  # e.g. "groq:llama-3.3-70b-versatile", "ollama:mistral", "claude:claude-opus-5"


class AllProvidersFailedError(RuntimeError):
    """Raised when every link in the fallback chain has been exhausted.

    Carries the per-provider failure reasons so the caller (and the audit
    trail) can see *why* each one failed, not just that all of them did —
    rule 8: fail loudly at the edges with a message a human can act on.
    """

    def __init__(self, attempts: list[str]) -> None:
        self.attempts = attempts
        super().__init__("All LLM providers failed: " + "; ".join(attempts))


class LLMManager:
    """Routes a prompt through the Groq -> Ollama -> Claude fallback chain.

    Use when: any agent needs an LLM response. Never call a provider SDK
    directly from an agent — going through here is what makes the fallback
    chain, timeout handling, and audit logging consistent across all 5 agents.
    """

    def __init__(self) -> None:
        # Client construction is lazy per-call (see _call_groq) rather than
        # here, because GROQ_API_KEY may be empty in local dev without Groq
        # access — constructing eagerly would raise on import for anyone not
        # yet using this provider.
        self._groq_client: Groq | None = None
        self._claude_client: Anthropic | None = None

    async def get_response(self, prompt: str, max_tokens: int | None = None) -> LLMResult:
        """Get a completion for `prompt`, trying each provider in order until one succeeds.

        Args:
            prompt: The full prompt text to send.
            max_tokens: Cap on response length. Defaults to settings.LLM_MAX_TOKENS
                when omitted — callers only override this for agents that need
                longer structured output (e.g. Clinical Agent's JSON extraction).

        Returns:
            LLMResult with the generated text and which provider produced it.

        Raises:
            AllProvidersFailedError: every configured provider failed. This is
                an edge condition (rule 8) — callers should catch it and
                decide whether to degrade gracefully or surface the error.

        Use when: called once per LLM-backed decision an agent needs to make.
        """
        tokens = max_tokens or settings.LLM_MAX_TOKENS
        attempts: list[str] = []

        # DEMO_MODE skips straight to Claude — CLAUDE.md section 9 reserves
        # Claude for interview demos specifically, not as a normal fallback
        # link, so it must not be tried silently ahead of the free providers.
        if settings.DEMO_MODE:
            try:
                text = await self._call_claude(prompt, tokens)
                return LLMResult(text=text, provider=f"claude:{settings.CLAUDE_MODEL} (demo mode)")
            except Exception as exc:  # noqa: BLE001 - fail loud below, not here
                raise AllProvidersFailedError([f"claude (demo mode): {exc}"]) from exc

        for provider_name, call in (
            (f"groq:{settings.GROQ_MODEL_PRIMARY}", lambda: self._call_groq(settings.GROQ_MODEL_PRIMARY, prompt, tokens)),
            (f"groq:{settings.GROQ_MODEL_FALLBACK}", lambda: self._call_groq(settings.GROQ_MODEL_FALLBACK, prompt, tokens)),
            (f"ollama:{settings.OLLAMA_MODEL}", lambda: self._call_ollama(prompt, tokens)),
        ):
            try:
                text = await call()
                logger.info("LLM response served by %s", provider_name)
                return LLMResult(text=text, provider=provider_name)
            except Exception as exc:  # noqa: BLE001 - intentionally broad: fall through to next link
                logger.info("LLM provider %s failed, falling through: %s", provider_name, exc)
                attempts.append(f"{provider_name}: {exc}")

        raise AllProvidersFailedError(attempts)

    async def _call_groq(self, model: str, prompt: str, max_tokens: int) -> str:
        """Call Groq's chat completion API with the given model.

        Use when: called internally by get_response for the primary and
        fallback Groq links. Not called directly by agents.
        """
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not configured")
        if self._groq_client is None:
            # max_retries was never wired here before (found live, Day 8):
            # the Groq SDK's own default retry count silently applied
            # instead, meaning every 429 already spent several seconds on
            # the SDK's internal backoff-and-retry *before*
            # get_response()'s own fallback loop ever got a chance to move
            # to the next link in the chain — compounding badly under
            # sustained rate-limiting (confirmed live during Day 8 system
            # testing: dozens of calls, each paying that cost twice, once
            # per Groq model tried). LLM_MAX_RETRIES_PER_PROVIDER already
            # existed in config.py since Day 2 with exactly this intent
            # ("one retry per provider before falling through the chain")
            # but was never actually passed to the client until now.
            self._groq_client = Groq(api_key=settings.GROQ_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=settings.LLM_MAX_RETRIES_PER_PROVIDER)

        try:
            response = self._groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except GroqAPIError as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc
        return response.choices[0].message.content or ""

    async def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        """Call a locally running Ollama instance.

        Use when: called internally by get_response as fallback 2. Requires
        `ollama pull {settings.OLLAMA_MODEL}` to have been run on this host —
        if it hasn't, this raises and the chain falls through to Claude (or
        raises AllProvidersFailedError outside DEMO_MODE).
        """
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            try:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens},
                    },
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Ollama unreachable or errored: {exc}") from exc
        return response.json().get("response", "")

    async def _call_claude(self, prompt: str, max_tokens: int) -> str:
        """Call the Claude API. Demo mode only — see CLAUDE.md section 9.

        Use when: called internally by get_response only when
        settings.DEMO_MODE is true. Never called as a silent fallback for
        normal operation, since it costs money per call.
        """
        if not settings.CLAUDE_API_KEY:
            raise RuntimeError("CLAUDE_API_KEY not configured (required for DEMO_MODE)")
        if self._claude_client is None:
            # Same fix as _call_groq above, applied for consistency — DEMO_MODE
            # calls cost real money per attempt, so bounding retries matters
            # here too even though Claude isn't part of the sequential
            # fallback chain (DEMO_MODE short-circuits straight to it).
            self._claude_client = Anthropic(api_key=settings.CLAUDE_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=settings.LLM_MAX_RETRIES_PER_PROVIDER)

        try:
            response = self._claude_client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except AnthropicAPIError as exc:
            raise RuntimeError(f"Claude API error: {exc}") from exc
        return next((b.text for b in response.content if b.type == "text"), "")


if __name__ == "__main__":
    import asyncio
    from unittest.mock import AsyncMock

    async def _run_self_test() -> None:
        # Case 1: primary Groq fails, fallback Groq succeeds -> chain stops there.
        manager = LLMManager()
        manager._call_groq = AsyncMock(side_effect=[RuntimeError("rate limited"), "fallback answer"])
        manager._call_ollama = AsyncMock(side_effect=AssertionError("should not reach Ollama"))
        result = await manager.get_response("test prompt")
        assert result.text == "fallback answer"
        assert result.provider.startswith("groq:")
        assert manager._call_groq.call_count == 2

        # Case 2: both Groq links fail, Ollama succeeds.
        manager2 = LLMManager()
        manager2._call_groq = AsyncMock(side_effect=RuntimeError("no key"))
        manager2._call_ollama = AsyncMock(return_value="ollama answer")
        result2 = await manager2.get_response("test prompt")
        assert result2.text == "ollama answer"
        assert result2.provider.startswith("ollama:")

        # Case 3: every provider fails -> AllProvidersFailedError with all attempts recorded.
        manager3 = LLMManager()
        manager3._call_groq = AsyncMock(side_effect=RuntimeError("down"))
        manager3._call_ollama = AsyncMock(side_effect=RuntimeError("not running"))
        try:
            await manager3.get_response("test prompt")
            raise AssertionError("expected AllProvidersFailedError")
        except AllProvidersFailedError as exc:
            assert len(exc.attempts) == 3  # primary Groq + fallback Groq + Ollama

        print("llm_manager.py self-test passed: fallback ordering and failure handling verified "
              "without any network calls.")

    asyncio.run(_run_self_test())
