"""Application configuration.

Centralizes every tunable value (API keys, thresholds, paths) behind a single
Settings object read from .env. Nothing in the rest of the codebase should
read os.environ directly — that would scatter config and make the audit
trail's "what threshold was active" question unanswerable (CLAUDE.md rule 4).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Type-safe application settings, loaded from environment variables / .env.

    Use when: any module needs an API key, threshold, or storage path. Import
    the module-level `settings` singleton below rather than instantiating
    this class again — a second instance could silently diverge if env vars
    change between imports.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM Gateway ---
    GROQ_API_KEY: str = ""
    CLAUDE_API_KEY: str = ""

    # CLAUDE.md section 9 names mixtral-8x7b-32768 / llama2-70b-4096, but Groq
    # decommissioned both Mixtral and Llama-2 from its API in 2024. Defaulting
    # to Groq's current free-tier Llama-3 models instead (flagged per section
    # 2's "say out loud that you're deviating" rule) — override here if Groq's
    # lineup changes again.
    GROQ_MODEL_PRIMARY: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_FALLBACK: str = "llama-3.1-8b-instant"

    # Local Ollama fallback (fallback 2 in the LLM chain) — zero cost, works
    # offline. CLAUDE.md section 9 names "mistral", but this dev machine has
    # llama3.2 pulled instead (`ollama list`), not mistral — defaulting to
    # what's actually available rather than a model that would just fail
    # through to Claude every time. Entirely optional: if Ollama isn't
    # running, this link fails and the chain falls through same as any other.
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Demo-mode-only Claude call (fallback 3) — never on the hot path unless
    # DEMO_MODE=true, so cost is opt-in only. claude-opus-5 per current
    # Anthropic model catalog.
    CLAUDE_MODEL: str = "claude-opus-5"

    # Shared LLM-call tuning, applied by llm_manager.py regardless of which
    # provider ends up serving the request.
    LLM_MAX_TOKENS: int = 1024  # generous enough for PHI/clinical JSON, small enough to stay fast on free tiers
    LLM_TIMEOUT_SECONDS: float = 30.0  # Groq is fast; 30s catches a hung connection without false-failing a slow model
    # Zero, not one — found live during Day 8 system testing (see
    # PROGRESS.md): the Groq SDK's own internal retry honors the server's
    # suggested Retry-After duration from a 429 response, which for a fully
    # exhausted daily quota can be many minutes (observed: "please try
    # again in 31m51s"). A single internal retry in that state doesn't fail
    # fast at all — it can sleep for the full suggested wait before ever
    # giving llm_manager.py's own fallback loop a chance to move to the
    # next provider, which is exactly the "don't hang" outcome this setting
    # was meant to prevent in the first place. Zero retries here means
    # get_response()'s own explicit chain (Groq primary -> Groq fallback ->
    # Ollama) is the only retry/fallback logic in play, and it fails each
    # dead link immediately instead of waiting on it.
    LLM_MAX_RETRIES_PER_PROVIDER: int = 0

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./test.db"

    # --- Mode flags ---
    DEMO_MODE: bool = False  # true = route LLM calls to Claude for interview demos (costs money)
    DEBUG: bool = False  # true = verbose logging; never enable in a deployed instance

    # --- Confidence / quality thresholds ---
    # Presidio's own documentation identifies 0.80 as the point past which false
    # positives drop below ~5% on general text; we inherit that default rather
    # than re-deriving it. See Day 8 system tests for the project's own recall/
    # precision measurement against MIMIC ground truth.
    PHI_CONFIDENCE_THRESHOLD: float = 0.80

    # Below 0.70, the Clinical Agent's LLM output starts including incidental
    # mentions (e.g. "no history of diabetes") as if they were findings.
    CLINICAL_PRIORITY_THRESHOLD: float = 0.70

    # Q&A is intentionally conservative: refusing to answer is preferable to a
    # confident hallucination in a clinical context, so the bar sits above the
    # PHI/clinical thresholds.
    QA_CONFIDENCE_THRESHOLD: float = 0.85

    # Target fraction of non-PHI content retained after de-identification —
    # masking too aggressively destroys clinical usefulness of the note.
    DE_ID_PRESERVATION_TARGET: float = 0.90

    # --- PHI Agent (Day 4) ---
    # spaCy model backing Presidio's NLP engine — "sm" (small) rather than
    # Presidio's own default "lg" (large): far smaller download, and NER
    # quality difference doesn't matter much here since every Presidio
    # candidate gets a second, LLM-based context check before being trusted.
    PHI_SPACY_MODEL: str = "en_core_web_sm"
    # How much surrounding text (characters, each side) gets sent to the LLM
    # when it judges whether a candidate span is genuinely identifying —
    # enough to see the sentence a name/date sits in without ballooning
    # every verification prompt to the full document.
    PHI_CONTEXT_WINDOW_CHARS: int = 60

    # --- Knowledge Agent / UniRAG (Day 6) ---
    # UniRAG (D:\projects\UniRag — a separate, already-built 6-day portfolio
    # project, not built as part of this sprint) is a real FastAPI service
    # with its own hybrid BM25+dense retrieval stack. Its own default port is
    # 8000 — identical to this project's own API (section 14) — so running
    # both side by side locally requires starting UniRAG explicitly on a
    # different port: `cd D:\projects\UniRag && uvicorn app.main:app --port 8001`.
    # See PROGRESS.md Day 6 for the full integration write-up.
    UNIRAG_BASE_URL: str = "http://localhost:8001"
    UNIRAG_TIMEOUT_SECONDS: float = 30.0
    # UniRAG's corpus is shared across every document anyone has ever
    # uploaded through it (including its own 3-document seeded sample corpus
    # about its own retrieval pipeline) — its own README lists "no
    # per-tenant data isolation" as a known limitation. UniRAG's /search has
    # no source-filter parameter, so unirag_connector.py over-fetches this
    # many raw results and filters client-side down to just the current
    # document's own chunks (matched by source filename) before the caller
    # ever sees them — a small k requested directly could be entirely
    # consumed by irrelevant chunks before filtering ever finds a real match.
    UNIRAG_SEARCH_OVERFETCH_K: int = 20
    # Final chunk count fed into the answer-generation prompt after
    # filtering — matches CLAUDE.md section 16 Day 6's search(query, k=5).
    QA_TOP_K: int = 5

    # --- Storage paths ---
    CHROMA_DB_PATH: str = "./data/chroma_db"
    UPLOADS_PATH: str = "./data/uploads"
    TEST_DATA_PATH: str = "./data/test_documents"
    RESULTS_PATH: str = "./data/results"
    # Days 8-9 (system testing): user-labeled PHI ground truth, one JSON file
    # per document in TEST_DATA_PATH — see src/validation/phi_validator.py's
    # module docstring for the exact expected format.
    GROUND_TRUTH_PATH: str = "./data/ground_truth"

    # --- App metadata (not env-configurable, but centralized here so
    # /health and elsewhere don't hardcode a version string separately) ---
    APP_VERSION: str = "0.1.0"

    def ensure_storage_dirs(self) -> None:
        """Create all configured storage directories if they don't exist.

        Returns:
            None.

        Use when: called once at FastAPI startup. Keeps data/ out of git
        (see .gitignore) while guaranteeing the app never crashes on a missing
        directory on first run.
        """
        for path in (self.CHROMA_DB_PATH, self.UPLOADS_PATH, self.TEST_DATA_PATH, self.RESULTS_PATH, self.GROUND_TRUTH_PATH):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()


if __name__ == "__main__":
    # Self-test: confirm settings load and directories can be created without
    # any external dependency (no DB, no network).
    settings.ensure_storage_dirs()
    assert 0.0 <= settings.PHI_CONFIDENCE_THRESHOLD <= 1.0
    assert 0.0 <= settings.CLINICAL_PRIORITY_THRESHOLD <= 1.0
    assert 0.0 <= settings.QA_CONFIDENCE_THRESHOLD <= 1.0
    print(f"Settings loaded OK. DATABASE_URL={settings.DATABASE_URL}, DEMO_MODE={settings.DEMO_MODE}")
