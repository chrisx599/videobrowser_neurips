import yaml
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    # Alternative env-var name to read the API key from (e.g. "GEMINI_API_KEY"
    # when routing Gemini through its OpenAI-compatible endpoint). Takes
    # precedence over `api_key` when set.
    api_key_env: Optional[str] = None
    # When set, forwarded to ChatOpenAI. Explicitly pinning `false` keeps us on
    # the chat.completions endpoint so OpenAI's server-side `web_search` tool
    # (only available on the Responses endpoint) can never be invoked.
    use_responses_api: Optional[bool] = None

class LLMSettings(BaseModel):
    default: LLMConfig
    # Use Dict[str, Any] for overrides to allow partial updates (patching)
    overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("overrides", mode="before")
    @classmethod
    def ensure_dict(cls, v):
        return v or {}

class TranscriptConfig(BaseModel):
    provider: str = "local"
    oxylabs_username: Optional[str] = Field(default_factory=lambda: os.getenv("OXYLABS_USERNAME"))
    oxylabs_password: Optional[str] = Field(default_factory=lambda: os.getenv("OXYLABS_PASSWORD"))
    model: str = "whisper-1"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    language: Optional[str] = None

class WatcherConfig(BaseModel):
    num_frames: int = 10
    video_downloader: str = "ytdlp" # "ytdlp" | "pytubefix"
    max_duration_seconds: Optional[int] = None  # skip videos longer than this; None = no limit
    # builder_stage2_grid / builder_stage2_gate (per-video gate on 2x2 grids)
    num_grids: Optional[int] = None
    frames_per_grid: Optional[int] = None
    grid_cell_size: Optional[int] = None
    # builder_stage2_visual_selector (comparative top-M pick across K candidates)
    frames_per_candidate: Optional[int] = None
    visual_top_m: Optional[int] = None

class AnalystConfig(BaseModel):
    num_frames: int = 128  # frames extracted per video for the final analyst VLM call

class SelectorConfig(BaseModel):
    top_k: int = 5

class CacheConfig(BaseModel):
    enabled: bool = True
    base_dir: str = "data/cache"


class ProxyConfig(BaseModel):
    enabled: bool = True
    protocol: str = "http"           # http | socks5h
    ssh_tunnel: bool = False         # start/stop SSH tunnel for proxy
    use_cookies: bool = True         # pass cookies.txt to yt-dlp; disable for proxy-only auth
    username: Optional[str] = Field(default_factory=lambda: os.getenv("PROXY_USERNAME"))
    password: Optional[str] = Field(default_factory=lambda: os.getenv("PROXY_PASSWORD"))
    pool_path: str = "data/ip_pools.txt"
    validated_pool_path: str = "data/ip_pools_alive.txt"

class OfflineSearchMethodConfig(BaseModel):
    method: str = "bm25"  # "keyword" | "bm25" | "embedding" | "hybrid"
    top_k: int = 10

class OfflineSearchBM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75

class OfflineSearchEmbeddingConfig(BaseModel):
    model_name: str = "models/BAAI/bge-m3"
    batch_size: int = 32

class OfflineSearchHybridConfig(BaseModel):
    children: list[str] = Field(default_factory=lambda: ["bm25", "embedding"])
    fusion: str = "rrf"  # "rrf" | "weighted"
    rrf_k: int = 60
    weights: Dict[str, float] = Field(default_factory=lambda: {"bm25": 0.5, "embedding": 0.5})

class OfflineSearchConfig(BaseModel):
    enabled: bool = False
    pool_path: str = "data/offline_search/pool.jsonl"
    index_dir: str = "data/offline_search/index"
    fields: list[str] = Field(default_factory=lambda: ["title", "description", "channel", "tags", "transcript"])
    default: OfflineSearchMethodConfig = Field(default_factory=OfflineSearchMethodConfig)
    bm25: OfflineSearchBM25Config = Field(default_factory=OfflineSearchBM25Config)
    embedding: OfflineSearchEmbeddingConfig = Field(default_factory=OfflineSearchEmbeddingConfig)
    hybrid: OfflineSearchHybridConfig = Field(default_factory=OfflineSearchHybridConfig)

class SearchConfig(BaseModel):
    text_search_provider: Optional[str] = "tavily"  # "tavily" | "serper" | "duckduckgo" | None
    video_search_provider: str = "youtube"  # "youtube" | "serper" | "duckduckgo" | "offline"
    offline: OfflineSearchConfig = Field(default_factory=OfflineSearchConfig)

class CheckerConfig(BaseModel):
    max_loop_steps: int = 3
    max_search_steps: Optional[int] = None
    max_watch_steps: Optional[int] = None

class PlannerConfig(BaseModel):
    max_queries: int = 3

class MemoryRetrievalConfig(BaseModel):
    proposal_top_n: int = 20
    selection_top_k: int = 3
    include_failure_memories: bool = True
    require_role_match: bool = True

class MemoryEmbeddingConfig(BaseModel):
    provider: str = "sentence_transformers"
    model_name: str = "models/BAAI/bge-m3"

class MemoryBanditConfig(BaseModel):
    algorithm: str = "linucb"
    alpha: float = 1.0
    online_update: bool = True
    model_path: Optional[str] = None
    meta_path: Optional[str] = None
    emb_dim: int = 1024
    hidden_dim: int = 128
    output_dim: int = 64

class MemoryRewardConfig(BaseModel):
    gap_weight: float = 1.0
    focus_weight: float = 0.5
    redundancy_weight: float = 0.7
    cost_weight: float = 0.2

class MemoryCriticConfig(BaseModel):
    enabled: bool = False
    max_tokens: int = 384
    thresholds_path: Optional[str] = None
    low_threshold: float = 0.30
    high_threshold: float = 0.70
    fallback_confidence: float = 0.5
    per_card_drop_threshold: float = 0.25

class MemoryUncertaintyConfig(BaseModel):
    enabled: bool = False
    score_var_weight: float = 0.4
    outcome_entropy_weight: float = 0.3
    embedding_spread_weight: float = 0.3

class MemoryScorerConfig(BaseModel):
    enabled: bool = False
    model_path: Optional[str] = None
    meta_path: Optional[str] = None
    hybrid_alpha: float = 0.5  # final_applicability = alpha*critic + (1-alpha)*scorer
    state_feature_keys: list[str] = Field(default_factory=list)  # empty = use scorer's default

class MemoryTranslationLLMConfig(BaseModel):
    model: str = "Qwen3-VL-8B-Instruct"
    base_url: str = "http://localhost:8025/v1"
    temperature: float = 0.2
    max_tokens: int = 256


class MemoryTranslationConfig(BaseModel):
    enabled: bool = False
    lambda_q: float = 0.5
    lambda_z: float = 0.3
    lambda_t: float = 0.2
    zq_cache_path: str = "data/training_runs/memory_v2c_translation/zq_cache.jsonl"
    zq_llm: MemoryTranslationLLMConfig = Field(default_factory=MemoryTranslationLLMConfig)


class MemoryConfig(BaseModel):
    enabled: bool = False
    ranker: str = "bandit"  # "bandit" | "llm_critic"
    source: Literal["none", "episodic", "skills"] = "episodic"
    skill_bank_path: str = "data/benchmark/skill_bank_rich100_v2.jsonl"
    skill_top_k: int = 3
    skill_max_injected: int = 2
    skill_max_tokens: int = 200
    skill_similarity_floor: float = 0.55
    store_path: str = "data/memory/memory_bank.jsonl"
    embedding_path: str = "data/memory/embeddings.npz"
    bandit_state_path: str = "data/memory/bandit_state.json"
    events_path: str = "data/memory/events.jsonl"
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)
    embedding: MemoryEmbeddingConfig = Field(default_factory=MemoryEmbeddingConfig)
    bandit: MemoryBanditConfig = Field(default_factory=MemoryBanditConfig)
    reward: MemoryRewardConfig = Field(default_factory=MemoryRewardConfig)
    critic: MemoryCriticConfig = Field(default_factory=MemoryCriticConfig)
    uncertainty: MemoryUncertaintyConfig = Field(default_factory=MemoryUncertaintyConfig)
    scorer: MemoryScorerConfig = Field(default_factory=MemoryScorerConfig)
    translation: MemoryTranslationConfig = Field(default_factory=MemoryTranslationConfig)

class LoggerConfig(BaseModel):
    enabled: bool = True
    log_dir: str = "data/logs"

class PromptsConfig(BaseModel):
    analyst_format_instructions: Optional[str] = None

class AppConfig(BaseModel):
    llm: LLMSettings
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    analyst: AnalystConfig = Field(default_factory=AnalystConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    checker: CheckerConfig = Field(default_factory=CheckerConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

_config: Optional[AppConfig] = None

def load_config(config_path: str = "config.yaml") -> AppConfig:
    global _config
    
    # Look for config in project root (assuming execution from root)
    # or relative to this file's parent's parent (if running as package)
    path = Path(config_path)
    if not path.exists():
        # Try to find it relative to the module if not found in cwd
        module_path = Path(__file__).parent.parent.parent / config_path
        if module_path.exists():
            path = module_path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.absolute()}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    _config = AppConfig(**config_data)
    return _config

def get_config() -> AppConfig:
    global _config
    if _config is None:
        return load_config()
    return _config
