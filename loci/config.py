"""Configuration loading: file → env-var → CLI-flag precedence."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Config sections as plain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    models_dir: Path = field(default_factory=lambda: Path("~/.loci/models"))
    raw_knowledge_dir: Path = field(default_factory=lambda: Path("~/.loci/raw"))
    knowledge_db: Path = field(default_factory=lambda: Path("~/.loci/knowledge/main.db"))
    packs_dir: Path = field(default_factory=lambda: Path("~/.loci/packs"))
    context_dir: Path = field(default_factory=lambda: Path("~/.loci/context"))
    bench_logs_dir: Path = field(default_factory=lambda: Path("~/.loci/logs/bench"))
    runtime_logs_dir: Path = field(default_factory=lambda: Path("~/.loci/logs/runtime"))


@dataclass
class ModelsConfig:
    chat: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    chat_low_mem: str = "llama-3.2-1b-instruct-q4_k_m.gguf"
    embedder: str = "bge-small-en-v1.5-q8_0.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    temperature: float = 0.4
    max_tokens: int = 512
    use_mmap: bool = True


@dataclass
class IngestConfig:
    chunk_tokens: int = 512
    chunk_overlap_sentences: int = 1
    embed_batch: int = 16
    spacy_model: str = "en_core_web_sm"
    resolve_coref: bool = True  # cheap within-chunk pronoun coreference


@dataclass
class RetrievalConfig:
    vec_top_k: int = 12
    fts_top_k: int = 12
    context_token_budget: int = 1800
    entity_sim_threshold: float = 0.92
    rrf_k: int = 60


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_peak_rss: bool = True
    jsonl: bool = True


@dataclass
class BenchConfig:
    qna_file: Path = field(default_factory=lambda: Path("bench/qna.json"))
    runs_per_question: int = 3
    rss_sample_hz: int = 10
    judge: str = "claude"
    judge_cmd: str = "claude -p"
    judge_single_prompt: bool = True
    judge_max_chars: int = 150000


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    _provenance: dict[str, str] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_SECTIONS = ("paths", "models", "ingest", "retrieval", "logging", "bench")


def _find_config_file(config_path: str | Path | None) -> Path | None:
    """Locate config file via: argument → LOCI_CONFIG env → cwd → home."""
    if config_path is not None:
        p = Path(config_path).expanduser()
        if not p.exists():
            raise ValueError(f"Config file not found: {p}")
        return p
    env = os.environ.get("LOCI_CONFIG")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise ValueError(f"LOCI_CONFIG path not found: {p}")
        return p
    local = Path("config.toml")
    if local.exists():
        return local
    home = Path("~/.loci/config.toml").expanduser()
    if home.exists():
        return home
    return None


def _coerce(current_val: object, raw: object) -> object:
    """Coerce raw TOML/env value to match the type of the existing field value."""
    if isinstance(current_val, Path):
        return Path(str(raw))
    if isinstance(current_val, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in ("1", "true", "yes")
    if isinstance(current_val, int):
        return int(raw)
    if isinstance(current_val, float):
        return float(raw)
    return raw


def _apply_section(
    section_obj: object,
    section_dict: dict,
    provenance: dict[str, str],
    source: str,
    section_name: str,
) -> None:
    for key, val in section_dict.items():
        if not hasattr(section_obj, key):
            raise ValueError(f"Unknown config key: {section_name}.{key}")
        current = getattr(section_obj, key)
        setattr(section_obj, key, _coerce(current, val))
        provenance[f"{section_name}.{key}"] = source


def _apply_env_overrides(cfg: Config, provenance: dict[str, str]) -> None:
    """Apply LOCI_SECTION__KEY environment variable overrides."""
    for env_key, raw_val in os.environ.items():
        if not env_key.startswith("LOCI_") or "__" not in env_key[5:]:
            continue
        rest = env_key[5:]
        section_name, field_name = rest.split("__", 1)
        section_name = section_name.lower()
        field_name = field_name.lower()
        section_obj = getattr(cfg, section_name, None)
        if section_obj is None or not hasattr(section_obj, field_name):
            continue
        current = getattr(section_obj, field_name)
        setattr(section_obj, field_name, _coerce(current, raw_val))
        provenance[f"{section_name}.{field_name}"] = f"env:{env_key}"


def load(
    config_path: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> Config:
    """Load config with full precedence: defaults → file → env vars → overrides."""
    cfg = Config()
    provenance: dict[str, str] = {}

    found = _find_config_file(config_path)
    if found:
        try:
            with open(found, "rb") as fh:
                raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Invalid TOML in {found}: {exc}") from exc

        for section_name, section_dict in raw.items():
            if not isinstance(section_dict, dict):
                raise ValueError(f"Config section [{section_name}] must be a TOML table")
            section_obj = getattr(cfg, section_name, None)
            if section_obj is None:
                raise ValueError(f"Unknown config section: [{section_name}]")
            _apply_section(section_obj, section_dict, provenance, str(found), section_name)

    _apply_env_overrides(cfg, provenance)

    if overrides:
        for dotkey, val in overrides.items():
            parts = dotkey.split(".", 1)
            if len(parts) != 2:
                continue
            section_name, field_name = parts
            section_obj = getattr(cfg, section_name, None)
            if section_obj is not None and hasattr(section_obj, field_name):
                current = getattr(section_obj, field_name)
                setattr(section_obj, field_name, _coerce(current, val))
                provenance[dotkey] = "cli-flag"

    cfg._provenance = provenance
    return cfg


def expanded(cfg: Config) -> Config:
    """Return cfg with all Path fields expanded (expanduser + resolve)."""
    for attr in vars(cfg.paths):
        val = getattr(cfg.paths, attr)
        if isinstance(val, Path):
            setattr(cfg.paths, attr, val.expanduser())
    return cfg
