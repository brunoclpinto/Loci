"""Loci CLI — main typer application."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import psutil
import typer
from rich.console import Console
from rich.table import Table

from loci import config as cfg_module
from loci.bench import measure

app = typer.Typer(name="loci", help="Local knowledge assistant.", no_args_is_help=True)
config_app = typer.Typer(help="Manage configuration.")
entities_app = typer.Typer(help="Entity management.")
pack_app = typer.Typer(help="Knowledge-pack management.")
synonyms_app = typer.Typer(help="Predicate synonym management.")
bench_app = typer.Typer(help="Benchmark and evaluation suite.")
facts_app = typer.Typer(help="Fact index management.")
app.add_typer(config_app, name="config")
app.add_typer(entities_app, name="entities")
app.add_typer(pack_app, name="pack")
app.add_typer(synonyms_app, name="synonyms")
app.add_typer(bench_app, name="bench")
app.add_typer(facts_app, name="facts")

console = Console()

_EXAMPLE_CONFIG = Path(__file__).parent.parent / "config.example.toml"
_DEFAULT_CONFIG = Path("~/.loci/config.toml")


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config."),
) -> None:
    """Copy config.example.toml to ~/.loci/config.toml."""
    if not _EXAMPLE_CONFIG.exists():
        console.print(f"[red]config.example.toml not found at {_EXAMPLE_CONFIG}[/red]")
        raise typer.Exit(1)
    dest = _DEFAULT_CONFIG.expanduser()
    if dest.exists() and not force:
        console.print(
            f"[yellow]Config already exists at {dest}. Use --force to overwrite.[/yellow]"
        )
        raise typer.Exit(0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_EXAMPLE_CONFIG.read_text())
    console.print(f"[green]Config written to {dest}[/green]")


@config_app.command("show")
def config_show(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.toml."
    ),
) -> None:
    """Print the effective merged configuration and where each value came from."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    table = Table(title="Effective Configuration", show_header=True, header_style="bold")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_column("Source", style="dim")

    for section_name in cfg_module._SECTIONS:
        section_obj = getattr(cfg, section_name)
        for field_name in vars(section_obj):
            key = f"{section_name}.{field_name}"
            val = getattr(section_obj, field_name)
            source = cfg._provenance.get(key, "default")
            table.add_row(key, str(val), source)

    console.print(table)


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Question to answer."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    explain: bool = typer.Option(False, "--explain", help="Print parse, SQL hits, and fusion ranking."),
    low_mem: bool = typer.Option(False, "--low-mem", help="Use 1B model with 2048 ctx."),
) -> None:
    """Ask a question and get a grounded answer with source citations."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db
    from loci.retrieve import retrieve
    from loci.generate import (
        build_messages, stream_response, extract_cited_tags,
        build_sources_footer, is_refusal,
    )

    cfg_exp = expanded(cfg)

    with measure("ask", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
        try:
            conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
        except Exception as exc:
            console.print(f"[red]Cannot open DB:[/red] {exc}")
            raise typer.Exit(1)

        try:
            # Attach registered packs
            from loci.pack import attach_registered_packs, pack_schemas as get_pack_schemas
            p_schemas = attach_registered_packs(conn, cfg_exp.paths.packs_dir)

            # --- Retrieval phase (embedder loaded then freed) ---
            embedder = None
            _emb_ctx = None
            if _embedder_model_exists(cfg_exp):
                try:
                    from loci.models import load_embedder
                    _emb_ctx = load_embedder(cfg_exp)
                    embedder = _emb_ctx.__enter__()
                except ImportError:
                    pass

            try:
                import spacy
                nlp = spacy.load(cfg.ingest.spacy_model, disable=["ner", "senter"])
            except Exception:
                nlp = None

            try:
                result = retrieve(question, conn=conn, cfg=cfg, embedder=embedder,
                                  nlp=nlp, explain=explain, pack_schemas=p_schemas)
            finally:
                if _emb_ctx is not None:
                    try:
                        _emb_ctx.__exit__(None, None, None)
                    except Exception:
                        pass

            counters["fact_hits"] = len(result.fact_hits)
            counters["chunk_hits"] = len(result.chunk_hits)

            if explain and result.explain_text:
                console.print(result.explain_text)
                console.print()

            # --- Generation phase ---
            if not result.fact_hits and not result.chunk_hits:
                console.print("[yellow]Not in my knowledge base.[/yellow]")
                conn.close()
                return

            if not _chat_model_exists(cfg_exp):
                console.print(
                    "[yellow]Chat model not found — showing raw context.[/yellow]\n"
                    f"Expected: {cfg_exp.paths.models_dir / cfg_exp.models.chat}\n"
                    "Run: loci models pull"
                )
                console.print(result.context_text)
                conn.close()
                return

            try:
                from loci.models import load_chat
            except ImportError:
                console.print("[yellow]llama-cpp-python not installed — showing raw context.[/yellow]")
                console.print(result.context_text)
                conn.close()
                return

            full_text = ""
            with load_chat(cfg_exp, low_mem=low_mem) as llm:
                messages = build_messages(question, result.context_text, [])
                for token in stream_response(
                    llm, messages,
                    max_tokens=cfg.models.max_tokens,
                    temperature=cfg.models.temperature,
                ):
                    console.print(token, end="", markup=False, highlight=False)
                    full_text += token
            console.print()

            cited = extract_cited_tags(full_text)
            footer = build_sources_footer(cited, result.fact_hits, result.chunk_hits)
            if footer:
                console.print(f"\n[dim]{footer}[/dim]")

            counters["tokens_generated"] = len(full_text.split())
            counters["cited_tags"] = len(cited)
            counters["is_refusal"] = int(is_refusal(full_text))

        finally:
            conn.close()


@app.command("ingest")
def ingest_cmd(
    path: Path = typer.Argument(..., help="File to ingest (.txt or .md)."),
    meta: list[str] = typer.Option(
        [], "--meta", "-m",
        help="Metadata as key=value pairs (repeatable): --meta book=Title --meta author=Name",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    low_mem: bool = typer.Option(False, "--low-mem", help="Use 1B fallback model."),
) -> None:
    """Ingest a document into the knowledge base."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    parsed_meta = _parse_meta(meta)

    from loci.config import expanded
    from loci.store import open_db

    cfg_exp = expanded(cfg)

    with measure("ingest", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
        try:
            conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
        except Exception as exc:
            console.print(f"[red]Cannot open DB:[/red] {exc}")
            raise typer.Exit(1)

        embedder = None
        if not _embedder_model_exists(cfg_exp):
            console.print(
                "[yellow]Embedder model not found — chunk vectors will be skipped.[/yellow]\n"
                f"Expected: {cfg_exp.paths.models_dir / cfg_exp.models.embedder}\n"
                "Run: loci models pull"
            )
        else:
            try:
                from loci.models import load_embedder
                _embedder_ctx = load_embedder(cfg_exp)
                embedder = _embedder_ctx.__enter__()
            except ImportError as exc:
                console.print(f"[yellow]llama-cpp-python not installed — skipping embeddings.[/yellow]")

        try:
            from loci.ingest import ingest_file
            stats = ingest_file(
                path, meta=parsed_meta, cfg=cfg, conn=conn, embedder=embedder
            )
        finally:
            if embedder is not None:
                try:
                    _embedder_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            conn.close()

        counters.update(stats)

    if stats.get("skipped"):
        console.print(f"[yellow]Skipped[/yellow] {path} — already ingested.")
        return

    console.print(
        f"[green]Ingested[/green] {path}\n"
        f"  chunks: {stats['chunks']}  facts: {stats['facts']}"
        f"  new entities: {stats['entities_new']}"
        f"  new aliases: {stats['linked_entities']}"
    )


@app.command("chat")
def chat_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    low_mem: bool = typer.Option(False, "--low-mem", help="Use 1B model with 2048 ctx."),
) -> None:
    """Interactive chat REPL with rolling 3-turn history and grounded answers."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db
    from loci.retrieve import retrieve
    from loci.generate import (
        build_messages, stream_response, extract_cited_tags,
        build_sources_footer, trim_history,
    )

    cfg_exp = expanded(cfg)

    if not _chat_model_exists(cfg_exp):
        console.print(
            f"[red]Chat model not found:[/red] {cfg_exp.paths.models_dir / cfg_exp.models.chat}\n"
            "Run: loci models pull"
        )
        raise typer.Exit(1)

    try:
        from loci.models import load_chat, load_embedder
    except ImportError:
        console.print("[red]llama-cpp-python not installed.[/red]")
        raise typer.Exit(1)

    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    from loci.pack import attach_registered_packs
    p_schemas = attach_registered_packs(conn, cfg_exp.paths.packs_dir)

    try:
        import spacy
        nlp = spacy.load(cfg.ingest.spacy_model, disable=["ner", "senter"])
    except Exception:
        nlp = None

    embedder = None
    _emb_ctx = None
    if _embedder_model_exists(cfg_exp):
        try:
            _emb_ctx = load_embedder(cfg_exp)
            embedder = _emb_ctx.__enter__()
        except Exception:
            pass

    console.print("[bold]Loci chat[/bold] — type [dim]quit[/dim] to exit.")
    if p_schemas:
        console.print(f"[dim]Packs loaded: {len(p_schemas)}[/dim]")

    history: list[dict] = []

    try:
        with load_chat(cfg_exp, low_mem=low_mem) as llm:
            while True:
                try:
                    question = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break

                if question.lower() in ("quit", "exit", "q", ""):
                    break

                with measure("chat_turn", log_dir=cfg_exp.paths.runtime_logs_dir,
                             silent=True) as counters:
                    result = retrieve(question, conn=conn, cfg=cfg,
                                      embedder=embedder, nlp=nlp,
                                      pack_schemas=p_schemas)
                    counters["fact_hits"] = len(result.fact_hits)
                    counters["chunk_hits"] = len(result.chunk_hits)

                    messages = build_messages(question, result.context_text, history)

                    console.print("\n[bold cyan]Loci:[/bold cyan] ", end="")
                    full_text = ""
                    for token in stream_response(
                        llm, messages,
                        max_tokens=cfg.models.max_tokens,
                        temperature=cfg.models.temperature,
                    ):
                        print(token, end="", flush=True)
                        full_text += token
                    print()

                    cited = extract_cited_tags(full_text)
                    footer = build_sources_footer(cited, result.fact_hits, result.chunk_hits)
                    if footer:
                        console.print(f"[dim]{footer}[/dim]")

                    counters["tokens_generated"] = len(full_text.split())
                    counters["cited_tags"] = len(cited)

                # Update rolling history (question only, no injected context)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": full_text})
                history = trim_history(history, max_turns=3)
    finally:
        if _emb_ctx is not None:
            try:
                _emb_ctx.__exit__(None, None, None)
            except Exception:
                pass
        conn.close()


# ---------------------------------------------------------------------------
# loci stats
# ---------------------------------------------------------------------------

@app.command("stats")
def stats_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Show knowledge-base statistics: counts, size, most-connected entities."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db, get_stats

    cfg_exp = expanded(cfg)
    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        stats = get_stats(conn)
    finally:
        conn.close()

    t = Table(title="Knowledge Base", show_header=True, header_style="bold")
    t.add_column("Metric", style="cyan")
    t.add_column("Value")
    for key in ("sources_count", "chunks_count", "entities_count",
                "aliases_count", "facts_count"):
        t.add_row(key.replace("_count", "s"), str(stats[key]))
    t.add_row("db size", f"{stats['db_size_mb']} MB")
    console.print(t)

    if stats["top_entities"]:
        t2 = Table(title="Most-Connected Entities", show_header=True, header_style="bold")
        t2.add_column("Entity", style="cyan")
        t2.add_column("Kind")
        t2.add_column("Facts", justify="right")
        for e in stats["top_entities"]:
            t2.add_row(e["name"], e["kind"], str(e["facts"]))
        console.print(t2)


# ---------------------------------------------------------------------------
# loci maintenance
# ---------------------------------------------------------------------------

@app.command("maintenance")
def maintenance_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run VACUUM and ANALYZE to reclaim space and refresh query planner stats."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db, db_vacuum, db_analyze

    cfg_exp = expanded(cfg)
    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        console.print("Running VACUUM…", end=" ")
        db_vacuum(conn)
        console.print("[green]done[/green]")
        console.print("Running ANALYZE…", end=" ")
        db_analyze(conn)
        console.print("[green]done[/green]")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# loci entities review / merge
# ---------------------------------------------------------------------------

@entities_app.command("review")
def entities_review_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Show pending entity-resolution conflicts for manual review."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db, get_pending_links

    cfg_exp = expanded(cfg)
    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        links = get_pending_links(conn)
    finally:
        conn.close()

    if not links:
        console.print("[green]No pending entity conflicts.[/green]")
        return

    t = Table(title=f"Pending Conflicts ({len(links)})", header_style="bold")
    t.add_column("ID", style="dim")
    t.add_column("Mention", style="cyan")
    t.add_column("Candidates")
    for link in links:
        candidates = ", ".join(
            f"{c['name']} (id={c['id']}, {c['kind']})"
            for c in link["candidates"]
        )
        t.add_row(str(link["id"]), link["mention"], candidates)
    console.print(t)
    console.print(
        "\nResolve with: [bold]loci entities merge <keep_name> <merge_name>[/bold]"
    )


@entities_app.command("merge")
def entities_merge_cmd(
    keep: str = typer.Argument(..., help="Canonical name of the entity to keep."),
    merge: str = typer.Argument(..., help="Canonical name of the entity to merge into keep."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Merge two entities: rewrite all facts/aliases from MERGE onto KEEP."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db, merge_entities

    cfg_exp = expanded(cfg)
    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        keep_row = conn.execute(
            "SELECT id FROM entities WHERE canonical_name=?", [keep]
        ).fetchone()
        merge_row = conn.execute(
            "SELECT id FROM entities WHERE canonical_name=?", [merge]
        ).fetchone()
        if keep_row is None:
            console.print(f"[red]Entity not found:[/red] {keep!r}")
            raise typer.Exit(1)
        if merge_row is None:
            console.print(f"[red]Entity not found:[/red] {merge!r}")
            raise typer.Exit(1)
        n = merge_entities(conn, keep_id=keep_row["id"], merge_id=merge_row["id"])
        console.print(
            f"[green]Merged[/green] {merge!r} → {keep!r}  ({n} facts now on kept entity)"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# loci pack export / add / list / remove
# ---------------------------------------------------------------------------

@pack_app.command("export")
def pack_export_cmd(
    name: str = typer.Option(..., "--name", help="Label for this pack."),
    out: Path = typer.Option(..., "--out", help="Output path (e.g. sherlock.locipack.db)."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Export all knowledge from the main DB into a portable .locipack.db file."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db
    from loci.pack import export_pack

    cfg_exp = expanded(cfg)
    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        with measure("pack_export", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
            counts = export_pack(conn, out.expanduser(), name)
            counters.update(counts)
    finally:
        conn.close()

    console.print(
        f"[green]Exported[/green] → {out}\n"
        f"  sources: {counts.get('sources', 0)}  chunks: {counts.get('chunks', 0)}"
        f"  entities: {counts.get('entities', 0)}  facts: {counts.get('facts', 0)}"
    )


@pack_app.command("add")
def pack_add_cmd(
    pack_path: Path = typer.Argument(..., help="Path to .locipack.db file."),
    name: Optional[str] = typer.Option(None, "--name", help="Registry name (default: filename stem)."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Register a .locipack.db file so queries fan out across it."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.pack import validate_pack, register_pack

    cfg_exp = expanded(cfg)
    resolved = pack_path.expanduser().resolve()
    pack_name = name or resolved.stem

    try:
        validate_pack(resolved)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Invalid pack:[/red] {exc}")
        raise typer.Exit(1)

    try:
        register_pack(cfg_exp.paths.packs_dir, resolved, pack_name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Registered[/green] pack '{pack_name}' → {resolved}")


@pack_app.command("list")
def pack_list_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """List registered knowledge packs."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.pack import load_registry

    cfg_exp = expanded(cfg)
    registry = load_registry(cfg_exp.paths.packs_dir)

    if not registry:
        console.print("[dim]No packs registered.[/dim]")
        return

    t = Table(title="Registered Packs", header_style="bold")
    t.add_column("Name", style="cyan")
    t.add_column("Path")
    t.add_column("Status")
    for entry in registry:
        p = Path(entry["path"])
        status = "[green]ok[/green]" if p.exists() else "[red]missing[/red]"
        t.add_row(entry["name"], entry["path"], status)
    console.print(t)


@pack_app.command("remove")
def pack_remove_cmd(
    name: str = typer.Argument(..., help="Registry name of the pack to remove."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Unregister a knowledge pack (does not delete the file)."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.pack import unregister_pack

    cfg_exp = expanded(cfg)
    try:
        unregister_pack(cfg_exp.paths.packs_dir, name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Removed[/green] pack '{name}' from registry.")


# ---------------------------------------------------------------------------
# loci bench ingest / query / report / compare / qna-skeleton
# ---------------------------------------------------------------------------

@bench_app.command("ingest")
def bench_ingest_cmd(
    path: Path = typer.Argument(..., help="File to benchmark."),
    db: Optional[Path] = typer.Option(None, "--db", help="DB path (default: temp)."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    low_mem: bool = typer.Option(False, "--low-mem"),
) -> None:
    """Benchmark ingestion throughput: chunks/s, facts/s, peak RSS, db growth."""
    import tempfile
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db
    from loci.ingest import ingest_file

    cfg_exp = expanded(cfg)

    cleanup = db is None
    if db is None:
        tmpdir = tempfile.mkdtemp(prefix="loci_bench_")
        db_path = Path(tmpdir) / "bench.db"
    else:
        db_path = db

    initial_size = db_path.stat().st_size if db_path.exists() else 0

    try:
        conn = open_db(db_path, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    embedder = None
    _emb_ctx = None
    if _embedder_model_exists(cfg_exp):
        try:
            from loci.models import load_embedder
            _emb_ctx = load_embedder(cfg_exp)
            embedder = _emb_ctx.__enter__()
        except Exception:
            pass

    try:
        import spacy
        nlp = spacy.load(cfg.ingest.spacy_model, disable=["ner", "senter"])
    except Exception:
        nlp = None

    with measure("bench_ingest", log_dir=cfg_exp.paths.bench_logs_dir) as counters:
        stats = ingest_file(path, cfg=cfg, conn=conn,
                            embedder=embedder, spacy_nlp=nlp)
        counters.update(stats)
        wall = counters.get("_peak_rss_mb", 0)  # placeholder until we read wall time

    if _emb_ctx is not None:
        try:
            _emb_ctx.__exit__(None, None, None)
        except Exception:
            pass
    conn.close()

    final_size = db_path.stat().st_size if db_path.exists() else 0
    db_growth = final_size - initial_size

    if cleanup:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    if stats.get("skipped"):
        console.print(f"[yellow]Skipped[/yellow] {path} — already ingested.")
        return

    t = Table(title=f"Bench Ingest: {path.name}", header_style="bold")
    t.add_column("Metric", style="cyan")
    t.add_column("Value")
    t.add_row("chunks", str(stats["chunks"]))
    t.add_row("facts", str(stats["facts"]))
    t.add_row("sentences_total", str(stats.get("sentences_total", "?")))
    t.add_row("sentences_skipped", str(stats.get("sentences_skipped", "?")))
    t.add_row("entities_new", str(stats["entities_new"]))
    t.add_row("db size growth", f"{db_growth/1024:.1f} KB")
    t.add_row("peak RSS", f"{counters.get('_peak_rss_mb', 0):.0f} MB")
    t.add_row("swap Δ", f"{counters.get('_swap_delta_mb', 0):+.1f} MB")
    console.print(t)


@bench_app.command("query")
def bench_query_cmd(
    qna_path: Path = typer.Option(Path("bench/qna.json"), "--qna", help="QnA JSON file."),
    runs: int = typer.Option(3, "--runs", help="Runs per question."),
    judge: str = typer.Option("claude", "--judge", help="Judge: 'claude' or 'none'."),
    label: str = typer.Option("", "--label", help="Run label for the log file."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    low_mem: bool = typer.Option(False, "--low-mem"),
) -> None:
    """Full eval: retrieve + generate for every QnA item; mechanical + judge scoring."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db
    from loci.retrieve import retrieve
    from loci.generate import (build_messages, stream_response,
                               extract_cited_tags, is_refusal,
                               strip_invalid_citations)
    from loci.bench import (load_qna, score_mechanical, QuestionResult,
                            write_run_log, render_report, run_judging)
    from loci.pack import attach_registered_packs

    cfg_exp = expanded(cfg)

    if not qna_path.exists():
        console.print(f"[red]QnA file not found:[/red] {qna_path}")
        raise typer.Exit(1)

    qna_items = load_qna(qna_path)
    console.print(f"Loaded {len(qna_items)} QnA items from {qna_path}")

    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    p_schemas = attach_registered_packs(conn, cfg_exp.paths.packs_dir)

    try:
        import spacy
        nlp = spacy.load(cfg.ingest.spacy_model, disable=["ner", "senter"])
    except Exception:
        nlp = None

    embedder = None
    _emb_ctx = None
    if _embedder_model_exists(cfg_exp):
        try:
            from loci.models import load_embedder
            _emb_ctx = load_embedder(cfg_exp)
            embedder = _emb_ctx.__enter__()
        except Exception:
            pass

    if not _chat_model_exists(cfg_exp):
        console.print("[red]Chat model not found — run: loci models pull[/red]")
        raise typer.Exit(1)

    try:
        from loci.models import load_chat
    except ImportError:
        console.print("[red]llama-cpp-python not installed.[/red]")
        raise typer.Exit(1)

    all_results: list[QuestionResult] = []

    try:
        t_model_load = time.perf_counter()
        with load_chat(cfg_exp, low_mem=low_mem) as llm:
            model_load_ms = (time.perf_counter() - t_model_load) * 1000

            for item in qna_items:
                console.print(f"  [{item.id}/{item.type}] {item.question[:55]}…")
                for run_idx in range(runs):
                    timings: dict = {
                        "model_load_ms": model_load_ms if run_idx == 0 else 0.0
                    }

                    # RSS sampling
                    rss_samples: list[float] = []
                    swap_before = psutil.swap_memory().used / 1024 / 1024
                    stop_ev = threading.Event()

                    def _sample(ev=stop_ev, samples=rss_samples):
                        while not ev.is_set():
                            try:
                                samples.append(
                                    psutil.Process().memory_info().rss / 1024 / 1024
                                )
                            except Exception:
                                pass
                            ev.wait(0.1)

                    sampler = threading.Thread(target=_sample, daemon=True)
                    sampler.start()

                    # HyDE-lite: generate a hypothetical answer for richer vec query
                    hyde_emb = None
                    if cfg.retrieval.hyde_query and embedder is not None:
                        try:
                            from loci.generate import generate_response
                            from loci.models import embed_batch as _eb
                            _hyde_msgs = [
                                {"role": "system", "content": "Answer the question in one short factual sentence. No hedging."},
                                {"role": "user", "content": item.question},
                            ]
                            _hypo = generate_response(llm, _hyde_msgs, max_tokens=40, temperature=0.0)
                            _h_vecs = _eb(embedder, [_hypo], normalize=True)
                            if _h_vecs:
                                hyde_emb = _h_vecs[0]
                        except Exception:
                            pass

                    result = retrieve(
                        item.question,
                        conn=conn, cfg=cfg,
                        embedder=embedder, nlp=nlp,
                        pack_schemas=p_schemas, timings=timings,
                        hyde_embedding=hyde_emb,
                    )
                    messages = build_messages(item.question, result.context_text, [])

                    full_text = ""
                    first_tok_t: list[float] = []
                    gen_start = time.perf_counter()
                    for tok in stream_response(
                        llm, messages,
                        max_tokens=cfg.models.max_tokens,
                        temperature=cfg.models.temperature,
                    ):
                        if not first_tok_t:
                            first_tok_t.append(time.perf_counter())
                        full_text += tok
                    gen_end = time.perf_counter()

                    stop_ev.set()
                    sampler.join(timeout=0.5)

                    timings["gen_ms"] = (gen_end - gen_start) * 1000
                    timings["ttft_ms"] = (
                        (first_tok_t[0] - gen_start) * 1000 if first_tok_t else 0.0
                    )
                    timings["tokens"] = len(full_text.split())

                    peak_rss = max(rss_samples, default=psutil.Process().memory_info().rss / 1024 / 1024)
                    swap_delta = psutil.swap_memory().used / 1024 / 1024 - swap_before

                    full_text = strip_invalid_citations(
                        full_text, result.fact_hits, result.chunk_hits
                    )
                    citations = extract_cited_tags(full_text)
                    mechanical = score_mechanical(
                        item, full_text, result.fact_hits, result.chunk_hits
                    )
                    all_results.append(QuestionResult(
                        q_id=item.id, q_type=item.type,
                        question=item.question, answerable=item.answerable,
                        answer=full_text, citations=citations,
                        mechanical=mechanical,
                        judge_score=None, judge_reason=None,
                        timings=timings,
                        peak_rss_mb=peak_rss, swap_delta_mb=swap_delta,
                        run_index=run_idx,
                    ))
    finally:
        if _emb_ctx is not None:
            try:
                _emb_ctx.__exit__(None, None, None)
            except Exception:
                pass
        conn.close()

    # Judging
    ea_map = {item.id: item.expected_answer for item in qna_items
              if item.expected_answer is not None}
    judge_scores = run_judging(
        all_results,
        judge=judge if judge != "claude" or cfg.bench.judge != "none" else cfg.bench.judge,
        judge_cmd=cfg.bench.judge_cmd,
        judge_max_chars=cfg.bench.judge_max_chars,
        log_dir=cfg_exp.paths.bench_logs_dir,
        run_label=label or "query",
        expected_answers=ea_map,
    )
    score_map = {s["id"]: s for s in judge_scores}
    for r in all_results:
        if r.q_id in score_map and r.run_index == 0:
            r.judge_score = score_map[r.q_id]["score"]
            r.judge_reason = score_map[r.q_id]["reason"]

    # Write run log
    import dataclasses as _dc
    cfg_snap = {
        s: {k: str(v) for k, v in vars(getattr(cfg, s)).items()}
        for s in ("models", "ingest", "retrieval", "bench")
    }
    run_label = label or f"qna_{len(qna_items)}q"
    log_path = write_run_log(
        all_results, cfg_snap, run_label, cfg_exp.paths.bench_logs_dir
    )
    console.print(f"\n[dim]Log: {log_path}[/dim]\n")
    console.print(render_report({"config": {"label": run_label, "ts": 0},
                                  "results": [r.to_dict() for r in all_results],
                                  "aggregate": {}}))


@bench_app.command("report")
def bench_report_cmd(
    run_path: Optional[Path] = typer.Argument(None, help="Run JSONL path. Default: latest."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Render a benchmark run report."""
    from loci.bench import read_run_log, render_report

    if run_path is None:
        try:
            cfg = cfg_module.load(config_path)
            from loci.config import expanded
            cfg_exp = expanded(cfg)
            logs = sorted(cfg_exp.paths.bench_logs_dir.glob("*.jsonl"))
            if not logs:
                console.print("[yellow]No run logs found.[/yellow]")
                raise typer.Exit(0)
            run_path = logs[-1]
        except Exception as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    try:
        run_data = read_run_log(run_path)
    except Exception as exc:
        console.print(f"[red]Cannot read log:[/red] {exc}")
        raise typer.Exit(1)

    console.print(render_report(run_data))


@bench_app.command("compare")
def bench_compare_cmd(
    run_a: Path = typer.Argument(..., help="First run JSONL."),
    run_b: Path = typer.Argument(..., help="Second run JSONL."),
) -> None:
    """Compare two bench runs metric-by-metric with regression highlighting."""
    from loci.bench import read_run_log, compare_runs

    try:
        da = read_run_log(run_a)
        db_ = read_run_log(run_b)
    except Exception as exc:
        console.print(f"[red]Cannot read logs:[/red] {exc}")
        raise typer.Exit(1)

    console.print(compare_runs(da, db_))


@bench_app.command("qna-skeleton")
def bench_qna_skeleton_cmd(
    source: str = typer.Argument(..., help="Source title or path substring to filter facts."),
    limit: int = typer.Option(20, "--limit", help="Max QnA items to generate."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output JSON path."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Draft candidate QnA items from ingested facts for a source (review before use)."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db

    cfg_exp = expanded(cfg)
    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        rows = conn.execute(
            """
            SELECT f.id, e.canonical_name AS subject, f.predicate,
                   f.object_text, f.qualifiers, f.sentence,
                   s.title, s.path
            FROM facts f
            JOIN entities e ON f.subject_id = e.id
            LEFT JOIN chunks c ON f.chunk_id = c.id
            LEFT JOIN sources s ON c.source_id = s.id
            WHERE (s.title LIKE ? OR s.path LIKE ?)
              AND f.confidence >= 0.8
            ORDER BY RANDOM()
            LIMIT ?
            """,
            [f"%{source}%", f"%{source}%", limit],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        console.print(f"[yellow]No facts found for source matching '{source}'.[/yellow]")
        return

    import json as _json
    items = []
    for i, r in enumerate(rows, 1):
        subj = r["subject"]
        pred = r["predicate"]
        obj = r["object_text"] or ""
        src_title = r["title"] or (Path(r["path"]).name if r["path"] else "")

        question = _question_template(subj, pred, obj)
        keywords = [obj] if obj else []
        items.append({
            "id": f"q{i:03d}",
            "type": "fact",
            "question": question,
            "expected_keywords": keywords,
            "expected_facts": [{"subject": subj.lower(), "predicate": pred}],
            "expected_sources": [src_title] if src_title else [],
            "answerable": True,
        })

    output = _json.dumps(items, indent=2)
    if out:
        out.write_text(output)
        console.print(f"[green]Wrote {len(items)} items to {out}[/green]")
    else:
        console.print(output)


def _question_template(subject: str, predicate: str, obj: str) -> str:
    templates = {
        "take":    f"What did {subject} take?",
        "observe": f"What did {subject} observe?",
        "enter":   f"Where did {subject} enter?",
        "be":      f"What was {subject}?",
        "possess": f"What did {subject} possess?",
        "say":     f"What did {subject} say?",
        "find":    f"What did {subject} find?",
        "give":    f"What did {subject} give?",
        "see":     f"What did {subject} see?",
        "know":    f"What did {subject} know?",
    }
    if predicate in templates:
        return templates[predicate]
    return f"What did {subject} {predicate}?"


# ---------------------------------------------------------------------------
# loci enhance
# ---------------------------------------------------------------------------

@app.command("enhance")
def enhance_cmd(
    limit: Optional[int] = typer.Option(None, "--limit", help="Max chunks to process."),
    all_chunks: bool = typer.Option(False, "--all", help="Re-run on ALL chunks (reset extracted_v first). Needed for the P1 taxonomy re-extraction pass."),
    passes: Optional[str] = typer.Option(None, "--passes", help="Comma-separated P2 passes: entity,implied. Omit for P1 chunk pass."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    low_mem: bool = typer.Option(False, "--low-mem"),
) -> None:
    """LLM-assisted extraction pass: finds passives, copulas, and possessives.

    --passes entity,implied  runs P2 cross-chunk and implication passes instead of the P1 chunk pass.
    """
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db, get_unextracted_chunks
    from loci.enhance import run_enhance, run_entity_pass, run_implied_pass, run_closure_pass, run_prune_pass

    cfg_exp = expanded(cfg)

    requested_passes = {p.strip() for p in passes.split(",")} if passes else set()
    llm_passes = requested_passes - {"closure", "prune"}

    # Prune and closure passes need no LLM — validate model only when LLM passes are requested.
    if not requested_passes or llm_passes:
        if not _chat_model_exists(cfg_exp):
            console.print(
                f"[red]Chat model not found:[/red] {cfg_exp.paths.models_dir / cfg_exp.models.chat}\n"
                "Run: loci models pull"
            )
            raise typer.Exit(1)
        try:
            from loci.models import load_chat
        except ImportError:
            console.print("[red]llama-cpp-python not installed.[/red]")
            raise typer.Exit(1)

    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    embedder = None
    _emb_ctx = None
    if _embedder_model_exists(cfg_exp):
        try:
            from loci.models import load_embedder
            _emb_ctx = load_embedder(cfg_exp)
            embedder = _emb_ctx.__enter__()
        except Exception:
            pass

    combined_stats: dict = {}
    try:
        if not requested_passes or llm_passes:
            # Load LLM for P1 or LLM-based P2 passes
            with load_chat(cfg_exp, low_mem=low_mem) as llm:
                if not requested_passes:
                    # P1 chunk pass (original behaviour)
                    pending = len(get_unextracted_chunks(conn, limit=limit))
                    console.print(f"[bold]Enhancing[/bold] {pending} chunks with LLM extraction…")
                    with measure("enhance", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
                        stats = run_enhance(conn, llm=llm, cfg=cfg, embedder=embedder,
                                            limit=limit, force_all=all_chunks)
                        counters.update(stats)
                    combined_stats = stats
                else:
                    if "entity" in requested_passes:
                        console.print("[bold]P2 entity pass[/bold] — cross-chunk entity-centric extraction…")
                        with measure("enhance_entity", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
                            stats = run_entity_pass(conn, llm=llm, cfg=cfg, embedder=embedder)
                            counters.update(stats)
                        console.print(
                            f"  entities: {stats.get('entities_processed', 0)}"
                            f"  new facts: {stats.get('facts_added', 0)}"
                            + (" [dim](skipped — already done)[/dim]" if stats.get("skipped") else "")
                        )
                        combined_stats.update(stats)
                    if "implied" in requested_passes:
                        console.print("[bold]P2 implied pass[/bold] — implication/archaic-vocab extraction…")
                        with measure("enhance_implied", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
                            stats = run_implied_pass(conn, llm=llm, cfg=cfg, embedder=embedder)
                            counters.update(stats)
                        console.print(
                            f"  chunks: {stats.get('chunks_processed', 0)}"
                            f"  new facts: {stats.get('facts_added', 0)}"
                            + (" [dim](skipped — already done)[/dim]" if stats.get("skipped") else "")
                        )
                        combined_stats.update(stats)

                    # Rebuild full FTS after LLM P2 passes
                    from loci.store import rebuild_fact_fts
                    n_fts = rebuild_fact_fts(conn)
                    conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_v','1')")
                    conn.commit()
                    console.print(f"[green]FTS rebuilt[/green] ({n_fts} facts indexed)")

        # Prune pass — removes misattributed llm/closure facts, resets closure_v
        if "prune" in requested_passes:
            console.print("[bold]Prune pass[/bold] — removing misattributed occupation facts…")
            with measure("enhance_prune", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
                stats = run_prune_pass(conn)
                counters.update(stats)
            console.print(
                f"  removed llm: {stats.get('pruned_llm', 0)}"
                f"  removed closure: {stats.get('pruned_closure', 0)}"
            )
            # Rebuild both FTS tables after pruning
            from loci.store import rebuild_fact_fts, rebuild_fact_fts_llm, _FACT_FTS_LLM_VERSION
            rebuild_fact_fts(conn)
            conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_v','1')")
            rebuild_fact_fts_llm(conn)
            conn.execute(
                "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_llm_v',?)",
                [_FACT_FTS_LLM_VERSION],
            )
            conn.commit()
            combined_stats.update(stats)

        # Closure pass — no LLM required; runs after any LLM passes or standalone
        if "closure" in requested_passes:
            console.print("[bold]Closure pass[/bold] — materialising vocabulary synonym chains…")
            with measure("enhance_closure", log_dir=cfg_exp.paths.runtime_logs_dir) as counters:
                stats = run_closure_pass(conn, cfg=cfg, embedder=embedder)
                counters.update(stats)
            console.print(
                f"  chains found: {stats.get('chains_found', 0)}"
                f"  new facts: {stats.get('facts_added', 0)}"
                + (" [dim](skipped — already done)[/dim]" if stats.get("skipped") else "")
            )
            combined_stats.update(stats)
    finally:
        if _emb_ctx is not None:
            try:
                _emb_ctx.__exit__(None, None, None)
            except Exception:
                pass
        conn.close()

    if not requested_passes:
        stats = combined_stats
        console.print(
            f"[green]Done.[/green]  chunks processed: {stats['chunks_processed']}"
            f"  new facts: {stats['facts_added']}"
        )
    else:
        console.print(f"[green]Passes complete.[/green] Total new facts: {combined_stats.get('facts_added', 0)}")


# ---------------------------------------------------------------------------
# loci synonyms learn
# ---------------------------------------------------------------------------

@synonyms_app.command("learn")
def synonyms_learn_cmd(
    auto: bool = typer.Option(False, "--auto", help="Auto-write pairs to DB."),
    threshold: float = typer.Option(0.85, "--threshold", help="Min cosine similarity."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Cluster predicate embeddings and suggest (or write) synonym pairs."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db

    cfg_exp = expanded(cfg)

    if not _embedder_model_exists(cfg_exp):
        console.print("[red]Embedder model not found — run: loci models pull[/red]")
        raise typer.Exit(1)

    try:
        from loci.models import load_embedder, embed_batch, cosine_dist_threshold
    except ImportError:
        console.print("[red]llama-cpp-python not installed.[/red]")
        raise typer.Exit(1)

    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        predicates = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT predicate FROM facts ORDER BY predicate"
            ).fetchall()
        ]
        if len(predicates) < 2:
            console.print("[yellow]Need at least 2 distinct predicates to find synonyms.[/yellow]")
            return

        console.print(f"Embedding {len(predicates)} predicates…")
        with load_embedder(cfg_exp) as embedder:
            embeddings = embed_batch(embedder, predicates, normalize=True)

        dist_threshold = cosine_dist_threshold(threshold)
        import math
        suggestions: list[tuple[str, str, float]] = []
        for i, (p1, e1) in enumerate(zip(predicates, embeddings)):
            for p2, e2 in zip(predicates[i + 1:], embeddings[i + 1:]):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(e1, e2)))
                if dist < dist_threshold:
                    sim = 1.0 - dist ** 2 / 2.0
                    suggestions.append((p1, p2, sim))

        suggestions.sort(key=lambda x: -x[2])

        if not suggestions:
            console.print(f"[yellow]No synonym pairs found at threshold {threshold}.[/yellow]")
            return

        t = Table(title=f"Synonym Pairs (sim ≥ {threshold})", header_style="bold")
        t.add_column("Predicate A", style="cyan")
        t.add_column("Predicate B", style="cyan")
        t.add_column("Similarity", justify="right")
        for p1, p2, sim in suggestions:
            t.add_row(p1, p2, f"{sim:.3f}")
        console.print(t)

        if auto:
            for p1, p2, _ in suggestions:
                conn.execute(
                    "INSERT OR IGNORE INTO predicate_synonyms (predicate, synonym) VALUES (?,?)",
                    [p1, p2],
                )
            conn.commit()
            console.print(f"[green]Wrote {len(suggestions)} synonym pairs to DB.[/green]")
        else:
            console.print(
                f"\nRun with [bold]--auto[/bold] to write these {len(suggestions)} pairs."
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# loci facts reindex-vec / reindex-fts
# ---------------------------------------------------------------------------

@facts_app.command("reindex-vec")
def facts_reindex_vec_cmd(
    also_fts: bool = typer.Option(False, "--fts", help="Also rebuild fts_facts."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Embed all facts into vec_facts (and optionally rebuild fts_facts)."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db, rebuild_fact_vec, rebuild_fact_fts, _FACT_VEC_VERSION

    cfg_exp = expanded(cfg)

    if not _embedder_model_exists(cfg_exp):
        console.print(
            f"[red]Embedder model not found:[/red] {cfg_exp.paths.models_dir / cfg_exp.models.embedder}\n"
            "Run: loci models pull"
        )
        raise typer.Exit(1)

    try:
        from loci.models import load_embedder
    except ImportError:
        console.print("[red]llama-cpp-python not installed.[/red]")
        raise typer.Exit(1)

    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        if also_fts:
            console.print("Rebuilding fts_facts…", end=" ")
            n_fts = rebuild_fact_fts(conn)
            conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_v','1')")
            conn.commit()
            console.print(f"[green]{n_fts} facts indexed[/green]")

        console.print("Embedding facts into vec_facts…")
        with load_embedder(cfg_exp) as embedder:
            n_vec = rebuild_fact_vec(conn, embedder)
        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_vec_v',?)",
            [_FACT_VEC_VERSION],
        )
        conn.commit()
        console.print(f"[green]Done.[/green] {n_vec} facts embedded.")
    finally:
        conn.close()


@facts_app.command("reindex-fts-llm")
def facts_reindex_fts_llm_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Rebuild the llm-only fact FTS index (fts_facts_llm)."""
    try:
        cfg = cfg_module.load(config_path)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    from loci.config import expanded
    from loci.store import open_db, rebuild_fact_fts_llm, _FACT_FTS_LLM_VERSION

    cfg_exp = expanded(cfg)

    try:
        conn = open_db(cfg_exp.paths.knowledge_db, vec_dim=cfg_exp.models.vec_dim)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

    try:
        n = rebuild_fact_fts_llm(conn)
        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_llm_v',?)",
            [_FACT_FTS_LLM_VERSION],
        )
        conn.commit()
        console.print(
            f"[green]fts_facts_llm rebuilt[/green] ({n} answer-shaped facts indexed)"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_meta(meta_list: list[str]) -> dict:
    result: dict[str, str] = {}
    for item in meta_list:
        if "=" in item:
            key, val = item.split("=", 1)
            result[key.strip()] = val.strip().strip("\"'")
    return result


def _embedder_model_exists(cfg) -> bool:
    return (cfg.paths.models_dir / cfg.models.embedder).exists()


def _chat_model_exists(cfg) -> bool:
    return (cfg.paths.models_dir / cfg.models.chat).exists()


if __name__ == "__main__":
    app()
