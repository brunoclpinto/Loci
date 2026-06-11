"""Loci CLI — main typer application."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from loci import config as cfg_module
from loci.bench import measure

app = typer.Typer(name="loci", help="Local knowledge assistant.", no_args_is_help=True)
config_app = typer.Typer(help="Manage configuration.")
app.add_typer(config_app, name="config")

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
            conn = open_db(cfg_exp.paths.knowledge_db)
        except Exception as exc:
            console.print(f"[red]Cannot open DB:[/red] {exc}")
            raise typer.Exit(1)

        try:
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
                                  nlp=nlp, explain=explain)
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
            conn = open_db(cfg_exp.paths.knowledge_db)
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
        conn = open_db(cfg_exp.paths.knowledge_db)
    except Exception as exc:
        console.print(f"[red]Cannot open DB:[/red] {exc}")
        raise typer.Exit(1)

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
                                      embedder=embedder, nlp=nlp)
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
