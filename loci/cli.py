from pathlib import Path

import typer
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from loci.config import load_settings
from loci.db.engine import session_scope
from loci.db.models import Context, ContextRelationship, EntityType, PredicateVocabulary
from loci.ingest.adapters.mapping.spec import MappingSpec
from loci.ingest.adapters.structured_file import StructuredFileAdapter
from loci.ingest.adapters.unstructured_text import UnstructuredTextAdapter
from loci.ingest.base import SourceDescriptor
from loci.ingest.chunk_embedding import QdrantChunkEmbedder
from loci.ingest.pipeline import IngestionPipeline
from loci.normalize.resolver import EntityResolver

app = typer.Typer(help="Loci — a strictly-organized, mixed-source knowledge base for LLMs.")
registry_app = typer.Typer(help="Inspect the entity-type / predicate schema registry.")
app.add_typer(registry_app, name="registry")
bench_app = typer.Typer(help="Multi-model bench harness (see benchWork/ for corpus/QnA data, gitignored).")
app.add_typer(bench_app, name="bench")

console = Console()

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config() -> AlembicConfig:
    cfg = AlembicConfig(str(_REPO_ROOT / "db" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "db" / "migrations"))
    return cfg


@app.command()
def migrate(revision: str = typer.Argument("head", help="Target revision (default: head).")) -> None:
    """Run Alembic migrations up to REVISION."""
    alembic_command.upgrade(_alembic_config(), revision)
    console.print(f"[green]Migrated to {revision}[/green]")


@registry_app.command("list-types")
def registry_list_types() -> None:
    """List registered entity types and their active schema version."""
    settings = load_settings()
    table = Table(title="Entity Types")
    table.add_column("Name")
    table.add_column("Current schema version")
    with session_scope(settings) as session:
        for et in session.scalars(select(EntityType).order_by(EntityType.name)):
            table.add_row(et.name, str(et.current_schema_version))
    console.print(table)


@registry_app.command("list-predicates")
def registry_list_predicates() -> None:
    """List registered predicates and their domain/range entity types."""
    settings = load_settings()
    table = Table(title="Predicate Vocabulary")
    table.add_column("Name")
    table.add_column("Domain types")
    table.add_column("Range types")
    with session_scope(settings) as session:
        for p in session.scalars(select(PredicateVocabulary).order_by(PredicateVocabulary.name)):
            table.add_row(p.name, ", ".join(p.domain_entity_types) or "*", ", ".join(p.range_entity_types) or "*")
    console.print(table)


@registry_app.command("list-contexts")
def registry_list_contexts() -> None:
    """List contexts and their relationships to other contexts."""
    settings = load_settings()
    table = Table(title="Contexts")
    table.add_column("Name")
    table.add_column("Related to")
    with session_scope(settings) as session:
        contexts = {c.id: c for c in session.scalars(select(Context))}
        rels_by_parent: dict = {}
        for cr in session.scalars(select(ContextRelationship)):
            rels_by_parent.setdefault(cr.parent_context_id, []).append(cr)
        for ctx in sorted(contexts.values(), key=lambda c: c.name):
            related = ", ".join(
                f"{cr.relation_type}:{contexts[cr.child_context_id].name}" for cr in rels_by_parent.get(ctx.id, [])
            )
            table.add_row(ctx.name, related or "-")
    console.print(table)


@registry_app.command("link-contexts")
def registry_link_contexts(
    parent: str = typer.Argument(..., help="Parent context name."),
    child: str = typer.Argument(..., help="Child context name."),
    relation_type: str = typer.Option("contains", "--type", help="e.g. contains, cites, derived_from."),
) -> None:
    """Create a relationship between two contexts (creating them if needed)."""
    settings = load_settings()
    with session_scope(settings) as session:
        parent_ctx = session.scalars(select(Context).where(Context.name == parent)).first()
        if parent_ctx is None:
            parent_ctx = Context(name=parent)
            session.add(parent_ctx)
            session.flush()
        child_ctx = session.scalars(select(Context).where(Context.name == child)).first()
        if child_ctx is None:
            child_ctx = Context(name=child)
            session.add(child_ctx)
            session.flush()
        session.add(
            ContextRelationship(parent_context_id=parent_ctx.id, child_context_id=child_ctx.id, relation_type=relation_type)
        )
    console.print(f"[green]Linked[/green]: {parent} --{relation_type}--> {child}")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Path to the file to ingest."),
    context: str = typer.Option("default", "--context", help="Context name (used if the mapping spec doesn't declare one)."),
    mapping: str = typer.Option(None, "--mapping", help="Path to a structured-file mapping spec (YAML). Required for CSV/JSON sources."),
    adapter: str = typer.Option(None, "--adapter", help="Force a specific adapter (csv|json|text|pdf|md)."),
) -> None:
    """Ingest a source file into the knowledge base."""
    settings = load_settings()
    path = Path(source)
    descriptor = SourceDescriptor(path=path, context_name=context, forced_adapter=adapter)

    chunk_embedder = None
    if path.suffix.lower() in {".csv", ".json"}:
        if not mapping:
            console.print("[red]--mapping is required for CSV/JSON sources.[/red]")
            raise typer.Exit(code=1)
        spec = MappingSpec.from_yaml(Path(mapping))
        adapter_impl = StructuredFileAdapter(spec)
    elif path.suffix.lower() in {".txt", ".md", ".pdf"}:
        adapter_impl = UnstructuredTextAdapter(settings)
        chunk_embedder = QdrantChunkEmbedder(settings)
    else:
        console.print(f"[red]No adapter available for {path.suffix!r} sources.[/red]")
        raise typer.Exit(code=1)

    resolver = EntityResolver(settings)
    pipeline = IngestionPipeline(settings, resolver, chunk_embedder=chunk_embedder)
    stats = pipeline.ingest(adapter_impl, descriptor)
    console.print(f"[green]Ingested[/green]: {stats}")


@bench_app.command("ingest")
def bench_ingest(
    extraction_model: str = typer.Option(None, "--extraction-model", help="Ollama tag; defaults to [bench].default_extraction_model."),
    qna: str = typer.Option(None, "--qna", help="QnA filename under [bench].qna_dir; defaults to [bench].qna_file."),
) -> None:
    """Ingest the book(s) referenced by a QnA set using the given extraction model, into a context namespaced by (book, model)."""
    from loci.bench.ids import ingest_run_id
    from loci.bench.ingest_phase import run_ingest_phase
    from loci.bench.logs import run_dir, write_json

    settings = load_settings()
    extraction_model = extraction_model or settings.bench.default_extraction_model
    qna = qna or settings.bench.qna_file

    result = run_ingest_phase(settings, extraction_model, qna)

    rid = ingest_run_id(qna, extraction_model)
    log_dir = run_dir(Path(settings.bench.log_dir), rid)
    write_json(log_dir / "ingest.json", result)
    console.print(f"[green]Ingest phase complete[/green] ({rid}): {result['duration_s']:.1f}s, books={result['books']}")


@bench_app.command("qa")
def bench_qa(
    extraction_model: str = typer.Option(None, "--extraction-model", help="Which ingested context to retrieve from."),
    answer_model: str = typer.Option(None, "--answer-model", help="Ollama tag; defaults to [bench].default_answer_model."),
    qna: str = typer.Option(None, "--qna", help="QnA filename under [bench].qna_dir."),
) -> None:
    """Answer every question in a QnA set, retrieving from the context(s) already ingested for the given extraction model."""
    from loci.bench.ids import run_id as compute_run_id
    from loci.bench.logs import run_dir, write_jsonl
    from loci.bench.qa_phase import run_qa_phase

    settings = load_settings()
    extraction_model = extraction_model or settings.bench.default_extraction_model
    answer_model = answer_model or settings.bench.default_answer_model
    qna = qna or settings.bench.qna_file

    result = run_qa_phase(settings, extraction_model, answer_model, qna)

    rid = compute_run_id(qna, extraction_model, answer_model)
    log_dir = run_dir(Path(settings.bench.log_dir), rid)
    write_jsonl(log_dir / "retrieval_generation.jsonl", result["rows"])
    console.print(f"[green]QA phase complete[/green] ({rid}): {len(result['rows'])} questions in {result['duration_s']:.1f}s")


@bench_app.command("grade")
def bench_grade(
    extraction_model: str = typer.Option(None, "--extraction-model"),
    answer_model: str = typer.Option(None, "--answer-model"),
    qna: str = typer.Option(None, "--qna"),
) -> None:
    """Grade a completed QA phase's answers against ground truth via a single Claude Code CLI call (0-100 per question)."""
    from loci.bench.grading import grade_qa_rows
    from loci.bench.ids import run_id as compute_run_id
    from loci.bench.logs import read_jsonl, run_dir, write_json

    settings = load_settings()
    extraction_model = extraction_model or settings.bench.default_extraction_model
    answer_model = answer_model or settings.bench.default_answer_model
    qna = qna or settings.bench.qna_file

    rid = compute_run_id(qna, extraction_model, answer_model)
    log_dir = run_dir(Path(settings.bench.log_dir), rid)
    qa_rows = read_jsonl(log_dir / "retrieval_generation.jsonl")
    if not qa_rows:
        console.print(f"[red]No QA rows found for run {rid} — run `loci bench qa` first.[/red]")
        raise typer.Exit(code=1)

    result = grade_qa_rows(qa_rows)
    write_json(log_dir / "grading.json", result)
    console.print(f"[green]Grading complete[/green] ({rid}): mean score {result['mean_score']:.1f}")


@bench_app.command("report")
def bench_report(
    extraction_model: str = typer.Option(None, "--extraction-model"),
    answer_model: str = typer.Option(None, "--answer-model"),
    qna: str = typer.Option(None, "--qna"),
) -> None:
    """Build this run's summary.json from its ingest/qa/grading logs, then print the leaderboard across all runs."""
    from loci.bench.ids import ingest_run_id
    from loci.bench.ids import run_id as compute_run_id
    from loci.bench.logs import read_json, read_jsonl, run_dir, write_json
    from loci.bench.report import build_summary, load_leaderboard, render_table

    settings = load_settings()
    extraction_model = extraction_model or settings.bench.default_extraction_model
    answer_model = answer_model or settings.bench.default_answer_model
    qna = qna or settings.bench.qna_file

    rid = compute_run_id(qna, extraction_model, answer_model)
    log_dir = run_dir(Path(settings.bench.log_dir), rid)
    ingest_dir = run_dir(Path(settings.bench.log_dir), ingest_run_id(qna, extraction_model))

    ingest_result = read_json(ingest_dir / "ingest.json")
    qa_rows = read_jsonl(log_dir / "retrieval_generation.jsonl")
    grading_result = read_json(log_dir / "grading.json")

    summary = build_summary(
        settings, rid, extraction_model, answer_model, qna, ingest_result, {"rows": qa_rows}, grading_result
    )
    write_json(log_dir / "summary.json", summary)

    leaderboard = load_leaderboard(settings)
    console.print(render_table(leaderboard))


@app.command()
def serve() -> None:
    """Run the MCP server (Streamable HTTP)."""
    from loci.mcp_server.server import run as run_mcp_server

    settings = load_settings()
    console.print(f"[green]Starting Loci MCP server[/green] on {settings.mcp.host}:{settings.mcp.port} ({settings.mcp.transport})")
    run_mcp_server(settings)


if __name__ == "__main__":
    app()
