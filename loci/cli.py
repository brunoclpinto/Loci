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


@app.command()
def serve() -> None:
    """Run the MCP server (Streamable HTTP)."""
    from loci.mcp_server.server import run as run_mcp_server

    settings = load_settings()
    console.print(f"[green]Starting Loci MCP server[/green] on {settings.mcp.host}:{settings.mcp.port} ({settings.mcp.transport})")
    run_mcp_server(settings)


if __name__ == "__main__":
    app()
