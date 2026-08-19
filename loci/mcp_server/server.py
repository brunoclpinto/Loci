from mcp.server.mcpserver import MCPServer

from loci.config import LociSettings, load_settings
from loci.mcp_server.tools.contexts import make_compare_contexts, make_get_context, make_list_contexts
from loci.mcp_server.tools.entities import make_get_entity, make_resolve_entity
from loci.mcp_server.tools.registry import make_get_entity_type_schema, make_list_entity_types, make_list_predicates
from loci.mcp_server.tools.relationships import make_get_relationships
from loci.mcp_server.tools.search_knowledge import make_search_knowledge


def build_server(settings: LociSettings) -> MCPServer:
    mcp = MCPServer(
        name="loci",
        title="Loci Knowledge Base",
        instructions=(
            "Loci is a strictly-organized, mixed-source knowledge base. Query it for "
            "structured facts (entities/relationships) and semantic passages, each carrying "
            "a citation back to its source and context. Prefer search_knowledge for open "
            "questions; use get_entity / get_relationships once you have an entity id. "
            "Facts are scoped to a context (e.g. a specific book vs. real_world) — a fact "
            "is only asserted true within the context it's returned from; use "
            "compare_contexts to see how the same entity differs across contexts."
        ),
    )

    for factory in (
        make_search_knowledge,
        make_get_entity,
        make_resolve_entity,
        make_get_relationships,
        make_list_contexts,
        make_get_context,
        make_compare_contexts,
        make_list_entity_types,
        make_get_entity_type_schema,
        make_list_predicates,
    ):
        mcp.tool()(factory(settings))

    return mcp


def run(settings: LociSettings | None = None) -> None:
    settings = settings or load_settings()
    server = build_server(settings)
    server.run(transport=settings.mcp.transport, host=settings.mcp.host, port=settings.mcp.port)
