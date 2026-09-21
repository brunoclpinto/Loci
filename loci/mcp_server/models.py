from pydantic import BaseModel


class Citation(BaseModel):
    source_adapter: str | None
    source_ref: str | None
    context_name: str


class EntitySummary(BaseModel):
    id: str
    type: str
    canonical_name: str
    aliases: list[str]
    context_name: str
    scope: str
    attributes: dict
    citation: Citation


class RelationshipSummary(BaseModel):
    id: str
    subject_id: str
    subject_name: str
    predicate: str
    object_id: str | None
    object_name: str | None
    object_literal: dict | None
    context_name: str
    citation: Citation


class ChunkHit(BaseModel):
    text: str
    score: float
    context_name: str
    citation: Citation


class SearchResult(BaseModel):
    chunk_hits: list[ChunkHit]
    entity_hits: list[EntitySummary]


class ContextRelation(BaseModel):
    relation_type: str
    context_name: str


class ContextSummary(BaseModel):
    id: str
    name: str
    description: str | None
    parents: list[ContextRelation]
    children: list[ContextRelation]
    entity_count: int
    relationship_count: int


class EntityTypeSchemaOut(BaseModel):
    name: str
    version: int
    json_schema: dict


class ContextEntityView(BaseModel):
    context_name: str
    entity: EntitySummary | None
    relationships: list[RelationshipSummary]
