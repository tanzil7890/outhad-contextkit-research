"""Configuration models for chunking system."""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChunkingConfig(BaseModel):
    """Configuration for document chunking.

    This configuration controls how long documents are split into
    smaller chunks for embedding and retrieval.

    Example:
        >>> config = ChunkingConfig(
        ...     enabled=True,
        ...     strategy="token",
        ...     chunk_size=512,
        ...     chunk_overlap=50
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable automatic document chunking"
    )

    strategy: Literal["character", "token", "semantic", "hierarchical"] = Field(
        default="token",
        description=(
            "Chunking strategy:\n"
            "- 'character': Split by character count\n"
            "- 'token': Split by token count (recommended)\n"
            "- 'semantic': Split by semantic similarity\n"
            "- 'hierarchical': Split preserving document structure"
        )
    )

    chunk_size: int = Field(
        default=512,
        ge=50,
        le=8000,
        description="Maximum chunk size (tokens or characters depending on strategy)"
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between consecutive chunks for context preservation"
    )

    separators: List[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="List of separators for recursive splitting (character strategy)"
    )

    min_chunk_size: int = Field(
        default=50,
        ge=1,
        description="Minimum chunk size (chunks smaller than this are merged)"
    )

    max_chunks_per_document: Optional[int] = Field(
        default=None,
        description="Maximum number of chunks per document (None = unlimited)"
    )

    # Token-specific config
    tokenizer_model: str = Field(
        default="gpt-4",
        description="Model name for token counting (tiktoken)"
    )

    # Semantic-specific config
    semantic_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic chunking"
    )

    # Hierarchical-specific config
    respect_markdown_headers: bool = Field(
        default=True,
        description="Preserve markdown header boundaries in hierarchical chunking"
    )

    # Retrieval config
    merge_chunks_on_retrieval: bool = Field(
        default=True,
        description="Automatically merge chunks from same document during retrieval"
    )

    chunk_context_window: int = Field(
        default=1,
        ge=0,
        description="Number of neighboring chunks to include for context (±N)"
    )

    min_document_size: int = Field(
        default=0,
        ge=0,
        description=(
            "Minimum document size (in characters) before chunking is applied. "
            "Documents smaller than this will be stored as a single memory. "
            "Set to 0 to always chunk. Recommended: 1500-2000 for narrative texts."
        )
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        """Ensure overlap is less than chunk_size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "strategy": "token",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "tokenizer_model": "gpt-4",
                "merge_chunks_on_retrieval": True,
                "chunk_context_window": 1,
                "min_document_size": 1500
            }
        }
    )
