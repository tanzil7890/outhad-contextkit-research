"""Base classes for document chunking."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class Chunk:
    """Represents a single chunk of text with metadata.

    Attributes:
        content: The chunk text content
        chunk_id: Unique identifier for this chunk
        document_id: Parent document identifier
        chunk_index: Position in the document (0-based)
        total_chunks: Total number of chunks in document
        metadata: Additional metadata
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        token_count: Number of tokens (if available)
    """
    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    token_count: Optional[int] = None

    def __post_init__(self):
        """Validate chunk data."""
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")
        if self.total_chunks <= 0:
            raise ValueError("Total chunks must be positive")
        if self.chunk_index >= self.total_chunks:
            raise ValueError("Chunk index exceeds total chunks")


class ChunkerBase(ABC):
    """Abstract base class for all text chunkers.

    All chunker implementations must inherit from this class and
    implement the split_text method.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs):
        """Initialize chunker.

        Args:
            chunk_size: Maximum size of each chunk (tokens or characters)
            chunk_overlap: Number of overlapping tokens/characters between chunks
            **kwargs: Additional chunker-specific configuration
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._validate_config()

    def _validate_config(self):
        """Validate chunker configuration."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

    @abstractmethod
    def split_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: The text to split
            document_id: Optional document identifier (auto-generated if None)
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of Chunk objects
        """
        pass

    def create_chunk(
        self,
        content: str,
        document_id: str,
        chunk_index: int,
        total_chunks: int,
        metadata: Dict[str, Any],
        start_char: Optional[int] = None,
        end_char: Optional[int] = None
    ) -> Chunk:
        """Helper to create a Chunk object with consistent formatting.

        Args:
            content: Chunk text content
            document_id: Parent document ID
            chunk_index: Position in document sequence
            total_chunks: Total chunks in document
            metadata: Chunk metadata
            start_char: Starting character position
            end_char: Ending character position

        Returns:
            Chunk object
        """
        chunk_id = str(uuid.uuid4())

        # Merge metadata
        full_metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "chunker_type": self.__class__.__name__,
            "chunk_size_config": self.chunk_size,
            "chunk_overlap_config": self.chunk_overlap,
            **(metadata or {})
        }

        return Chunk(
            content=content.strip(),
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            metadata=full_metadata,
            start_char=start_char,
            end_char=end_char
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )


class ChunkMerger:
    """Utility for merging chunks during retrieval.

    Reconstructs parent documents or adds context windows
    around retrieved chunks.
    """

    @staticmethod
    def merge_by_document(
        chunks: List[Dict[str, Any]],
        include_neighbors: bool = True,
        neighbor_window: int = 1
    ) -> List[Dict[str, Any]]:
        """Merge chunks belonging to the same document.

        Args:
            chunks: List of chunk results with metadata
            include_neighbors: Whether to fetch neighboring chunks
            neighbor_window: Number of neighboring chunks to include (±N)

        Returns:
            List of merged results, one per document
        """
        from collections import defaultdict

        # Group by document_id
        doc_groups = defaultdict(list)
        standalone_results = []

        for chunk in chunks:
            metadata = chunk.get("metadata")
            # Handle None metadata
            if metadata is None:
                metadata = {}

            doc_id = metadata.get("document_id")

            if doc_id:
                doc_groups[doc_id].append(chunk)
            else:
                # Non-chunked result
                standalone_results.append(chunk)

        merged_results = []

        # Merge each document's chunks
        for doc_id, doc_chunks in doc_groups.items():
            # Sort by chunk_index
            doc_chunks.sort(key=lambda x: (x.get("metadata") or {}).get("chunk_index", 0))

            # Merge content
            merged_content = "\n\n".join([c.get("memory", "") for c in doc_chunks])

            # Take max score
            merged_score = max([c.get("score", 0.0) for c in doc_chunks])

            # Combine metadata
            first_chunk_metadata = doc_chunks[0].get("metadata") or {}

            # Collect chunk indices safely
            chunk_indices = []
            for c in doc_chunks:
                c_meta = c.get("metadata")
                if c_meta and "chunk_index" in c_meta:
                    chunk_indices.append(c_meta["chunk_index"])

            merged_results.append({
                "id": doc_id,
                "memory": merged_content,
                "score": merged_score,
                "metadata": {
                    **first_chunk_metadata,
                    "chunk_count": len(doc_chunks),
                    "is_merged": True,
                    "chunk_indices": chunk_indices
                }
            })

        # Add standalone results
        merged_results.extend(standalone_results)

        # Sort by score
        return sorted(merged_results, key=lambda x: x.get("score", 0.0), reverse=True)
