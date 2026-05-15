"""Text splitter implementations for document chunking.

This module provides production-ready text splitters:
- RecursiveCharacterTextSplitter: Character-based chunking with recursive separators
- SentenceTextSplitter: Sentence boundary detection with size constraints
- TokenTextSplitter: Token-based chunking using tiktoken (production recommended)
"""

import re
from typing import Any, Dict, List, Optional
import uuid

from .base import Chunk, ChunkerBase


class RecursiveCharacterTextSplitter(ChunkerBase):
    """Split text recursively using a hierarchy of separators.

    This splitter attempts to keep semantically related text together by
    trying separators in order of priority (paragraphs, then sentences, etc.).

    Args:
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Number of overlapping characters between chunks
        separators: List of separators in priority order
        keep_separator: Whether to keep the separator in the chunk

    Example:
        >>> splitter = RecursiveCharacterTextSplitter(
        ...     chunk_size=1000,
        ...     chunk_overlap=100,
        ...     separators=["\\n\\n", "\\n", ". ", " ", ""]
        ... )
        >>> chunks = splitter.split_text("Your long document...")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.keep_separator = keep_separator

    def split_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into chunks using recursive separator strategy.

        Args:
            text: Text to split
            document_id: Optional document identifier
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Get text splits
        splits = self._split_text_recursive(text, self.separators)

        # Merge splits into chunks with overlap
        chunks = self._merge_splits(splits, document_id, metadata)

        return chunks

    def _split_text_recursive(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """Recursively split text using separator hierarchy.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text splits
        """
        final_splits = []

        # Get the appropriate separator
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by the separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Process each split
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_splits.extend(good_splits)
                    good_splits = []

                if not new_separators:
                    # No more separators, split by character
                    final_splits.append(split)
                else:
                    # Recursively split with next separator
                    final_splits.extend(
                        self._split_text_recursive(split, new_separators)
                    )

        if good_splits:
            final_splits.extend(good_splits)

        # Re-add separators if needed
        if self.keep_separator and separator:
            final_splits = [
                split + separator if i < len(final_splits) - 1 else split
                for i, split in enumerate(final_splits)
            ]

        return [s for s in final_splits if s]

    def _merge_splits(
        self,
        splits: List[str],
        document_id: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Merge splits into chunks with overlap.

        Args:
            splits: List of text splits
            document_id: Document identifier
            metadata: Metadata for chunks

        Returns:
            List of Chunk objects
        """
        # First pass: create chunk data without Chunk objects
        chunk_data = []
        current_chunk = []
        current_length = 0
        char_position = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length > self.chunk_size and current_chunk:
                # Store chunk data
                chunk_text = "".join(current_chunk)
                chunk_data.append({
                    "content": chunk_text,
                    "start_char": char_position - current_length,
                    "end_char": char_position
                })

                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text, split]
                    current_length = len(overlap_text) + split_length
                else:
                    current_chunk = [split]
                    current_length = split_length
            else:
                current_chunk.append(split)
                current_length += split_length

            char_position += split_length

        # Add final chunk data
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunk_data.append({
                "content": chunk_text,
                "start_char": char_position - current_length,
                "end_char": char_position
            })

        # Second pass: create Chunk objects with correct total_chunks
        total_chunks = len(chunk_data)
        chunks = []
        for idx, data in enumerate(chunk_data):
            chunk = Chunk(
                content=data["content"],
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                metadata=metadata.copy(),
                start_char=data["start_char"],
                end_char=data["end_char"],
                token_count=None
            )
            chunks.append(chunk)

        return chunks


class SentenceTextSplitter(ChunkerBase):
    """Split text at sentence boundaries while respecting size constraints.

    Uses regex patterns to detect sentence boundaries and groups sentences
    into chunks that don't exceed the maximum size.

    Args:
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Number of overlapping characters between chunks
        sentence_pattern: Regex pattern for sentence boundaries

    Example:
        >>> splitter = SentenceTextSplitter(chunk_size=1000, chunk_overlap=50)
        >>> chunks = splitter.split_text("First sentence. Second sentence.")
    """

    # Regex pattern for sentence boundaries
    DEFAULT_SENTENCE_PATTERN = r'(?<=[.!?])\s+'

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        sentence_pattern: Optional[str] = None,
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.sentence_pattern = sentence_pattern or self.DEFAULT_SENTENCE_PATTERN
        self._sentence_regex = re.compile(self.sentence_pattern)

    def split_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into chunks at sentence boundaries.

        Args:
            text: Text to split
            document_id: Optional document identifier
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Split into sentences
        sentences = self._sentence_regex.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # First pass: create chunk data without Chunk objects
        chunk_data = []
        current_chunk = []
        current_length = 0
        char_position = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Store chunk data
                chunk_text = " ".join(current_chunk)
                chunk_data.append({
                    "content": chunk_text,
                    "start_char": char_position - current_length,
                    "end_char": char_position
                })

                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Find sentences that fit in overlap window
                    overlap_sentences = []
                    overlap_length = 0
                    for sent in reversed(current_chunk):
                        if overlap_length + len(sent) <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_length += len(sent)
                        else:
                            break

                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

            char_position += sentence_length

        # Add final chunk data
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_data.append({
                "content": chunk_text,
                "start_char": char_position - current_length,
                "end_char": char_position
            })

        # Second pass: create Chunk objects with correct total_chunks
        total_chunks = len(chunk_data)
        chunks = []
        for idx, data in enumerate(chunk_data):
            chunk = Chunk(
                content=data["content"],
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                metadata=metadata.copy(),
                start_char=data["start_char"],
                end_char=data["end_char"],
                token_count=None
            )
            chunks.append(chunk)

        return chunks


class TokenTextSplitter(ChunkerBase):
    """Split text by token count using tiktoken (PRODUCTION RECOMMENDED).

    This is the most accurate splitter for working with LLM APIs as it
    uses the same tokenizer as the target model.

    Args:
        chunk_size: Maximum chunk size in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        model_name: Model name for tiktoken encoding

    Example:
        >>> splitter = TokenTextSplitter(
        ...     chunk_size=512,
        ...     chunk_overlap=50,
        ...     model_name="gpt-4"
        ... )
        >>> chunks = splitter.split_text("Your long document...")

    Note:
        Requires tiktoken: pip install tiktoken
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: str = "gpt-4",
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.model_name = model_name

        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(model_name)
        except ImportError:
            raise ImportError(
                "tiktoken is required for TokenTextSplitter. "
                "Install it with: pip install tiktoken"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize tiktoken for model '{model_name}': {e}")

    def split_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into chunks by token count.

        Args:
            text: Text to split
            document_id: Optional document identifier
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Encode text to tokens
        tokens = self._encoding.encode(text)

        # First pass: create chunk data without Chunk objects
        chunk_data = []
        start_idx = 0

        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode to text
            chunk_text = self._encoding.decode(chunk_tokens)

            # Skip empty or whitespace-only chunks
            if chunk_text.strip():
                # Store chunk data
                chunk_data.append({
                    "content": chunk_text,
                    "token_count": len(chunk_tokens)
                })

            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap

            # Prevent infinite loop (if overlap >= chunk_size)
            if self.chunk_overlap >= self.chunk_size:
                break

        # Second pass: create Chunk objects with correct total_chunks
        total_chunks = len(chunk_data)
        chunks = []
        for idx, data in enumerate(chunk_data):
            chunk = Chunk(
                content=data["content"],
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                metadata=metadata.copy(),
                start_char=None,  # Token-based, not character-based
                end_char=None,
                token_count=data["token_count"]
            )
            chunks.append(chunk)

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self._encoding.encode(text))


# Register splitters with factory
from .factory import ChunkerFactory

ChunkerFactory.register("character", RecursiveCharacterTextSplitter)
ChunkerFactory.register("sentence", SentenceTextSplitter)
ChunkerFactory.register("token", TokenTextSplitter)
