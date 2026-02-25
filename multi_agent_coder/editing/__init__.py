"""Diff-aware file editing — surgical edits via code graph + LLM diffs."""

from .scope_resolver import ScopeResolver, EditScope, SymbolRange
from .context_slicer import ContextSlicer, FileSlice, SliceBlock
from .diff_parser import DiffParser, ParsedDiff, FilePatch, DiffHunk
from .patch_applier import PatchApplier, ApplyResult
from .chunk_editor import ChunkEditor, FileChunk, ChunkEditResponse
from .metrics import log_edit_metric, read_edit_stats

__all__ = [
    "ScopeResolver", "EditScope", "SymbolRange",
    "ContextSlicer", "FileSlice", "SliceBlock",
    "DiffParser", "ParsedDiff", "FilePatch", "DiffHunk",
    "PatchApplier", "ApplyResult",
    "ChunkEditor", "FileChunk", "ChunkEditResponse",
    "log_edit_metric", "read_edit_stats",
]
