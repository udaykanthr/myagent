"""
Context slicer — extracts minimal file slices to send to the LLM
based on a resolved EditScope.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .scope_resolver import EditScope, SymbolRange

logger = logging.getLogger(__name__)

# Patterns to detect import lines
_IMPORT_PATTERNS = [
    re.compile(r"^\s*(import\s|from\s\S+\s+import)"),   # Python
    re.compile(r"^\s*(const|let|var)\s+.*=\s*require\("),  # JS CJS
    re.compile(r"^\s*import\s+.+\s+from\s+"),            # JS ESM
    re.compile(r"^\s*import\s+['\"]"),                    # Go / Java
    re.compile(r"^\s*using\s+"),                          # C#
    re.compile(r"^\s*#include\s+"),                       # C/C++
    re.compile(r"^\s*use\s+"),                            # Rust
    re.compile(r"^\s*require\s+"),                        # Ruby
]

# Language detection by file extension
_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".cs": "csharp", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp",
}


@dataclass
class SliceBlock:
    """A contiguous block of source lines from a file."""
    line_start: int       # 1-indexed line number in the full file
    line_end: int
    content: str          # actual source lines
    editable: bool        # True = LLM may change this
    symbol_name: str
    annotation: str       # comment prepended for LLM clarity


@dataclass
class FileSlice:
    """A structured slice of a file for LLM consumption."""
    file_path: str
    language: str
    total_lines: int
    slices: list[SliceBlock] = field(default_factory=list)
    imports_block: str = ""
    class_signature: Optional[str] = None


class ContextSlicer:
    """Extract minimal file content to send to the LLM."""

    def slice_file(
        self,
        file_path: str,
        scope: EditScope,
        context_lines: int = 5,
    ) -> FileSlice:
        """Slice a single file based on the resolved scope.

        Parameters
        ----------
        file_path:
            Path to the file to slice.
        scope:
            The resolved EditScope containing primary and context symbols.
        context_lines:
            Number of lines to include before and after each primary symbol.

        Returns
        -------
        FileSlice
            The structured file slice ready for prompt formatting.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except OSError as exc:
            logger.warning("[DiffEdit] Cannot read %s: %s", file_path, exc)
            return FileSlice(
                file_path=file_path,
                language=self._detect_language(file_path),
                total_lines=0,
            )

        total = len(all_lines)
        language = self._detect_language(file_path)

        # Extract imports block
        imports_end = self._find_imports_end(all_lines)
        imports_block = "".join(all_lines[:imports_end]).rstrip()

        # Collect class signatures for methods
        class_sig = None
        for sym in scope.primary_symbols:
            if sym.file_path == file_path and sym.parent_class:
                class_sig = self._find_class_signature(all_lines, sym.parent_class)
                break

        # Build slice blocks for this file
        blocks: list[SliceBlock] = []

        # Primary symbols (editable)
        for sym in scope.primary_symbols:
            if sym.file_path != file_path:
                continue
            start = max(1, sym.line_start - context_lines)
            end = min(total, sym.line_end + context_lines)
            content = "".join(all_lines[start - 1 : end])
            annotation = (
                f"# ═══ EDITABLE: {sym.symbol_name} "
                f"(lines {sym.line_start}-{sym.line_end}) ═══"
            )
            blocks.append(SliceBlock(
                line_start=start,
                line_end=end,
                content=content,
                editable=True,
                symbol_name=sym.symbol_name,
                annotation=annotation,
            ))

        # Context symbols (read-only — signature + docstring only)
        for sym in scope.context_symbols:
            if sym.file_path != file_path:
                continue
            sig_content, sig_end = self._extract_signature_and_docstring(
                all_lines, sym.line_start, sym.line_end
            )
            annotation = (
                f"# ─── CONTEXT ONLY (do not edit): {sym.symbol_name} ───"
            )
            blocks.append(SliceBlock(
                line_start=sym.line_start,
                line_end=sig_end,
                content=sig_content,
                editable=False,
                symbol_name=sym.symbol_name,
                annotation=annotation,
            ))

        # Sort blocks by line_start, merge overlapping
        blocks.sort(key=lambda b: b.line_start)
        blocks = self._merge_overlapping(blocks, all_lines)

        return FileSlice(
            file_path=file_path,
            language=language,
            total_lines=total,
            slices=blocks,
            imports_block=imports_block,
            class_signature=class_sig,
        )

    def slice_files(
        self,
        scopes: dict[str, EditScope],
    ) -> dict[str, FileSlice]:
        """Slice multiple files.

        Parameters
        ----------
        scopes:
            Mapping of file_path → EditScope.

        Returns
        -------
        dict[str, FileSlice]
            Mapping of file_path → FileSlice.
        """
        result: dict[str, FileSlice] = {}
        for file_path, scope in scopes.items():
            result[file_path] = self.slice_file(file_path, scope)
        return result

    def format_for_prompt(
        self,
        slices: dict[str, FileSlice],
    ) -> str:
        """Format file slices into the structured prompt text the LLM expects.

        Parameters
        ----------
        slices:
            Mapping of file_path → FileSlice.

        Returns
        -------
        str
            Formatted prompt text with annotated slices.
        """
        parts: list[str] = []

        for file_path, fslice in slices.items():
            parts.append(
                f"=== FILE: {file_path} ({fslice.total_lines} lines total) ==="
            )
            parts.append(f"Language: {fslice.language}")
            parts.append("")

            # Imports
            if fslice.imports_block:
                parts.append("[IMPORTS]")
                parts.append(fslice.imports_block)
                parts.append("")

            # Class signature
            if fslice.class_signature:
                parts.append(fslice.class_signature)
                parts.append("")

            # Slices with omission markers
            prev_end = 0
            imports_end = fslice.imports_block.count("\n") + 1 if fslice.imports_block else 0

            for block in fslice.slices:
                gap_start = max(prev_end, imports_end)
                gap = block.line_start - gap_start - 1
                if gap > 3:
                    parts.append(f"# ... [{gap} lines omitted] ...")
                    parts.append("")

                parts.append(block.annotation)
                parts.append(block.content.rstrip())
                parts.append("")

                prev_end = block.line_end

            # Trailing omission
            trailing = fslice.total_lines - prev_end
            if trailing > 3:
                parts.append(f"# ... [{trailing} lines omitted] ...")

            parts.append(f"=== END FILE ===")
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language(file_path: str) -> str:
        """Detect language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return _EXT_TO_LANG.get(ext, "unknown")

    @staticmethod
    def _find_imports_end(lines: list[str]) -> int:
        """Find the line index (0-based) where imports end."""
        last_import = 0
        in_docstring = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip module docstrings at top of file
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    continue
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_docstring = True
                continue
            if in_docstring:
                continue

            # Skip blank lines and comments
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue

            # Check if this is an import line
            is_import = any(p.match(line) for p in _IMPORT_PATTERNS)
            if is_import:
                last_import = i + 1  # 1 past the last import line
            elif last_import > 0:
                # First non-import, non-blank line after imports
                break

        return last_import

    @staticmethod
    def _find_class_signature(lines: list[str], class_name: str) -> Optional[str]:
        """Find the class definition line for a given class name."""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"class {class_name}") and (
                stripped.endswith(":") or "(" in stripped
            ):
                return stripped
        return None

    @staticmethod
    def _extract_signature_and_docstring(
        lines: list[str],
        line_start: int,
        line_end: int,
    ) -> tuple[str, int]:
        """Extract the signature line and docstring of a symbol.

        Returns (content, end_line_number).
        """
        if line_start < 1 or line_start > len(lines):
            return "", line_start

        # Start with the signature line
        sig_line = lines[line_start - 1]
        content_lines = [sig_line]
        end = line_start

        # Look for docstring immediately after
        for i in range(line_start, min(line_end, len(lines))):
            stripped = lines[i].strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                content_lines.append(lines[i])
                end = i + 1
                # Multi-line docstring
                if stripped.count(quote) < 2:
                    for j in range(i + 1, min(line_end, len(lines))):
                        content_lines.append(lines[j])
                        end = j + 1
                        if quote in lines[j]:
                            break
                break
            elif stripped and not stripped.startswith("#"):
                break
            elif stripped.startswith("#"):
                content_lines.append(lines[i])
                end = i + 1

        return "".join(content_lines), end

    @staticmethod
    def _merge_overlapping(
        blocks: list[SliceBlock],
        all_lines: list[str],
    ) -> list[SliceBlock]:
        """Merge overlapping or adjacent slice blocks."""
        if len(blocks) <= 1:
            return blocks

        merged: list[SliceBlock] = [blocks[0]]
        for block in blocks[1:]:
            prev = merged[-1]
            if block.line_start <= prev.line_end + 1:
                # Merge: extend the previous block
                new_end = max(prev.line_end, block.line_end)
                new_content = "".join(all_lines[prev.line_start - 1 : new_end])
                # Keep editable if either block is editable
                new_editable = prev.editable or block.editable
                names = f"{prev.symbol_name}, {block.symbol_name}"
                annotation = prev.annotation if prev.editable else block.annotation
                merged[-1] = SliceBlock(
                    line_start=prev.line_start,
                    line_end=new_end,
                    content=new_content,
                    editable=new_editable,
                    symbol_name=names,
                    annotation=annotation,
                )
            else:
                merged.append(block)

        return merged
