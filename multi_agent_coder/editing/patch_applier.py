"""
Patch applier — applies parsed diff hunks surgically to files with
atomic writes and syntax validation.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from .diff_parser import DiffHunk, FilePatch, ParsedDiff

logger = logging.getLogger(__name__)


class PatchApplyError(Exception):
    """Raised when a patch cannot be applied cleanly."""


@dataclass
class ApplyResult:
    """Result of applying a parsed diff."""
    success: bool = False
    files_modified: list[str] = field(default_factory=list)
    hunks_applied: int = 0
    hunks_failed: int = 0
    failed_hunks: list[DiffHunk] = field(default_factory=list)
    syntax_valid: bool = True
    error: str = ""


class PatchApplier:
    """Apply parsed diff hunks surgically to files."""

    def __init__(
        self,
        fuzzy_match_window: int = 3,
        validate_syntax: bool = True,
        fallback_on_syntax_error: bool = True,
    ) -> None:
        self._fuzzy_window = fuzzy_match_window
        self._validate_syntax = validate_syntax
        self._fallback_on_syntax_error = fallback_on_syntax_error

    def apply(self, parsed_diff: ParsedDiff) -> ApplyResult:
        """Apply all file patches from a parsed diff.

        Multi-file patches are applied transactionally: if any file
        fails syntax validation, all changes are rolled back.

        Parameters
        ----------
        parsed_diff:
            The parsed diff from DiffParser.

        Returns
        -------
        ApplyResult
            Summary of the patch application.
        """
        result = ApplyResult()

        if not parsed_diff.file_patches:
            result.error = "No patches to apply"
            return result

        # Phase 1: Compute all patched contents without writing
        patched_files: dict[str, tuple[list[str], list[str]]] = {}  # path → (new_lines, old_lines)

        for patch in parsed_diff.file_patches:
            try:
                old_lines, new_lines, applied, failed, failed_hunks = (
                    self._apply_file_patch(patch)
                )
                patched_files[patch.file_path] = (new_lines, old_lines)
                result.hunks_applied += applied
                result.hunks_failed += failed
                result.failed_hunks.extend(failed_hunks)
            except Exception as exc:
                logger.warning(
                    "[DiffEdit] Failed to apply patch for %s: %s",
                    patch.file_path, exc,
                )
                result.error = f"Patch failed for {patch.file_path}: {exc}"
                return result

        # Phase 2: Validate syntax of all patched files
        if self._validate_syntax:
            for file_path, (new_lines, _) in patched_files.items():
                if not self._check_syntax(file_path, new_lines):
                    result.syntax_valid = False
                    result.error = f"Syntax error in patched {file_path}"
                    if self._fallback_on_syntax_error:
                        logger.warning(
                            "[DiffEdit] Syntax validation failed for %s, "
                            "aborting all patches",
                            file_path,
                        )
                        return result

        # Phase 3: Write all files atomically
        written: list[str] = []
        try:
            for file_path, (new_lines, old_lines) in patched_files.items():
                self._safe_write(file_path, new_lines)
                written.append(file_path)
        except Exception as exc:
            # Rollback: restore original content for already-written files
            logger.error(
                "[DiffEdit] Write failed for %s, rolling back %d files: %s",
                file_path, len(written), exc,
            )
            for rollback_path in written:
                _, old = patched_files[rollback_path]
                try:
                    self._safe_write(rollback_path, old)
                except Exception as rb_exc:
                    logger.error(
                        "[DiffEdit] Rollback failed for %s: %s",
                        rollback_path, rb_exc,
                    )
            result.error = f"Write failed: {exc}"
            return result

        result.success = True
        result.files_modified = written
        return result

    # ------------------------------------------------------------------
    # Single-file patch application
    # ------------------------------------------------------------------

    def _apply_file_patch(
        self,
        patch: FilePatch,
    ) -> tuple[list[str], list[str], int, int, list[DiffHunk]]:
        """Apply all hunks for a single file (in memory).

        Returns (old_lines, new_lines, applied_count, failed_count, failed_hunks).
        """
        with open(patch.file_path, "r", encoding="utf-8", errors="replace") as f:
            old_lines = f.readlines()

        new_lines = list(old_lines)
        applied = 0
        failed = 0
        failed_hunks: list[DiffHunk] = []

        # Sort hunks by line_number DESCENDING so we can apply bottom-up
        # without shifting line numbers
        sorted_hunks = sorted(patch.hunks, key=lambda h: h.line_number, reverse=True)

        for hunk in sorted_hunks:
            success = self._apply_hunk(new_lines, hunk)
            if success:
                applied += 1
            else:
                failed += 1
                failed_hunks.append(hunk)
                logger.warning(
                    "[DiffEdit] Hunk at line %d failed for %s",
                    hunk.line_number, patch.file_path,
                )

        return old_lines, new_lines, applied, failed, failed_hunks

    def _apply_hunk(self, lines: list[str], hunk: DiffHunk) -> bool:
        """Apply a single hunk to the lines array.

        Tries exact match first, then fuzzy match within ±fuzzy_window lines.
        """
        # Try exact match at the specified line number
        start = hunk.line_number - 1  # Convert to 0-indexed

        if hunk.is_insertion:
            # Insert replacement lines at the specified position
            insert_pos = min(start, len(lines))
            replacement = [l + "\n" if not l.endswith("\n") else l
                           for l in hunk.replacement_lines]
            lines[insert_pos:insert_pos] = replacement
            return True

        if self._lines_match(lines, start, hunk.original_lines):
            self._replace_lines(lines, start, hunk)
            return True

        # Try fuzzy match within ±window
        for offset in range(1, self._fuzzy_window + 1):
            for try_start in (start - offset, start + offset):
                if try_start < 0:
                    continue
                if self._lines_match(lines, try_start, hunk.original_lines):
                    logger.debug(
                        "[DiffEdit] Fuzzy match: hunk line %d matched at %d "
                        "(offset %+d)",
                        hunk.line_number, try_start + 1,
                        try_start - start,
                    )
                    self._replace_lines(lines, try_start, hunk)
                    return True

        return False

    @staticmethod
    def _lines_match(
        file_lines: list[str],
        start: int,
        original_lines: list[str],
    ) -> bool:
        """Check if original_lines match file_lines starting at start index."""
        if start < 0 or start + len(original_lines) > len(file_lines):
            return False

        for i, orig in enumerate(original_lines):
            actual = file_lines[start + i].rstrip("\n").rstrip("\r")
            if orig.rstrip() != actual.rstrip():
                return False

        return True

    @staticmethod
    def _replace_lines(
        lines: list[str],
        start: int,
        hunk: DiffHunk,
    ) -> None:
        """Replace lines in the array with the hunk's replacement lines."""
        replacement = [
            l + "\n" if not l.endswith("\n") else l
            for l in hunk.replacement_lines
        ]
        end = start + len(hunk.original_lines)
        lines[start:end] = replacement

    # ------------------------------------------------------------------
    # Syntax validation
    # ------------------------------------------------------------------

    def _check_syntax(self, file_path: str, lines: list[str]) -> bool:
        """Validate that the patched file has no syntax errors using tree-sitter."""
        try:
            from ..kb.local.parser import parse_file, EXTENSION_TO_LANGUAGE
        except ImportError:
            logger.debug("[DiffEdit] tree-sitter parser not available, skipping syntax check")
            return True

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in EXTENSION_TO_LANGUAGE:
            return True  # Can't validate — assume OK

        # Write to temp file, parse, clean up
        content = "".join(lines)
        tmp_dir = os.path.dirname(os.path.abspath(file_path))
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=ext, dir=tmp_dir, prefix=".agentchanti_syntax_"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            parsed = parse_file(tmp_path)
            if parsed.parse_error:
                logger.warning(
                    "[DiffEdit] Syntax error in patched %s: %s",
                    file_path, parsed.parse_error,
                )
                return False
            return True
        except Exception as exc:
            logger.debug(
                "[DiffEdit] Syntax check exception for %s: %s", file_path, exc
            )
            return True  # On exception, don't block
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Atomic file write
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_write(file_path: str, lines: list[str]) -> None:
        """Write lines to file atomically via temp file + rename."""
        content = "".join(lines)
        abs_path = os.path.abspath(file_path)
        tmp_path = abs_path + ".agentchanti_tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)

            # On Windows, os.rename fails if destination exists
            if os.path.exists(abs_path):
                shutil.move(tmp_path, abs_path)
            else:
                os.rename(tmp_path, abs_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
