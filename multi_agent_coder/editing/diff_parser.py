"""
Diff parser — parses the structured diff format returned by the LLM
into actionable patch operations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Markers
_DIFF_START = "@@DIFF_START@@"
_DIFF_END = "@@DIFF_END@@"

# Patterns
_FILE_PATTERN = re.compile(r"^FILE:\s*(.+)$", re.MULTILINE)
_ORIGINAL_PATTERN = re.compile(
    r"^<{7}\s*ORIGINAL\s*\(line\s+(\d+)\)", re.MULTILINE
)
_SEPARATOR = "======="
_UPDATED_PATTERN = re.compile(r"^>{7}\s*UPDATED", re.MULTILINE)


@dataclass
class DiffHunk:
    """A single diff hunk: original lines → replacement lines."""
    line_number: int           # 1-indexed line number from ORIGINAL marker
    original_lines: list[str] = field(default_factory=list)
    replacement_lines: list[str] = field(default_factory=list)

    @property
    def is_insertion(self) -> bool:
        return len(self.original_lines) == 0

    @property
    def is_deletion(self) -> bool:
        return len(self.replacement_lines) == 0


@dataclass
class FilePatch:
    """All diff hunks for a single file."""
    file_path: str
    hunks: list[DiffHunk] = field(default_factory=list)


@dataclass
class ParsedDiff:
    """The complete parsed diff from an LLM response."""
    file_patches: list[FilePatch] = field(default_factory=list)
    parse_successful: bool = True
    parse_errors: list[str] = field(default_factory=list)


class DiffParser:
    """Parse structured diffs from LLM responses."""

    def parse(self, llm_response: str) -> ParsedDiff | None:
        """Parse a structured diff from the LLM response.

        Parameters
        ----------
        llm_response:
            The raw LLM response text.

        Returns
        -------
        ParsedDiff | None
            Parsed diff, or None if the response doesn't contain valid
            diff markers (triggers fallback to full-file mode).
        """
        # Find diff block
        start_idx = llm_response.find(_DIFF_START)
        end_idx = llm_response.find(_DIFF_END)

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.warning(
                "[DiffEdit] No valid diff markers found in LLM response"
            )
            return None

        diff_block = llm_response[start_idx + len(_DIFF_START) : end_idx].strip()
        if not diff_block:
            logger.warning("[DiffEdit] Empty diff block")
            return None

        return self._parse_diff_block(diff_block)

    def validate(
        self,
        parsed: ParsedDiff,
        file_contents: dict[str, list[str]],
    ) -> ParsedDiff | None:
        """Validate parsed diff hunks against actual file contents.

        Parameters
        ----------
        parsed:
            The parsed diff to validate.
        file_contents:
            Mapping of file_path → list of file lines.

        Returns
        -------
        ParsedDiff | None
            The validated diff with invalid hunks removed, or None if
            more than 50% of hunks are invalid.
        """
        total_hunks = 0
        invalid_hunks = 0
        validated_patches: list[FilePatch] = []

        for patch in parsed.file_patches:
            lines = file_contents.get(patch.file_path, [])
            valid_hunks: list[DiffHunk] = []

            for hunk in patch.hunks:
                total_hunks += 1
                if self._validate_hunk(hunk, lines):
                    valid_hunks.append(hunk)
                else:
                    invalid_hunks += 1
                    parsed.parse_errors.append(
                        f"Hunk at line {hunk.line_number} in "
                        f"{patch.file_path} does not match file content"
                    )
                    logger.warning(
                        "[DiffEdit] Invalid hunk at line %d in %s: "
                        "original lines don't match",
                        hunk.line_number, patch.file_path,
                    )

            if valid_hunks:
                validated_patches.append(FilePatch(
                    file_path=patch.file_path,
                    hunks=valid_hunks,
                ))

        if total_hunks > 0 and invalid_hunks / total_hunks > 0.5:
            logger.warning(
                "[DiffEdit] >50%% hunks invalid (%d/%d), aborting",
                invalid_hunks, total_hunks,
            )
            return None

        parsed.file_patches = validated_patches
        parsed.parse_successful = len(validated_patches) > 0
        return parsed

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_diff_block(self, block: str) -> ParsedDiff:
        """Parse the content between @@DIFF_START@@ and @@DIFF_END@@."""
        result = ParsedDiff()
        current_file: str | None = None
        current_patch: FilePatch | None = None

        # Split into file sections
        file_sections = re.split(r"^FILE:\s*", block, flags=re.MULTILINE)

        for section in file_sections:
            section = section.strip()
            if not section:
                continue

            # First line is the file path
            lines = section.split("\n")
            file_path = lines[0].strip()
            if not file_path:
                continue

            current_patch = FilePatch(file_path=file_path)
            body = "\n".join(lines[1:])

            # Parse hunks within this file section
            hunks = self._parse_hunks(body, file_path)
            current_patch.hunks = hunks

            if hunks:
                result.file_patches.append(current_patch)
            else:
                result.parse_errors.append(
                    f"No valid hunks parsed for {file_path}"
                )

        result.parse_successful = len(result.file_patches) > 0
        return result

    def _parse_hunks(self, body: str, file_path: str) -> list[DiffHunk]:
        """Parse all hunks from a file section body."""
        hunks: list[DiffHunk] = []

        # Find all ORIGINAL markers
        orig_matches = list(_ORIGINAL_PATTERN.finditer(body))
        if not orig_matches:
            return hunks

        for i, orig_match in enumerate(orig_matches):
            line_number = int(orig_match.group(1))

            # Determine the end of this hunk (start of next ORIGINAL or end of body)
            hunk_start = orig_match.end()
            if i + 1 < len(orig_matches):
                hunk_end = orig_matches[i + 1].start()
            else:
                hunk_end = len(body)

            hunk_text = body[hunk_start:hunk_end]

            # Split on separator
            sep_idx = hunk_text.find(_SEPARATOR)
            if sep_idx == -1:
                logger.warning(
                    "[DiffEdit] Missing separator in hunk at line %d of %s",
                    line_number, file_path,
                )
                continue

            original_text = hunk_text[:sep_idx]
            remaining = hunk_text[sep_idx + len(_SEPARATOR):]

            # Find UPDATED marker
            updated_match = _UPDATED_PATTERN.search(remaining)
            if updated_match:
                replacement_text = remaining[:updated_match.start()]
            else:
                replacement_text = remaining

            # Clean up lines
            original_lines = self._clean_lines(original_text)
            replacement_lines = self._clean_lines(replacement_text)

            hunks.append(DiffHunk(
                line_number=line_number,
                original_lines=original_lines,
                replacement_lines=replacement_lines,
            ))

        return hunks

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        """Clean and split text into lines, preserving content."""
        lines = text.split("\n")
        # Remove leading/trailing empty lines but preserve internal ones
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return lines

    @staticmethod
    def _validate_hunk(hunk: DiffHunk, file_lines: list[str]) -> bool:
        """Check if hunk's original lines match the file at the given line number."""
        if hunk.is_insertion:
            # Insertions are valid as long as the line number is within range
            return 1 <= hunk.line_number <= len(file_lines) + 1

        start = hunk.line_number - 1  # Convert to 0-indexed
        end = start + len(hunk.original_lines)

        if start < 0 or end > len(file_lines):
            return False

        # Compare original lines against actual file content
        for i, orig_line in enumerate(hunk.original_lines):
            actual_line = file_lines[start + i].rstrip("\n").rstrip("\r")
            if orig_line.rstrip() != actual_line.rstrip():
                return False

        return True
