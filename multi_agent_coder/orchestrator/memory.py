"""
FileMemory — thread-safe file content tracker with context-window budget.
"""

import os
import re
import threading

from ..embedding_store import EmbeddingStore
from ..cli_display import log


def _estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token)."""
    return len(text) // 4


# Patterns for extracting function/class signatures across languages
_SKELETON_PATTERNS = {
    "python": [
        re.compile(r"^( *)(class\s+\w+[^:]*:)", re.MULTILINE),
        re.compile(r"^( *)(def\s+\w+\s*\([^)]*\)[^:]*:)", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"^( *)((?:export\s+)?(?:default\s+)?class\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^( *)((?:export\s+)?(?:async\s+)?function\s+\w+\s*\([^)]*\)[^{]*)\{", re.MULTILINE),
        re.compile(r"^( *)((?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)", re.MULTILINE),
    ],
    "typescript": None,  # reuses javascript
    "go": [
        re.compile(r"^()(func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\s*\([^)]*\)[^{]*)\{", re.MULTILINE),
        re.compile(r"^()(type\s+\w+\s+struct)\s*\{", re.MULTILINE),
        re.compile(r"^()(type\s+\w+\s+interface)\s*\{", re.MULTILINE),
    ],
    "java": [
        re.compile(r"^( *)((?:public|private|protected)?\s*(?:static\s+)?class\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^( *)((?:public|private|protected)\s+(?:static\s+)?(?:[\w<>\[\]]+\s+)?\w+\s*\([^)]*\)[^{]*)\{", re.MULTILINE),
    ],
    "rust": [
        re.compile(r"^( *)((?:pub\s+)?fn\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^( *)((?:pub\s+)?struct\s+\w+[^{;]*)\{", re.MULTILINE),
        re.compile(r"^( *)((?:pub\s+)?impl\s+[^{]*)\{", re.MULTILINE),
    ],
}
_SKELETON_PATTERNS["typescript"] = _SKELETON_PATTERNS["javascript"]

# Import line patterns
_IMPORT_LINE_PATTERNS = [
    re.compile(r"^\s*(import\s|from\s\S+\s+import)"),
    re.compile(r"^\s*(const|let|var)\s+.*=\s*require\("),
    re.compile(r"^\s*import\s+.+\s+from\s+"),
    re.compile(r"^\s*import\s+['\"]"),
    re.compile(r"^\s*using\s+"),
    re.compile(r"^\s*#include\s+"),
    re.compile(r"^\s*use\s+"),
    re.compile(r"^\s*require\s+"),
]

_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".java": "java", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
}


def _extract_file_skeleton(content: str, file_path: str = "") -> str:
    """Extract a structural skeleton from file content: imports + signatures."""
    lines = content.splitlines(True)
    total = len(lines)

    ext = os.path.splitext(file_path)[1].lower()
    lang = _EXT_TO_LANG.get(ext, "python")

    # Extract imports
    import_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        if any(p.match(line) for p in _IMPORT_LINE_PATTERNS):
            import_lines.append(stripped)

    # Extract signatures with line numbers
    patterns = _SKELETON_PATTERNS.get(lang, _SKELETON_PATTERNS["python"])
    if patterns is None:
        patterns = _SKELETON_PATTERNS["python"]

    signatures: list[str] = []
    for pattern in patterns:
        for m in pattern.finditer(content):
            indent = m.group(1)
            sig = m.group(2).strip()
            line_num = content[:m.start()].count("\n") + 1
            end_line = _find_block_end(lines, line_num - 1, len(indent))
            prefix = "  " if indent else ""
            signatures.append(f"{prefix}{sig} (line {line_num}-{end_line})")

    parts = [f"#### [FILE_STRUCTURE]: {file_path} ({total} lines)"]
    if import_lines:
        parts.append("[IMPORTS]")
        parts.extend(import_lines)
    if signatures:
        parts.append("[SYMBOLS]")
        parts.extend(signatures)

    return "\n".join(parts)


def _find_block_end(lines: list[str], start_idx: int, base_indent: int) -> int:
    """Find the last line of a block starting at start_idx."""
    last_content_line = start_idx + 1  # 1-indexed
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent <= base_indent and stripped:
            break
        last_content_line = i + 1
    return last_content_line


class FileMemory:
    """Tracks every file's path and current contents across all steps.

    When an EmbeddingStore is provided, context retrieval uses semantic
    similarity instead of simple filename matching.
    """

    def __init__(self, embedding_store: EmbeddingStore | None = None,
                 top_k: int = 5):
        self._files: dict[str, str] = {}   # filepath -> contents
        self._store = embedding_store
        self._top_k = top_k
        self._lock = threading.Lock()

    def update(self, files: dict[str, str]):
        """Store or overwrite file contents and update embeddings.

        Protected manifest files (package.json, go.mod, etc.) are skipped
        when they already exist on disk to prevent LLM-generated corruption.
        """
        from ..executor import Executor

        with self._lock:
            for fpath, content in files.items():
                basename = os.path.basename(fpath)
                if basename in Executor._PROTECTED_FILENAMES and os.path.isfile(fpath):
                    log.warning(f"[FileMemory] Skipping protected file update: "
                                f"{fpath} (already exists on disk)")
                    continue

                self._files[fpath] = content
                if self._store:
                    self._store.add(fpath, content)

    def get(self, filepath: str) -> str | None:
        with self._lock:
            return self._files.get(filepath)

    def all_files(self) -> dict[str, str]:
        with self._lock:
            return dict(self._files)

    def as_dict(self) -> dict[str, str]:
        """Snapshot for checkpoint serialization."""
        with self._lock:
            return dict(self._files)

    def related_context(self, step_text: str, max_tokens: int | None = None) -> str:
        """Build a compact context string with the most relevant files.

        Uses semantic search when embeddings are available, otherwise
        falls back to filename substring matching.
        """
        with self._lock:
            if self._store and self._store.size > 0:
                return self._semantic_context(step_text, max_tokens)
            return self._substring_context(step_text, max_tokens)

    def related_context_slim(self, step_text: str,
                            max_tokens: int | None = None) -> str:
        """Build context with file skeletons instead of full contents.

        For each relevant file, includes only imports and function/class
        signatures with line ranges.
        """
        with self._lock:
            scored = self._score_files(step_text)
            parts: list[str] = []
            budget = max_tokens or float("inf")
            used = 0
            for _score, fpath, content in scored:
                skeleton = _extract_file_skeleton(content, fpath)
                entry_tokens = _estimate_tokens(skeleton)
                if used + entry_tokens > budget:
                    break
                parts.append(skeleton)
                used += entry_tokens
            log.debug(f"[FileMemory] Slim context returned {len(parts)} skeletons "
                      f"({used} est. tokens)")
            return "\n\n".join(parts)

    def _semantic_context(self, step_text: str, max_tokens: int | None) -> str:
        results = self._store.search(step_text, top_k=self._top_k)
        parts: list[str] = []
        budget = max_tokens or float("inf")
        used = 0
        for fpath, score in results:
            content = self._files.get(fpath, "")
            if not content:
                continue
            entry = (
                f"#### [FILE]: {fpath} (relevance: {score:.2f})\n"
                f"```\n{content}\n```"
            )
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > budget:
                break
            parts.append(entry)
            used += entry_tokens
        log.debug(f"[FileMemory] Semantic search returned {len(parts)} files "
                  f"({used} est. tokens)")
        return "\n\n".join(parts)

    def _substring_context(self, step_text: str, max_tokens: int | None) -> str:
        """Score-based fallback matching when embeddings are unavailable."""
        scored = self._score_files(step_text)
        parts: list[str] = []
        budget = max_tokens or float("inf")
        used = 0
        for _score, fpath, content in scored:
            entry = f"#### [FILE]: {fpath}\n```\n{content}\n```"
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > budget:
                break
            parts.append(entry)
            used += entry_tokens
        log.debug(f"[FileMemory] Substring fallback returned {len(parts)} files "
                  f"({used} est. tokens)")
        return "\n\n".join(parts)

    def _score_files(self, step_text: str) -> list[tuple[int, str, str]]:
        """Score files by relevance to step text. Returns sorted list."""
        step_lower = step_text.lower()
        scored: list[tuple[int, str, str]] = []
        for fpath, content in self._files.items():
            score = 0
            basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
            stem, ext = os.path.splitext(basename)

            if fpath in step_text or basename in step_text:
                score += 100
            stem_parts = [p for p in stem.replace("-", "_").split("_") if len(p) > 2]
            for part in stem_parts:
                if part.lower() in step_lower:
                    score += 50
                    break
            if ext:
                ext_name = ext.lstrip(".").lower()
                if re.search(r'\b' + re.escape(ext_name) + r'\b', step_lower):
                    matches = list(re.finditer(r'\b' + re.escape(ext_name) + r'\b', step_lower))
                    has_standalone = any(
                        m.start() == 0 or step_lower[m.start() - 1] != '.'
                        for m in matches
                    )
                    if has_standalone:
                        score += 10
            if score > 0:
                scored.append((score, fpath, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def summary(self) -> str:
        with self._lock:
            if not self._files:
                return "(no files yet)"
            return ", ".join(self._files.keys())
