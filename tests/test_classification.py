"""
Tests for the command extraction and classification utilities,
specifically heredoc handling in _extract_commands_from_text.
"""
import pytest

from multi_agent_coder.orchestrator.classification import (
    _extract_commands_from_text,
    _looks_like_command,
)


# ── Heredoc preservation tests ───────────────────────────────


def test_heredoc_preserved_as_single_command():
    """A heredoc block in triple backticks should be kept as one command."""
    text = (
        "FIX:\n"
        "```bash\n"
        "cat << 'EOF' > Makefile\n"
        "CC = gcc\n"
        "CFLAGS = -Wall\n"
        "all: snake\n"
        "EOF\n"
        "```\n"
    )
    commands = _extract_commands_from_text(text)
    assert len(commands) == 1
    assert "cat << 'EOF' > Makefile" in commands[0]
    assert "CC = gcc" in commands[0]
    assert "EOF" in commands[0]


def test_heredoc_with_unquoted_delimiter():
    """Heredoc with unquoted delimiter (cat << EOF) should also be preserved."""
    text = (
        "```bash\n"
        "cat << EOF > config.ini\n"
        "[settings]\n"
        "debug = true\n"
        "EOF\n"
        "```\n"
    )
    commands = _extract_commands_from_text(text)
    assert len(commands) == 1
    assert "cat << EOF > config.ini" in commands[0]
    assert "[settings]" in commands[0]


def test_heredoc_with_double_quoted_delimiter():
    """Heredoc with double-quoted delimiter (cat << \"END\") should be preserved."""
    text = (
        '```bash\n'
        'cat << "END" > file.txt\n'
        'some content\n'
        'END\n'
        '```\n'
    )
    commands = _extract_commands_from_text(text)
    assert len(commands) == 1
    assert 'cat << "END" > file.txt' in commands[0]


def test_heredoc_with_dash_delimiter():
    """Heredoc with <<- (tab-stripped) should be preserved."""
    text = (
        "```bash\n"
        "cat <<- MARKER > output.txt\n"
        "  indented content\n"
        "MARKER\n"
        "```\n"
    )
    commands = _extract_commands_from_text(text)
    assert len(commands) == 1
    assert "indented content" in commands[0]


def test_non_heredoc_block_still_splits_by_line():
    """Normal command blocks should still be split into individual commands."""
    text = (
        "```bash\n"
        "mkdir -p build\n"
        "npm install\n"
        "npm run build\n"
        "```\n"
    )
    commands = _extract_commands_from_text(text)
    assert len(commands) == 3
    assert "mkdir -p build" in commands
    assert "npm install" in commands
    assert "npm run build" in commands


def test_mixed_heredoc_and_inline_commands():
    """Heredoc in a code block + inline backtick commands should both be extracted."""
    text = (
        "First run `mkdir -p src` then:\n"
        "```bash\n"
        "cat << 'EOF' > src/main.c\n"
        "#include <stdio.h>\n"
        "int main() { return 0; }\n"
        "EOF\n"
        "```\n"
    )
    commands = _extract_commands_from_text(text)
    assert len(commands) == 2
    # The heredoc block
    heredoc_cmd = [c for c in commands if "cat <<" in c][0]
    assert "#include <stdio.h>" in heredoc_cmd
    # The inline command
    assert "mkdir -p src" in commands


def test_heredoc_followed_by_make_command():
    """Reproduces the exact scenario from the bug report."""
    text = (
        "FIX:\n"
        "```bash\n"
        "cat << 'EOF' > Makefile\n"
        "CC = gcc\n"
        "CFLAGS = -Wall -Wextra -Werror -Isrc\n"
        "LDFLAGS = -lncurses -lm\n"
        "\n"
        "all: snake\n"
        "\n"
        "snake: build/main.o build/game.o\n"
        "\t$(CC) build/main.o build/game.o -o snake $(LDFLAGS)\n"
        "\n"
        "build/%.o: src/%.c\n"
        "\tmkdir -p build\n"
        "\t$(CC) $(CFLAGS) -c $< -o $@\n"
        "\n"
        "clean:\n"
        "\trm -rf build snake test_logic\n"
        "\n"
        "test:\n"
        "\t$(CC) $(CFLAGS) tests/test_logic.c src/game.c -o test_logic $(LDFLAGS)\n"
        "\t./test_logic\n"
        "\n"
        ".PHONY: all clean test\n"
        "EOF\n"
        "make all\n"
        "```\n"
    )
    commands = _extract_commands_from_text(text)
    # The whole block is a heredoc, so it should be one command
    assert len(commands) == 1
    assert "cat << 'EOF' > Makefile" in commands[0]
    assert "make all" in commands[0]
    assert "CC = gcc" in commands[0]


# ── _looks_like_command basic tests ──────────────────────────


def test_looks_like_command_basic():
    """Basic commands should be recognized."""
    assert _looks_like_command("npm install")
    assert _looks_like_command("pip install flask")
    assert _looks_like_command("make all")
    assert _looks_like_command("mkdir -p build")


def test_looks_like_command_rejects_paths():
    """Bare file paths should not be treated as commands."""
    assert not _looks_like_command("src/main.py")
    assert not _looks_like_command("tests/test_logic.c")
