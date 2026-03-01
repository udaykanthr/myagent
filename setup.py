from setuptools import setup, find_packages

setup(
    name="multi_agent_coder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pyyaml",
        "textual",
        # Phase 1 — Code Graph (Local KB)
        "networkx>=3.0",
        "tree-sitter>=0.22",
        "tree-sitter-python",
        "tree-sitter-javascript",
        "tree-sitter-typescript",
        "tree-sitter-java",
        "tree-sitter-c",
        "tree-sitter-cpp",
        "tree-sitter-go",
        "tree-sitter-rust",
        "tree-sitter-ruby",
        "tree-sitter-php",
        "tree-sitter-c-sharp",
        "watchdog>=3.0",
        "tqdm>=4.60",
    ],
    extras_require={
        # Phase 2 — Semantic Layer (install separately when needed)
        "semantic": [
            "openai>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentchanti=multi_agent_coder.orchestrator.cli:main",
        ],
    },
    author="Uday Kanth",
    description="A multi-agent coder that connects to local LLMs.",
)
