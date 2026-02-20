from setuptools import setup, find_packages

setup(
    name="multi_agent_coder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pyyaml",
        "textual",
    ],
    entry_points={
        "console_scripts": [
            "agentchanti=multi_agent_coder.orchestrator.cli:main",
        ],
    },
    author="Uday Kanth",
    description="A multi-agent coder that connects to local LLMs.",
)
