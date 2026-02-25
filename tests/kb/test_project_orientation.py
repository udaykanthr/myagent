"""
Unit tests for multi_agent_coder.kb.project_orientation

Tests ProjectProfile formatting and ProjectOrientation detection logic
across different project types (React/JS, Python, Java, Go, Rust).
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest

from multi_agent_coder.kb.project_orientation import (
    ProjectOrientation,
    ProjectProfile,
)


# ---------------------------------------------------------------------------
# ProjectProfile tests
# ---------------------------------------------------------------------------

class TestProjectProfile:

    def test_default_values(self):
        p = ProjectProfile()
        assert p.language == "unknown"
        assert p.framework is None
        assert p.source_root == "src"
        assert p.test_frameworks == []
        assert p.entry_points == []

    def test_source_extensions_typescript(self):
        p = ProjectProfile(language="typescript")
        assert ".ts" in p.source_extensions
        assert ".tsx" in p.source_extensions

    def test_source_extensions_python(self):
        p = ProjectProfile(language="python")
        assert p.source_extensions == [".py"]

    def test_source_extensions_unknown_falls_back_to_js(self):
        p = ProjectProfile(language="unknown")
        assert p.source_extensions == [".js"]

    def test_format_for_prompt_contains_project_name(self):
        p = ProjectProfile(project_name="my-app", project_version="1.0.0")
        text = p.format_for_prompt()
        assert "my-app" in text
        assert "1.0.0" in text

    def test_format_for_prompt_contains_language(self):
        p = ProjectProfile(language="typescript", framework="react")
        text = p.format_for_prompt()
        assert "typescript" in text
        assert "react" in text

    def test_format_for_prompt_contains_directory_structure(self):
        p = ProjectProfile(
            project_root="/home/user/project",
            source_root="my-app/src",
            source_root_absolute="/home/user/project/my-app/src",
        )
        text = p.format_for_prompt()
        assert "/home/user/project/my-app/src" in text
        assert "DIRECTORY STRUCTURE" in text

    def test_format_for_prompt_contains_commands(self):
        p = ProjectProfile(
            test_command="jest --coverage",
            build_command="react-scripts build",
            dev_command="react-scripts start",
            package_manager="npm",
        )
        text = p.format_for_prompt()
        assert "npm run test" in text
        assert "jest --coverage" in text
        assert "npm run dev" in text
        assert "npm run build" in text

    def test_format_for_prompt_contains_test_frameworks(self):
        p = ProjectProfile(
            test_frameworks=["jest", "@testing-library/react"],
        )
        text = p.format_for_prompt()
        assert "jest" in text
        assert "@testing-library/react" in text

    def test_format_for_prompt_strict_rules(self):
        p = ProjectProfile(
            language="typescript",
            source_root_absolute="/proj/my-app/src",
            test_frameworks=["jest"],
        )
        text = p.format_for_prompt()
        assert "STRICT RULES" in text
        assert "NEVER create files outside" in text
        assert ".ts or .tsx" in text
        assert "NEVER use a different language" in text

    def test_format_for_prompt_header_and_footer(self):
        p = ProjectProfile()
        text = p.format_for_prompt()
        assert text.startswith("=== PROJECT CONTEXT")
        assert text.endswith("=== END PROJECT CONTEXT ===")

    def test_format_for_prompt_no_framework(self):
        p = ProjectProfile(language="python")
        text = p.format_for_prompt()
        assert "Framework:" not in text

    def test_format_for_prompt_with_test_root(self):
        p = ProjectProfile(
            project_root="/proj",
            test_root="__tests__",
        )
        text = p.format_for_prompt()
        assert "__tests__" in text

    def test_format_for_prompt_with_entry_points(self):
        p = ProjectProfile(entry_points=["src/index.tsx", "src/App.tsx"])
        text = p.format_for_prompt()
        assert "src/index.tsx" in text
        assert "src/App.tsx" in text


# ---------------------------------------------------------------------------
# ProjectOrientation — React/JS project
# ---------------------------------------------------------------------------

class TestOrientationReactProject:

    @pytest.fixture
    def react_project(self, tmp_path):
        """Create a minimal React/TypeScript project."""
        pkg = {
            "name": "my-react-app",
            "version": "2.1.0",
            "scripts": {
                "test": "react-scripts test",
                "build": "react-scripts build",
                "start": "react-scripts start",
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
            },
            "devDependencies": {
                "typescript": "^5.0.0",
                "@testing-library/react": "^14.0.0",
                "jest": "^29.0.0",
            },
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "tsconfig.json").write_text("{}")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.tsx").write_text("// entry")
        (tmp_path / "src" / "App.tsx").write_text("// app")
        (tmp_path / "src" / "__tests__").mkdir()
        return tmp_path

    def test_detects_react_typescript(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        assert p.language == "typescript"
        assert p.framework == "react"

    def test_detects_project_name(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        assert p.project_name == "my-react-app"
        assert p.project_version == "2.1.0"

    def test_detects_test_frameworks(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        assert "jest" in p.test_frameworks
        assert "@testing-library/react" in p.test_frameworks

    def test_detects_source_root(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        assert p.source_root == "src"
        assert p.source_root_absolute == str(react_project / "src")

    def test_detects_test_root(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        assert p.test_root == "src/__tests__"

    def test_detects_entry_points(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        assert "src/index.tsx" in p.entry_points
        assert "src/App.tsx" in p.entry_points

    def test_detects_scripts(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        assert p.test_command == "react-scripts test"
        assert p.build_command == "react-scripts build"
        assert p.dev_command == "react-scripts start"

    def test_caches_profile(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p1 = o.get_profile()
        p2 = o.get_profile()
        assert p1 is p2

    def test_full_prompt_output(self, react_project):
        o = ProjectOrientation(graph=None, project_root=str(react_project))
        p = o.get_profile()
        text = p.format_for_prompt()
        assert "typescript" in text
        assert "react" in text
        assert "jest" in text
        assert str(react_project / "src") in text


# ---------------------------------------------------------------------------
# ProjectOrientation — Python project
# ---------------------------------------------------------------------------

class TestOrientationPythonProject:

    @pytest.fixture
    def python_project(self, tmp_path):
        """Create a minimal Python project."""
        (tmp_path / "requirements.txt").write_text("flask>=2.0\npytest>=7.0\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("# flask app")
        (tmp_path / "tests").mkdir()
        (tmp_path / "conftest.py").write_text("# pytest conf")
        return tmp_path

    def test_detects_python(self, python_project):
        o = ProjectOrientation(graph=None, project_root=str(python_project))
        p = o.get_profile()
        assert p.language == "python"

    def test_detects_flask_framework(self, python_project):
        o = ProjectOrientation(graph=None, project_root=str(python_project))
        p = o.get_profile()
        assert p.framework == "flask"

    def test_detects_pytest(self, python_project):
        o = ProjectOrientation(graph=None, project_root=str(python_project))
        p = o.get_profile()
        assert "pytest" in p.test_frameworks
        assert p.test_command == "pytest"

    def test_detects_test_root(self, python_project):
        o = ProjectOrientation(graph=None, project_root=str(python_project))
        p = o.get_profile()
        assert p.test_root == "tests"


# ---------------------------------------------------------------------------
# ProjectOrientation — Next.js project
# ---------------------------------------------------------------------------

class TestOrientationNextjsProject:

    @pytest.fixture
    def nextjs_project(self, tmp_path):
        pkg = {
            "name": "my-next-app",
            "version": "1.0.0",
            "scripts": {"dev": "next dev", "build": "next build"},
            "dependencies": {"next": "^14.0.0", "react": "^18.0.0"},
            "devDependencies": {"typescript": "^5.0.0", "vitest": "^1.0.0"},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "tsconfig.json").write_text("{}")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.ts").write_text("// entry")
        return tmp_path

    def test_detects_nextjs(self, nextjs_project):
        o = ProjectOrientation(graph=None, project_root=str(nextjs_project))
        p = o.get_profile()
        assert p.framework == "nextjs"
        assert p.language == "typescript"
        assert "vitest" in p.test_frameworks


# ---------------------------------------------------------------------------
# ProjectOrientation — package manager detection
# ---------------------------------------------------------------------------

class TestPackageManagerDetection:

    def test_detects_yarn(self, tmp_path):
        pkg = {"name": "test", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "yarn.lock").write_text("")
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.package_manager == "yarn"

    def test_detects_pnpm(self, tmp_path):
        pkg = {"name": "test", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "pnpm-lock.yaml").write_text("")
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.package_manager == "pnpm"

    def test_detects_bun(self, tmp_path):
        pkg = {"name": "test", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "bun.lockb").write_text("")
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.package_manager == "bun"

    def test_defaults_to_npm(self, tmp_path):
        pkg = {"name": "test", "version": "1.0.0"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.package_manager == "npm"


# ---------------------------------------------------------------------------
# ProjectOrientation — Go, Rust, Java projects
# ---------------------------------------------------------------------------

class TestOrientationOtherLanguages:

    def test_go_project(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/myapp\n\ngo 1.21\n")
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.language == "go"

    def test_rust_project(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "myapp"\n')
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.language == "rust"

    def test_java_spring_project(self, tmp_path):
        (tmp_path / "pom.xml").write_text(
            "<project><dependency>spring-boot-starter</dependency></project>"
        )
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.language == "java"
        assert p.framework == "spring"


# ---------------------------------------------------------------------------
# ProjectOrientation — graph-based detection
# ---------------------------------------------------------------------------

class TestOrientationWithGraph:

    def test_uses_graph_file_nodes(self, tmp_path):
        """Source layout detection uses graph FILE nodes."""
        from multi_agent_coder.kb.local.graph import CodeGraph, NodeType
        from multi_agent_coder.kb.local.parser import ParsedFile

        g = CodeGraph()
        # Simulate files under my-app/src/
        for fname in ["App.tsx", "index.tsx", "utils.ts"]:
            path = str(tmp_path / "my-app" / "src" / fname)
            g.add_parsed_file(ParsedFile(
                path=path, language="typescript", hash=fname,
            ))

        # Create the directory so source layout detects it
        (tmp_path / "my-app" / "src").mkdir(parents=True, exist_ok=True)
        # Create package.json for language detection
        pkg = {
            "name": "test-app",
            "version": "1.0.0",
            "dependencies": {"react": "^18.0.0"},
            "devDependencies": {"typescript": "^5.0.0"},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "tsconfig.json").write_text("{}")

        o = ProjectOrientation(graph=g, project_root=str(tmp_path))
        p = o.get_profile()

        # Graph should detect my-app/src via file nodes
        assert p.language == "typescript"
        assert p.framework == "react"

    def test_get_all_file_nodes_works(self):
        """Verify the new get_all_file_nodes method on CodeGraph."""
        from multi_agent_coder.kb.local.graph import CodeGraph, NodeType
        from multi_agent_coder.kb.local.parser import ParsedFile, ParsedFunction

        g = CodeGraph()
        g.add_parsed_file(ParsedFile(
            path="src/a.py", language="python", hash="aaa",
            functions=[
                ParsedFunction(
                    name="func_a", file_path="src/a.py",
                    line_start=1, line_end=10,
                ),
            ],
        ))
        g.add_parsed_file(ParsedFile(
            path="src/b.py", language="python", hash="bbb",
        ))

        file_nodes = g.get_all_file_nodes()
        assert len(file_nodes) == 2
        paths = {n["path"] for n in file_nodes}
        assert "src/a.py" in paths
        assert "src/b.py" in paths
        # Each node should have node_type == FILE
        for node in file_nodes:
            assert node["node_type"] == NodeType.FILE


# ---------------------------------------------------------------------------
# ProjectOrientation — empty / unknown project
# ---------------------------------------------------------------------------

class TestOrientationEmptyProject:

    def test_empty_directory(self, tmp_path):
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        assert p.language == "unknown"
        assert p.framework is None
        assert p.test_frameworks == []

    def test_format_for_prompt_still_works(self, tmp_path):
        o = ProjectOrientation(graph=None, project_root=str(tmp_path))
        p = o.get_profile()
        text = p.format_for_prompt()
        assert "PROJECT CONTEXT" in text
        assert "unknown" in text


# ---------------------------------------------------------------------------
# Integration: pipeline signature compatibility
# ---------------------------------------------------------------------------

class TestPipelineSignatureCompat:

    def test_execute_step_accepts_project_profile(self):
        """_execute_step should accept project_profile kwarg."""
        import inspect
        from multi_agent_coder.orchestrator.pipeline import _execute_step

        sig = inspect.signature(_execute_step)
        assert "project_profile" in sig.parameters

    def test_run_diagnosis_loop_accepts_project_profile(self):
        """_run_diagnosis_loop should accept project_profile kwarg."""
        import inspect
        from multi_agent_coder.orchestrator.pipeline import _run_diagnosis_loop

        sig = inspect.signature(_run_diagnosis_loop)
        assert "project_profile" in sig.parameters
