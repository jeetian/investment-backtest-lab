from pathlib import Path


def test_project_scope_docs_are_utf8_readable():
    docs = [
        Path("README.md"),
        Path("AGENTS.md"),
        Path("docs/PROJECT_SCOPE_ZH.md"),
        Path("docs/ROADMAP_ZH.md"),
        Path("docs/VALIDATION_STRATEGY_ZH.md"),
        Path("docs/QUICKSTART_ZH.md"),
        Path("docs/SETUP_ZH.md"),
        Path("docs/VALIDATION_ZH.md"),
    ]

    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert text.strip()


def test_readme_and_agents_link_to_scope_documents():
    readme = Path("README.md").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    for link in [
        "docs/PROJECT_SCOPE_ZH.md",
        "docs/ROADMAP_ZH.md",
        "docs/VALIDATION_STRATEGY_ZH.md",
    ]:
        assert link in readme
        assert link in agents
