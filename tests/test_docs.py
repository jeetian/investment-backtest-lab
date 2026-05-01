from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    *sorted((ROOT / "docs").glob("*.md")),
]

MOJIBAKE_MARKERS = (
    "\ufffd",
    "銝",
    "嚗",
    "雿",
    "蝑",
    "撠",
    "摰",
    "鞈",
    "瘛",
    "蝛",
)


def test_markdown_files_are_utf8_and_without_common_mojibake():
    for path in MARKDOWN_FILES:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"{path.name} should not be empty"
        for marker in MOJIBAKE_MARKERS:
            assert marker not in text, f"{path.name} contains mojibake marker {marker!r}"


def test_project_direction_links_are_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    for link in (
        "docs/PROJECT_SCOPE_ZH.md",
        "docs/ROADMAP_ZH.md",
        "docs/VALIDATION_STRATEGY_ZH.md",
        "docs/FRAMEWORK_HYGIENE_ZH.md",
        "docs/STRATEGY_RESEARCH_PLAN_ZH.md",
        "docs/ALLOCATION_WORKFLOW_ZH.md",
    ):
        assert link in readme

    assert "月度配置研究訊號" in agents
    assert "Optuna" in agents
