"""Final quality audit of the repository (run before packaging): python scripts/audit_project.py"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs", "dist", "build", ".venv", "geoxplain.egg-info"}
problems: list[str] = []


def files(suffixes: tuple[str, ...] | None = None):
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and not p.name.endswith(".egg-info"):
            if suffixes is None or p.suffix in suffixes:
                yield p


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        problems.append(msg)


REQUIRED = ["README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "requirements.txt", "requirements-dev.txt", ".gitignore",
            ".gitattributes", "CHANGELOG.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "AUTHORS.md",
            "reproduce.py", "PROJECT_MANIFEST.md", "data/README.md", "qgis/processing/README.md",
            ".github/workflows/tests.yml", ".github/workflows/lint.yml", ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md", ".github/pull_request_template.md",
            *[f"docs/{n}.md" for n in ["installation", "quickstart", "architecture", "models", "explainability",
                                        "spatial_explainability", "spatial_validation", "qgis", "cli", "api", "examples",
                                        "reproducibility", "limitations", "methodology"]],
            *[f"figures/{n}.png" for n in ["architecture", "workflow", "model_comparison", "shap_summary", "feature_importance",
                                            "spatial_prediction", "spatial_shap_panels", "spatial_validation_comparison",
                                            "model_performance", "uncertainty_explanation", "logo"]],
            "figures/logo.svg", "docs/assets/logo.png", "notebooks/01_end_to_end_workflow.ipynb",
            "results/example_run/report.html", "results/example_run/reproducibility.json", "data/samples.csv", "data/stack.tif"]
print("Required files")
for r in REQUIRED:
    check((ROOT / r).exists(), r)

print("Markdown links and images resolve")
link = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
for md in files((".md",)):
    for target in link.findall(md.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (md.parent / target.split("#")[0]).resolve()
        if not path.exists():
            check(False, f"{md.relative_to(ROOT)} -> {target}")
print("  (no output above the next heading means all local links resolve)")

print("Source hygiene")
bad = re.compile(r"\b(TODO|FIXME|XXX)\b|raise NotImplementedError|placeholder", re.I)
for py in (ROOT / "src").rglob("*.py"):
    for i, line in enumerate(py.read_text().splitlines(), 1):
        if bad.search(line):
            check(False, f"{py.relative_to(ROOT)}:{i}: {line.strip()[:80]}")
secret = re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{6,}['\"]|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
for p in files((".py", ".md", ".yml", ".yaml", ".toml", ".json", ".cff", ".txt", ".cfg")):
    if p.name == "audit_project.py":
        continue
    if secret.search(p.read_text(errors="ignore")):
        check(False, f"possible secret in {p.relative_to(ROOT)}")
check(not any(p.suffix == ".pyc" for p in files()), "no .pyc files")
big = [p for p in files() if p.stat().st_size > 8 * 1024 * 1024]
check(not big, f"no files over 8 MB {[str(p.relative_to(ROOT)) for p in big]}")

print("Metadata consistency")
version = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "src/geoxplain/_version.py").read_text()).group(1)
check('license = "GPL-3.0-or-later"' in (ROOT / "pyproject.toml").read_text(), "pyproject license is GPL-3.0-or-later")
check(f"version: {version}" in (ROOT / "CITATION.cff").read_text(), f"CITATION version == {version}")
check(f"## [{version}]" in (ROOT / "CHANGELOG.md").read_text(), "CHANGELOG has the release heading")
check("0009-0006-0917-3372" in (ROOT / "CITATION.cff").read_text(), "ORCID present in CITATION.cff")
check("GNU GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text()[:200], "LICENSE is the GPL text")
try:
    import yaml

    for wf in (ROOT / ".github/workflows").glob("*.yml"):
        doc = yaml.safe_load(wf.read_text())
        check(bool(doc.get("jobs")), f"workflow {wf.name} parses and has jobs")
except ImportError:
    print("  skip yaml (PyYAML missing)")

print("Documented CLI commands exist")
from geoxplain.cli import build_parser  # noqa: E402

commands = set(build_parser()._subparsers._group_actions[0].choices)
used = set()
for md in files((".md",)):
    used |= set(re.findall(r"geoxplain (?:-v )?([a-z][a-z-]+)", md.read_text()))
unknown = {u for u in used if u not in commands and u in {"generate-data", "train", "evaluate", "explain", "spatial-explain",
                                                             "compare", "run", "report", "info", "predict", "validate"}}
check(not unknown, f"all documented sub-commands are implemented {unknown or ''}")

print("Examples referenced in docs exist")
for rel in ["examples/basic_classification/run.py", "examples/regression/run.py", "examples/spatial_explainability/run.py",
            "examples/model_comparison/run.py", "examples/example_config.json"]:
    check((ROOT / rel).exists(), rel)

print()
if problems:
    print(f"AUDIT FAILED with {len(problems)} problem(s)")
    sys.exit(1)
print("AUDIT PASSED")
