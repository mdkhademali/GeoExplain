"""Create GeoExplain_v<version>.zip from the repository, excluding caches and build artifacts."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "outputs", "dist", "build",
                ".ipynb_checkpoints", "htmlcov"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".joblib"}
EXCLUDE_NAMES = {".coverage", ".DS_Store"}


def main(dest: Path) -> None:
    ns: dict = {}
    exec((ROOT / "src/geoxplain/_version.py").read_text(), ns)  # noqa: S102 - trusted local file
    name = f"GeoExplain_v{ns['__version__']}.zip"
    out = dest / name
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if not path.is_file() or set(rel.parts) & EXCLUDE_DIRS or path.suffix in EXCLUDE_SUFFIX:
                continue
            if path.name in EXCLUDE_NAMES or any(p.endswith(".egg-info") for p in rel.parts) or path.name.startswith("GeoExplain_v"):
                continue
            zf.write(path, Path("GeoExplain") / rel)
            count += 1
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    for pkg in sorted(d for d in (ROOT / "src" / "geoxplain").iterdir() if d.is_dir() and d.name != "__pycache__"):
        if not any(n.startswith(f"GeoExplain/src/geoxplain/{pkg.name}/") for n in names):
            out.unlink()
            raise SystemExit(f"ZIP is missing package directory {pkg.name}; aborting")
    print(f"{out}  ({count} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent
    dest.mkdir(parents=True, exist_ok=True)
    main(dest)
