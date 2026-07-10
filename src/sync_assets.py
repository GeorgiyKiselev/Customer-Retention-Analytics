"""Copy presentation-ready analysis artifacts into the GitHub-rendered assets folder.

The analysis pipeline writes raw outputs to ``outputs/``. This helper mirrors
selected visual artifacts into ``assets/`` so README image embeds render
reliably on GitHub without pointing at operational output folders.
"""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"

PLOT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}
SHAP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".html"}
TABLE_SNIPPET_EXTENSIONS = {".md"}


def _copy_matching_files(source_dir: Path, target_dir: Path, extensions: set[str]) -> list[Path]:
    """Copy files with allowed extensions from source to target and return targets."""

    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    if not source_dir.exists():
        return copied

    for source_path in sorted(source_dir.iterdir(), key=lambda path: path.name.lower()):
        if not source_path.is_file():
            continue
        if source_path.suffix.lower() not in extensions:
            continue
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied.append(target_path)
    return copied


def sync_assets() -> dict[str, list[Path]]:
    """Create assets folders and copy presentation-ready output files."""

    copied = {
        "plots": _copy_matching_files(
            OUTPUTS_DIR / "plots",
            ASSETS_DIR / "plots",
            PLOT_EXTENSIONS,
        ),
        "shap": _copy_matching_files(
            OUTPUTS_DIR / "shap",
            ASSETS_DIR / "shap",
            SHAP_EXTENSIONS,
        ),
        "tables": _copy_matching_files(
            OUTPUTS_DIR / "tables",
            ASSETS_DIR / "tables",
            TABLE_SNIPPET_EXTENSIONS,
        ),
    }

    # Ensure the expected folder structure exists even when no files are copied.
    for subdir in ("plots", "shap", "tables"):
        (ASSETS_DIR / subdir).mkdir(parents=True, exist_ok=True)

    return copied


def main() -> None:
    """CLI entry point."""

    copied = sync_assets()
    total = sum(len(paths) for paths in copied.values())
    print(f"Copied {total} presentation asset(s).")
    for group, paths in copied.items():
        print(f"  {group}: {len(paths)} file(s)")
        for path in paths:
            print(f"    - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
