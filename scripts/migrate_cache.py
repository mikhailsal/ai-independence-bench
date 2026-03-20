#!/usr/bin/env python3
"""Migrate cache directories to config-based names.

Renames ``{model_slug}`` → ``{model_slug}@{reasoning}-t{temp}``,
moves bare experiment dirs into ``run_1/``, and splits the Step Flash
directory into three separate config directories.

Usage:
    python scripts/migrate_cache.py --dry-run   # preview only
    python scripts/migrate_cache.py             # execute migration
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CACHE_DIR, CONFIGS_PATH, model_id_to_slug, slug_to_model_id

EXPERIMENT_DIRS = {"identity", "resistance", "stability"}

STEP_FLASH_SLUG = "stepfun--step-3.5-flash:free"
STEP_FLASH_SPLIT = [
    # (new_config_suffix, source_runs — "bare" means legacy experiment dirs at top level)
    ("@low-t0.7", ["bare", 2, 3, 4, 5, 6]),
    ("@low-t0.0", [7, 8, 9, 10, 11, 12]),
    ("@low-t1.0", [13, 14, 15, 16, 17, 18]),
]


def _build_migration_map() -> dict[str, str]:
    """Build old_dir_name → new_dir_name mapping from YAML config.

    Returns a dict keyed by old slug (e.g. ``openai--gpt-5.4-mini``)
    to new config dir name (e.g. ``openai--gpt-5.4-mini@low-t1.0``).
    """
    import yaml

    if not CONFIGS_PATH.exists():
        print(f"ERROR: {CONFIGS_PATH} not found. Create it first.", file=sys.stderr)
        sys.exit(1)

    data = yaml.safe_load(CONFIGS_PATH.read_text(encoding="utf-8"))
    if not data or "models" not in data:
        print("ERROR: models.yaml is empty or malformed.", file=sys.stderr)
        sys.exit(1)

    seen_model_ids: dict[str, list[dict]] = {}
    for entry in data["models"]:
        mid = entry["model_id"]
        seen_model_ids.setdefault(mid, []).append(entry)

    mapping: dict[str, str] = {}
    for mid, entries in seen_model_ids.items():
        slug = model_id_to_slug(mid)
        if slug == STEP_FLASH_SLUG:
            continue
        if len(entries) > 1:
            continue
        entry = entries[0]
        temp = float(entry["temperature"])
        reasoning = entry["reasoning_effort"]
        new_name = f"{slug}@{reasoning}-t{temp}"
        mapping[slug] = new_name

    return mapping


def _has_bare_experiment_dirs(model_dir: Path) -> bool:
    """Check if a model directory has bare experiment dirs (not inside run_N/)."""
    return any(
        (model_dir / exp).is_dir()
        for exp in EXPERIMENT_DIRS
    )


def _normalize_run_1(model_dir: Path, *, dry_run: bool) -> None:
    """Move bare experiment dirs into ``run_1/`` subdirectory."""
    if not _has_bare_experiment_dirs(model_dir):
        return

    run_1_dir = model_dir / "run_1"
    if dry_run:
        print(f"  NORMALIZE: move bare experiment dirs → run_1/")
        return

    run_1_dir.mkdir(exist_ok=True)
    for exp_name in EXPERIMENT_DIRS:
        src = model_dir / exp_name
        if src.is_dir():
            dst = run_1_dir / exp_name
            shutil.move(str(src), str(dst))


def _migrate_standard(mapping: dict[str, str], *, dry_run: bool) -> int:
    """Rename standard model directories (not Step Flash).

    Returns number of directories migrated.
    """
    count = 0
    for old_slug, new_name in sorted(mapping.items()):
        old_dir = CACHE_DIR / old_slug
        if not old_dir.exists():
            continue

        new_dir = CACHE_DIR / new_name
        if new_dir.exists():
            print(f"  SKIP: {new_name} already exists")
            continue

        print(f"  {old_slug} → {new_name}")

        if not dry_run:
            _normalize_run_1(old_dir, dry_run=False)
            old_dir.rename(new_dir)
        else:
            if _has_bare_experiment_dirs(old_dir):
                print(f"    NORMALIZE: move bare experiment dirs → run_1/")

        count += 1
    return count


def _migrate_step_flash(*, dry_run: bool) -> int:
    """Split Step Flash into 3 separate config directories.

    Returns number of new directories created.
    """
    src_dir = CACHE_DIR / STEP_FLASH_SLUG
    if not src_dir.exists():
        print(f"  SKIP: {STEP_FLASH_SLUG} not found")
        return 0

    count = 0
    for suffix, source_runs in STEP_FLASH_SPLIT:
        new_name = f"{STEP_FLASH_SLUG}{suffix}"
        new_dir = CACHE_DIR / new_name
        if new_dir.exists():
            print(f"  SKIP: {new_name} already exists")
            continue

        print(f"  {STEP_FLASH_SLUG} → {new_name} (runs: {source_runs})")

        if not dry_run:
            new_dir.mkdir(parents=True, exist_ok=True)

        for dest_run_num, src_run in enumerate(source_runs, start=1):
            if src_run == "bare":
                src_path = src_dir
                items_to_copy = [
                    src_path / exp for exp in EXPERIMENT_DIRS if (src_path / exp).is_dir()
                ]
            else:
                src_path = src_dir / f"run_{src_run}"
                items_to_copy = [
                    src_path / exp for exp in EXPERIMENT_DIRS if (src_path / exp).is_dir()
                ]

            dest_run_dir = new_dir / f"run_{dest_run_num}"
            print(f"    src {'bare' if src_run == 'bare' else f'run_{src_run}'} → run_{dest_run_num}")

            if not dry_run:
                dest_run_dir.mkdir(parents=True, exist_ok=True)
                for item in items_to_copy:
                    shutil.copytree(str(item), str(dest_run_dir / item.name))

        count += 1

    if not dry_run and count > 0:
        shutil.rmtree(str(src_dir))
        print(f"  REMOVED original: {STEP_FLASH_SLUG}")

    return count


def _validate(*, verbose: bool = True) -> bool:
    """Validate that all cache dirs use the new naming scheme."""
    ok = True
    if not CACHE_DIR.exists():
        return True

    for d in sorted(CACHE_DIR.iterdir()):
        if not d.is_dir():
            continue
        if "@" not in d.name:
            if verbose:
                print(f"  INVALID: {d.name} (missing @ config suffix)")
            ok = False
            continue

        has_run_dir = any(
            sub.is_dir() and sub.name.startswith("run_")
            for sub in d.iterdir()
        )
        has_bare_exp = any(
            (d / exp).is_dir() for exp in EXPERIMENT_DIRS
        )
        if has_bare_exp:
            if verbose:
                print(f"  INVALID: {d.name} has bare experiment dirs (not in run_N/)")
            ok = False
        if not has_run_dir:
            if verbose:
                print(f"  WARNING: {d.name} has no run_N/ directories")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate cache to config-based directory names.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files.")
    args = parser.parse_args()

    dry_run = args.dry_run
    label = "[DRY RUN] " if dry_run else ""

    print(f"\n{label}Cache Migration: config-based directory names")
    print(f"  Cache dir: {CACHE_DIR}")
    print()

    mapping = _build_migration_map()
    print(f"Phase 1: Migrate standard directories ({len(mapping)} models)")
    std_count = _migrate_standard(mapping, dry_run=dry_run)
    print(f"  → {std_count} directories {'would be' if dry_run else ''} migrated\n")

    print("Phase 2: Split Step Flash into 3 configs")
    sf_count = _migrate_step_flash(dry_run=dry_run)
    print(f"  → {sf_count} directories {'would be' if dry_run else ''} created\n")

    print("Phase 3: Validation")
    if _validate():
        print("  All directories valid ✓")
    else:
        print("  Some directories have issues (see above)")

    print(f"\n{label}Migration complete.")


if __name__ == "__main__":
    main()
