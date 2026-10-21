"""Integrity checker for the standalone hardening-submodule export.

The supplied ZIP does not contain the parent repository's dataset, taxonomy, models,
checkpoints, or upstream pipeline. This checker therefore intentionally makes no claim
about those absent artifacts. It preserves the files actually present and used by this
standalone submodule: lexical/scope/policy config, schemas, the copied scope-report record,
and manually reviewed source-evidence records.

Commands:
    python -m security_hardening.integrity --check
    python -m security_hardening.integrity --write  # refresh after an intentional config/schema/evidence change

The dynamic control catalogue and live source snapshots are not frozen by this manifest:
those are versioned through the catalogue/synchronisation workflow instead.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "integrity_manifest.json"
FROZEN_FILE_GLOBS = (
    "requirements.txt",
    "security_hardening/config/*.yaml",
    "security_hardening/schemas/*.json",
    "security_hardening/data/dataset_scope_report.json",
    "security_hardening/snapshots/section_evidence/*.json",
)
FROZEN_TREES: tuple = ()
OPTIONAL_FILES: tuple = ()
MUTABLE_RUNTIME_STATE = (
    "security_hardening/data/source_sync_state.json",
    "security_hardening/data/source_sync_report_*.json",
    "security_hardening/catalogue/candidates/*",
    "security_hardening/catalogue/backups/*",
    "security_hardening/snapshots/live/*",
)
_IGNORED_NAMES = {".DS_Store", "Thumbs.db"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_noise(path: Path, root: Optional[Path] = None) -> bool:
    if path.name in _IGNORED_NAMES or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        return True
    if root is not None:
        rel = _rel(path, root)
        return any(fnmatch.fnmatch(rel, pat) for pat in MUTABLE_RUNTIME_STATE)
    return False


def frozen_files(root: Path) -> List[Path]:
    seen: Dict[str, Path] = {}
    for pattern in FROZEN_FILE_GLOBS:
        for path in root.glob(pattern):
            if path.is_file() and not _is_noise(path, root):
                seen[_rel(path, root)] = path
    return [seen[k] for k in sorted(seen)]


def tree_digest(root: Path, tree: str) -> Optional[Dict[str, object]]:
    base = root / tree
    if not base.is_dir():
        return None
    lines = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and not _is_noise(path, root):
            lines.append(f"{_rel(path, base)}\t{sha256_file(path)}\n")
    return {"path": tree, "files": len(lines), "aggregate_sha256": hashlib.sha256("".join(lines).encode()).hexdigest()}


def build_manifest(root: Path = REPO_ROOT) -> Dict[str, object]:
    files = [{"path": _rel(path, root), "sha256": sha256_file(path), "bytes": path.stat().st_size,
              **({"optional": True} if _rel(path, root) in OPTIONAL_FILES else {})}
             for path in frozen_files(root)]
    trees = [item for item in (tree_digest(root, tree) for tree in FROZEN_TREES) if item]
    return {
        "manifest_version": 2,
        "purpose": "Preserve the standalone hardening submodule config, schemas, copied scope-report artifact, and dated local source-section review records. This manifest does not validate parent-pipeline datasets, models, checkpoints, or taxonomy, which were absent from the supplied ZIP.",
        "file_globs": list(FROZEN_FILE_GLOBS),
        "excluded_mutable_runtime_state": list(MUTABLE_RUNTIME_STATE),
        "files": files,
        "trees": trees,
    }


def verify_manifest(root: Path = REPO_ROOT, manifest: Optional[Dict[str, object]] = None) -> List[str]:
    """Return human-readable problems; an empty list means frozen included files are unchanged."""
    if manifest is None:
        manifest_path = MANIFEST_PATH if root.resolve() == REPO_ROOT.resolve() else root / "security_hardening" / "data" / "integrity_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: List[str] = []
    expected = {entry["path"]: entry for entry in manifest["files"]}  # type: ignore[index]
    for relpath, entry in expected.items():
        path = root / relpath
        if not path.is_file():
            if not entry.get("optional"):
                problems.append(f"MISSING  {relpath}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"MODIFIED {relpath}")
    for path in frozen_files(root):
        relpath = _rel(path, root)
        if relpath not in expected:
            problems.append(f"ADDED    {relpath}")
    for entry in manifest.get("trees", []):  # type: ignore[union-attr]
        current = tree_digest(root, entry["path"])
        if current is None:
            continue
        if current["aggregate_sha256"] != entry["aggregate_sha256"] or current["files"] != entry["files"]:
            problems.append(f"CHANGED  tree {entry['path']} ({entry['files']} files recorded, {current['files']} found)")
    return problems


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.repo_root)
    if args.write:
        manifest = build_manifest(root)
        target = (root / "security_hardening" / "data" / "integrity_manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {target}: {len(manifest['files'])} frozen files")
        return 0
    problems = verify_manifest(root)
    for problem in problems:
        print(problem)
    print("included hardening assets unchanged" if not problems else f"{len(problems)} difference(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
