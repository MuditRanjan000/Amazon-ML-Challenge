"""Keep agent instruction files byte-identical across AI tools.

Groups (first name = canonical, but the most recently edited file wins so an
edit made to any mirror is never lost):
  AGENTS.md  <-> CLAUDE.md <-> GEMINI.md          (shared, committed)
  CLAUDE.local.md <-> GEMINI.local.md             (personal, gitignored)

Overwritten versions are backed up to .tmp/agent_docs_backup/ first.

Usage:
  python execution/sync_agent_docs.py            # sync
  python execution/sync_agent_docs.py --check    # exit 1 if out of sync, write nothing
  python execution/sync_agent_docs.py --selftest # run built-in checks in a temp dir
"""
import shutil
import sys
import time
from pathlib import Path

GROUPS = [
    ["AGENTS.md", "CLAUDE.md", "GEMINI.md"],
    ["CLAUDE.local.md", "GEMINI.local.md"],
]


def sync(root: Path, check: bool = False) -> list[str]:
    """Return a list of drift/changes; in check mode nothing is written."""
    changes = []
    for group in GROUPS:
        paths = [root / name for name in group]
        existing = [p for p in paths if p.exists()]
        if not existing:
            continue  # e.g. teammate without a personal layer
        source = max(existing, key=lambda p: p.stat().st_mtime)
        content = source.read_bytes()
        for p in paths:
            if p.exists() and p.read_bytes() == content:
                continue
            changes.append(f"{source.name} -> {p.name}")
            if check:
                continue
            if p.exists():
                backup_dir = root / ".tmp" / "agent_docs_backup"
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, backup_dir / f"{p.name}.{time.strftime('%Y%m%d-%H%M%S')}")
            p.write_bytes(content)
    return changes


def _selftest():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "AGENTS.md").write_text("v1")
        assert sync(root, check=True) == ["AGENTS.md -> CLAUDE.md", "AGENTS.md -> GEMINI.md"]
        assert not (root / "CLAUDE.md").exists(), "check mode must not write"
        sync(root)
        assert (root / "CLAUDE.md").read_text() == (root / "GEMINI.md").read_text() == "v1"
        assert sync(root, check=True) == []

        # Edit a mirror (newest) -> it propagates, and the old canonical is backed up.
        (root / "CLAUDE.md").write_text("v2")
        future = time.time() + 10
        os.utime(root / "CLAUDE.md", (future, future))
        sync(root)
        assert (root / "AGENTS.md").read_text() == (root / "GEMINI.md").read_text() == "v2"
        backups = list((root / ".tmp" / "agent_docs_backup").iterdir())
        assert any(b.read_text() == "v1" for b in backups), "overwritten version must be backed up"

        # Personal group absent -> skipped; present -> mirrored.
        (root / "CLAUDE.local.md").write_text("me")
        sync(root)
        assert (root / "GEMINI.local.md").read_text() == "me"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    repo = Path(__file__).resolve().parent.parent
    check = "--check" in sys.argv
    result = sync(repo, check=check)
    for line in result:
        print(("DRIFT " if check else "synced ") + line)
    sys.exit(1 if (check and result) else 0)
