#!/usr/bin/env python3
"""
/workspace/scripts/migrate_spec_version.py

LughaGen Training Framework v1.5 §18
Manages /workspace/registry/spec_versions.yaml as the append-only record of
every spec version used during the study.

Operations:
    register <spec.md>     — register a new spec version (sets prior to superseded)
    current                — print the currently-valid spec version
    check <spec.md>        — verify a spec file's SHA-256 matches the current entry
                              (used by preflight.sh)
    list                   — list all spec versions

The §18 lock procedure GUARANTEES that registry runs' spec_hash field is never
modified after the fact. This script never touches runs.csv.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

import yaml


REGISTRY_DIR = Path("/workspace/registry")
SPEC_VERSIONS_YAML = REGISTRY_DIR / "spec_versions.yaml"


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_spec_version(spec_path: Path) -> str:
    """
    Extract the spec version number from the document title line. The spec's
    first H1 is 'LughaGen Training Framework — Implementation Specification vX.Y'.
    """
    with spec_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Implementation Specification v(\d+\.\d+(?:\.\d+)?)", line)
            if m:
                return m.group(1)
    raise ValueError(f"Could not find spec version in {spec_path}")


def load_spec_versions() -> dict:
    """Load spec_versions.yaml; return {'spec_versions': []} if missing."""
    if not SPEC_VERSIONS_YAML.exists():
        return {"spec_versions": []}
    with SPEC_VERSIONS_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "spec_versions" not in data:
        data["spec_versions"] = []
    return data


def save_spec_versions(data: dict) -> None:
    """Write spec_versions.yaml."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with SPEC_VERSIONS_YAML.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def current_version() -> Optional[dict]:
    """Return the entry with valid_until: null, or None."""
    data = load_spec_versions()
    for entry in data["spec_versions"]:
        if entry.get("valid_until") is None:
            return entry
    return None


def register_new_version(spec_path: Path) -> int:
    """
    Register a new spec version. Marks the previous current entry as
    superseded.
    """
    spec_path = spec_path.resolve()
    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}", file=sys.stderr)
        return 1

    new_version = extract_spec_version(spec_path)
    new_hash = sha256_of_file(spec_path)

    data = load_spec_versions()

    # Check for duplicate (idempotent)
    for entry in data["spec_versions"]:
        if entry["version"] == new_version and entry["sha256"] == new_hash:
            if entry.get("valid_until") is None:
                print(f"Spec v{new_version} (hash {new_hash[:12]}) is already current.")
                return 0
            else:
                print(f"Spec v{new_version} was previously registered and superseded. "
                      f"Refusing to re-register a superseded spec.", file=sys.stderr)
                return 1

    # Find prior current entry (if any) and supersede it
    now = utc_now_iso()
    for entry in data["spec_versions"]:
        if entry.get("valid_until") is None:
            entry["valid_until"] = now
            entry["superseded_by"] = new_version

    # Append the new entry
    new_entry = {
        "version": new_version,
        "sha256": new_hash,
        "file": spec_path.name,
        "valid_from": now,
        "valid_until": None,
        "superseded_by": None,
    }
    data["spec_versions"].append(new_entry)

    save_spec_versions(data)
    print(f"Registered spec v{new_version} (hash {new_hash[:12]}) as current.")
    return 0


def check_current(spec_path: Path) -> int:
    """
    Verify a spec file's SHA-256 matches the current entry in spec_versions.yaml.
    Used by preflight.sh — exit 0 if match, 1 if mismatch.
    """
    spec_path = spec_path.resolve()
    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}", file=sys.stderr)
        return 1

    actual_hash = sha256_of_file(spec_path)
    current = current_version()
    if current is None:
        print("No current spec version registered. "
              "Run `migrate_spec_version.py register <spec.md>` first.",
              file=sys.stderr)
        return 1

    if current["sha256"] != actual_hash:
        print(f"FAIL: spec hash mismatch.\n"
              f"  registry current: v{current['version']} hash {current['sha256'][:12]}\n"
              f"  file on disk:     hash {actual_hash[:12]}",
              file=sys.stderr)
        return 1

    print(f"PASS: spec v{current['version']} hash matches (hash {actual_hash[:12]}).")
    return 0


def list_versions() -> int:
    data = load_spec_versions()
    if not data["spec_versions"]:
        print("No spec versions registered yet.")
        return 0
    print(f"{'Version':<8} {'Hash':<14} {'valid_from':<22} {'valid_until':<22} {'Superseded by'}")
    print("-" * 90)
    for e in data["spec_versions"]:
        until = e.get("valid_until") or "CURRENT"
        by = e.get("superseded_by") or "-"
        print(f"{e['version']:<8} {e['sha256'][:12]:<14} "
              f"{e['valid_from']:<22} {until:<22} {by}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LughaGen spec version lock manager (§18)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="Register a new spec version")
    p_reg.add_argument("spec", type=Path, help="Path to spec markdown file")

    sub.add_parser("current", help="Print the current spec version")

    p_chk = sub.add_parser("check", help="Verify spec hash matches current registry entry")
    p_chk.add_argument("spec", type=Path, help="Path to spec markdown file")

    sub.add_parser("list", help="List all spec versions")

    args = parser.parse_args()

    if args.cmd == "register":
        return register_new_version(args.spec)
    elif args.cmd == "current":
        cur = current_version()
        if cur is None:
            print("No current spec version.")
            return 1
        print(f"v{cur['version']} (hash {cur['sha256'][:12]}, from {cur['valid_from']})")
        return 0
    elif args.cmd == "check":
        return check_current(args.spec)
    elif args.cmd == "list":
        return list_versions()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
