"""Repository checkout and top-ranked-file cache for localization runs."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from get_repo_structure.get_repo_structure import parse_python_file


def _run_git(repo_path, *args):
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _slug(repo):
    return repo.replace("/", "__")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_issue_cache(issue, ranked_files, cache_dir, force=False):
    """Prepare/reuse a detached worktree and persist the exact ranked files."""
    issue_id = issue["instance_id"]
    repo = issue["repo"]
    commit = issue["base_commit"]
    ranked_files = list(ranked_files[:3])
    cache_root = Path(cache_dir).resolve()
    issue_dir = cache_root / issue_id
    metadata_path = issue_dir / "metadata.json"

    if not force and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            validate_issue_cache(metadata, issue, ranked_files)
            metadata["cache_hit"] = True
            metadata["repository_prepare_seconds"] = 0.0
            metadata["file_cache_seconds"] = 0.0
            return metadata
        except (OSError, ValueError, KeyError, AssertionError):
            pass

    repositories = cache_root / "repositories"
    checkouts = cache_root / "checkouts"
    repositories.mkdir(parents=True, exist_ok=True)
    checkouts.mkdir(parents=True, exist_ok=True)
    bare_repo = repositories / f"{_slug(repo)}.git"
    clone_started = time.monotonic()
    if not bare_repo.exists():
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(bare_repo)],
            check=True,
        )
    checkout_path = checkouts / _slug(repo) / "current" / repo.rsplit("/", 1)[-1]
    checkout_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkout_path.exists():
        subprocess.run(
            ["git", "--git-dir", str(bare_repo), "worktree", "add", "--detach", str(checkout_path), commit],
            check=True,
        )
    else:
        if _run_git(checkout_path, "status", "--porcelain"):
            raise ValueError(f"Shared repository checkout has uncommitted changes: {checkout_path}")
        _run_git(checkout_path, "checkout", "--detach", commit)
    checked_out_commit = _run_git(checkout_path, "rev-parse", "HEAD")
    if checked_out_commit != commit:
        raise ValueError(f"Commit mismatch for {issue_id}: expected {commit}, got {checked_out_commit}")
    repository_prepare_seconds = time.monotonic() - clone_started

    file_started = time.monotonic()
    missing_files = []
    file_metadata = []
    for rank, relative_path in enumerate(ranked_files, start=1):
        source = checkout_path / relative_path
        if not source.is_file():
            missing_files.append({"path": relative_path, "rank": rank, "reason": "missing"})
            continue
        try:
            content = source.read_bytes()
            # Agentless parses repository files as text. Decode strictly so an unreadable
            # source is recorded rather than silently changed.
            text = content.decode("utf-8")
            target = issue_dir / "files" / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            file_metadata.append({"path": relative_path, "rank": rank, "sha256": hashlib.sha256(content).hexdigest()})
        except (OSError, UnicodeError) as exc:
            missing_files.append({"path": relative_path, "rank": rank, "reason": f"unreadable: {exc}"})

    metadata = {
        "instance_id": issue_id,
        "repo": repo,
        "base_commit": commit,
        "checked_out_commit": checked_out_commit,
        "repository_path": str(checkout_path),
        "bare_repository_path": str(bare_repo),
        "cache_directory": str(issue_dir),
        "ranked_files": [{"path": path, "rank": rank} for rank, path in enumerate(ranked_files, start=1)],
        "cached_files": file_metadata,
        "missing_or_unreadable_files": missing_files,
        "cache_status": "complete" if not missing_files and len(file_metadata) == len(ranked_files) else "incomplete",
        "cache_hit": False,
        "repository_prepare_seconds": repository_prepare_seconds,
        "file_cache_seconds": time.monotonic() - file_started,
    }
    issue_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def validate_issue_cache(metadata, issue, ranked_files=None):
    """Raise ValueError if cache identity, commit, ranked files, or bytes disagree."""
    if metadata["instance_id"] != issue["instance_id"] or metadata["repo"] != issue["repo"]:
        raise ValueError("Cache issue/repository identity mismatch")
    if metadata["base_commit"] != issue["base_commit"] or metadata["checked_out_commit"] != issue["base_commit"]:
        raise ValueError("Cache base commit mismatch")
    if ranked_files is not None and [x["path"] for x in metadata["ranked_files"]] != list(ranked_files[:3]):
        raise ValueError("Cached file ranking mismatch")
    if metadata["cache_status"] != "complete":
        raise ValueError("Cache is incomplete")
    for item in metadata["cached_files"]:
        cached = Path(metadata["cache_directory"]) / "files" / item["path"]
        if not cached.is_file() or _sha256(cached) != item["sha256"]:
            raise ValueError(f"Cached file missing or changed: {item['path']}")
        source_path = item["path"]
        original_result = subprocess.run(
            ["git", "--git-dir", metadata["bare_repository_path"], "show", f"{issue['base_commit']}:{source_path}"],
            capture_output=True,
        )
        if original_result.returncode != 0:
            repo_root = metadata["repo"].rsplit("/", 1)[-1]
            if not source_path.startswith(repo_root + "/"):
                raise subprocess.CalledProcessError(
                    original_result.returncode, original_result.args, original_result.stdout, original_result.stderr
                )
            source_path = source_path[len(repo_root) + 1:]
            original_result = subprocess.run(
                ["git", "--git-dir", metadata["bare_repository_path"], "show", f"{issue['base_commit']}:{source_path}"],
                check=True,
                capture_output=True,
            )
        original = original_result.stdout
        if hashlib.sha256(original).hexdigest() != item["sha256"]:
            raise ValueError(f"Cached file differs from base commit: {item['path']}")
    actual_commit = subprocess.run(
        ["git", "--git-dir", metadata["bare_repository_path"], "rev-parse", f"{issue['base_commit']}^{{commit}}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if actual_commit != issue["base_commit"] or metadata["checked_out_commit"] != issue["base_commit"]:
        raise ValueError("Recorded or resolved commit does not match the issue base commit")
    return True


def get_cached_structure(metadata):
    """Build Agentless's normal structure representation from cached ranked files."""
    repo_root = metadata["repo"].rsplit("/", 1)[-1]
    structure = {repo_root: {}}
    for item in metadata["cached_files"]:
        relative_path = item["path"]
        parts = relative_path.split("/")
        if parts[0] == repo_root:
            parts = parts[1:]
            parent = structure[repo_root]
        else:
            # RRF paths are repository-relative and are not consistent about
            # including the repository name. Keep the flattened Agentless path
            # identical to the ranked path in either case.
            parent = structure
        for directory in parts[:-1]:
            parent = parent.setdefault(directory, {})
        filename = parts[-1]
        cached_path = Path(metadata["cache_directory"]) / "files" / relative_path
        if filename.endswith(".py"):
            classes, functions, lines = parse_python_file(str(cached_path))
            parent[filename] = {"classes": classes, "functions": functions, "text": lines}
        else:
            parent[filename] = {}
    return structure


def load_issue_cache(cache_dir, issue):
    metadata_path = Path(cache_dir) / issue["instance_id"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    validate_issue_cache(metadata, issue)
    return metadata
