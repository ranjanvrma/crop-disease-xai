"""Block AI-assistant names and attribution from the repository.

    python scripts/check_names.py --staged          # pre-commit hook: staged changes + file names
    python scripts/check_names.py --message FILE    # commit-msg hook: the commit message
    python scripts/check_names.py --all             # CI: every tracked file + every commit's metadata

Exits 1 and lists every hit if anything matches. The patterns are written so
that this file does not match itself.
"""

import argparse
import re
import subprocess
import sys

# Character classes keep the literal words out of this file.
BLOCKED = re.compile(
    r"c[l]aude|a[n]thropic|co-authored-by:\s*c[l]aude|generated with \[?c[l]aude",
    re.IGNORECASE,
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def hits_in(text: str, where: str) -> list[str]:
    return [f"{where}:{n}: {line.strip()[:120]}" for n, line in enumerate(text.splitlines(), 1) if BLOCKED.search(line)]


def check_staged() -> list[str]:
    hits = [f"file name: {p}" for p in git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
            if BLOCKED.search(p)]
    added = [line[1:] for line in git("diff", "--cached", "-U0", "--no-color").splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    return hits + hits_in("\n".join(added), "staged change")


def check_message(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [line for line in f if not line.startswith("#")]  # git's own comment lines
    return hits_in("".join(lines), "commit message")


def check_all() -> list[str]:
    hits = []
    for path in git("ls-files").splitlines():
        if BLOCKED.search(path):
            hits.append(f"file name: {path}")
        try:
            data = open(path, "rb").read()
        except OSError:
            continue
        if b"\0" not in data[:8192]:  # skip binary files
            hits += hits_in(data.decode("utf-8", errors="replace"), path)
    log = git("log", "--all", "--format=%H%x1f%an <%ae>%x1f%cn <%ce>%x1f%B%x1e")
    for record in filter(None, (r.strip() for r in log.split("\x1e"))):
        sha, author, committer, body = (record.split("\x1f") + ["", "", ""])[:4]
        for label, text in (("author", author), ("committer", committer), ("message", body)):
            if BLOCKED.search(text):
                hits.append(f"commit {sha[:8]} {label}: {text.strip().splitlines()[0][:120]}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--message", metavar="FILE")
    mode.add_argument("--all", action="store_true")
    args = ap.parse_args()

    hits = check_staged() if args.staged else check_message(args.message) if args.message else check_all()
    if hits:
        print("Blocked: AI-assistant names/attribution are not allowed in this repository.", file=sys.stderr)
        print("\n".join(f"  {h}" for h in hits), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
