#!/usr/bin/env python3
"""Stage all changes, commit, and push /py to GitHub via dulwich.

Usage:
    python3 push.py "commit message"
"""
import sys
from pathlib import Path

from dulwich import porcelain
from dulwich.repo import Repo

REPO_DIR = Path(__file__).resolve().parent
TOKEN_FILE = REPO_DIR / ".git_push_token"
REMOTE = "github.com/henryliang3027/py.git"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python3 push.py \"commit message\"")
    message = sys.argv[1]

    token = TOKEN_FILE.read_text().strip()

    repo = Repo(str(REPO_DIR))
    porcelain.add(repo, paths=[str(REPO_DIR)])

    status = porcelain.status(repo)
    if any(status.staged.values()):
        porcelain.commit(repo, message=message.encode())
        print(f"committed: {message}")
    else:
        print("nothing to commit")

    porcelain.push(
        repo,
        remote_location=f"https://{token}@{REMOTE}",
        refspecs=[b"refs/heads/master:refs/heads/main"],
    )
    print("pushed")


if __name__ == "__main__":
    main()
