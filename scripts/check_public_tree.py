"""Check the publishable tree without printing any detected secret values."""
import argparse
import fnmatch
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "absolute host path": re.compile(r"(?<![\w:])/(?:mnt|home|Users|root|workspace|private|data)/[^\s\"'`]+"),
    "Windows user path": re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access token": re.compile(r"\b(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{32,})\b"),
    "hostname metadata": re.compile(r"[\"']hostname[\"']\s*[:=]|\bhostname=\""),
    "credential assignment": re.compile(r"(?i)\b(?:api_key|password|secret|access_token)\s*[=:]\s*[\"'][^\"'\s]{12,}[\"']"),
    "personal email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}
PRIVATE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
BINARY_SUFFIXES = {".pyc", ".pyo", ".safetensors", ".pt", ".pth", ".bin", ".pem", ".key"}


def scan(root):
    """Returns relative filenames and issue categories, never matching values."""
    ignore = [line.strip() for line in (root / ".gitignore").read_text().splitlines()
              if line.strip() and not line.startswith("#")]

    def ignored(path):
        relative = path.relative_to(root).as_posix()
        matched = False
        for rule in ignore:
            negate = rule.startswith("!")
            rule = rule.lstrip("!")
            directory = rule.endswith("/")
            rooted = rule.startswith("/")
            rule = rule.strip("/")
            parts = relative.split("/")
            if rooted:
                hit = relative == rule or (directory and relative.startswith(rule + "/"))
            else:
                hit = any(fnmatch.fnmatch(part, rule) for part in (parts if directory else [parts[-1]]))
            if hit:
                matched = not negate
        return matched

    issues, count = [], 0
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts or ignored(path):
            continue
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            issues.append((name, "symbolic link"))
            continue
        if not path.is_file():
            continue
        count += 1
        if path.suffix in BINARY_SUFFIXES:
            issues.append((name, "binary cache, weight or key"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append((name, "non-text file"))
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                issues.append((name, label))
        if any(ip not in ("127.0.0.1", "0.0.0.0") for ip in PRIVATE_IP.findall(text)):
            issues.append((name, "non-loopback IP address"))
    return count, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    count, issues = scan(args.root.resolve())
    for name, issue in issues:
        print(f"{name}: {issue}")
    print(f"Checked {count} publishable files; {len(issues)} findings.")
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
