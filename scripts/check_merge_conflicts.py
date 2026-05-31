#!/usr/bin/env python3
import sys
import subprocess
import os
import re

# Standard git merge conflict markers
CONFLICT_MARKERS = [
    re.compile(r'^<<<<<<<(\s|$)'),
    re.compile(r'^=======(\s|$)'),
    re.compile(r'^>>>>>>>(\s|$)'),
    re.compile(r'^\|\|\|\|\|\|\|(\s|$)')
]

def get_files_to_check():
    """Get list of files tracked by git in the repository."""
    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        files = result.stdout.strip().split('\n')
        # Filter out binary or empty filenames
        return [f for f in files if f and os.path.isfile(f)]
    except (subprocess.SubprocessError, FileNotFoundError):
        # Fallback to manual directory traversal if git is not available
        exclude_dirs = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'venv', '.venv'}
        exclude_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.zip', '.h5', '.pyc'}
        files = []
        for root, dirs, filenames in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext.lower() not in exclude_exts:
                    files.append(os.path.join(root, filename))
        return files

def main():
    files = get_files_to_check()
    conflicts_found = False

    for file_path in files:
        # Skip binary files if any slipped through
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    for marker in CONFLICT_MARKERS:
                        if marker.match(line):
                            print(f"Error: Merge conflict marker found in {file_path} on line {line_num}:")
                            print(f"  {line.strip()}")
                            conflicts_found = True
        except Exception as e:
            # Skip files that can't be read as text
            continue

    if conflicts_found:
        print("\nValidation failed: Unresolved merge conflict markers detected.")
        sys.exit(1)
    else:
        print("Success: No merge conflict markers found.")
        sys.exit(0)

if __name__ == '__main__':
    main()
