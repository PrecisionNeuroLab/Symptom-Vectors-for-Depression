"""
Rebuild tmlr-submission.zip (repo root, gitignored): the TMLR Beyond-PDF online-editor import.
Top level of the ZIP: compile_submission.py, submission_folder/, tmlr_do_not_modify/ from
tmlr-beyond-pdf-author-kit/, with .DS_Store and other OS clutter skipped.

    python make_tmlr_zip.py           build if the submission changed since the last ZIP
    python make_tmlr_zip.py --watch   keep rebuilding on every change (Ctrl+C to stop)

The ZIP's comment holds a fingerprint of the files it was built from, so an unchanged
submission is never rewritten.
"""
import argparse
import hashlib
import os
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(ROOT, 'tmlr-beyond-pdf-author-kit')
ZIP = os.path.join(ROOT, 'tmlr-submission.zip')
PARTS = ['compile_submission.py', 'submission_folder', 'tmlr_do_not_modify']
SKIP = {'.DS_Store', 'Thumbs.db', 'desktop.ini', '__pycache__', '.ipynb_checkpoints'}


def submission_files():
    """Sorted (name in the ZIP, path on disk) for every file that goes in."""
    out = []
    for part in PARTS:
        path = os.path.join(KIT, part)
        if os.path.isfile(path):
            out.append((part, path))
            continue
        for folder, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in SKIP]
            out += [(os.path.relpath(os.path.join(folder, f), KIT).replace(os.sep, '/'), os.path.join(folder, f))
                    for f in files if f not in SKIP]
    return sorted(out)


def fingerprint(files):
    """SHA-256 over every file's name and content."""
    h = hashlib.sha256()
    for name, path in files:
        h.update(name.encode())
        with open(path, 'rb') as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def built_from():
    """Fingerprint stored in the current ZIP, or None."""
    try:
        with zipfile.ZipFile(ZIP) as z:
            return z.comment.decode()
    except (OSError, zipfile.BadZipFile):
        return None


def build(quiet=False):
    """Write the ZIP if the submission changed; returns True when it was rewritten."""
    files = submission_files()
    fp = fingerprint(files)
    if fp == built_from():
        return False
    tmp = ZIP + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in files:
            z.write(path, name)
        z.comment = fp.encode()
    os.replace(tmp, ZIP)  # atomic: never a half-written ZIP
    if not quiet:
        print(f"{time.strftime('%H:%M:%S')} wrote {os.path.relpath(ZIP)} ({len(files)} files, "
              f"{os.path.getsize(ZIP) / 1e6:.1f} MB); re-import replaces the editor's project, save edits there first")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--watch', action='store_true', help='rebuild on every change until Ctrl+C')
    parser.add_argument('--quiet', action='store_true', help='print nothing')
    args = parser.parse_args()
    if not build(args.quiet) and not args.quiet and not args.watch:
        print(f"{os.path.relpath(ZIP)} is up to date")
    if args.watch:
        print(f"watching {os.path.relpath(KIT)} (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(2)
                build(args.quiet)
        except KeyboardInterrupt:
            pass
