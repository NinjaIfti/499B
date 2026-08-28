"""Report which bibliography entries the book actually cites.

IEEEtran only prints entries that are cited, so the entry count in the .bib
file is NOT the reference count that appears in the printed document.

Run from the Book/ directory:  py -3 ../tools/bibcheck.py
or from the repo root:         py -3 tools/bibcheck.py Book
"""
import os
import re
import sys

BS = chr(92)  # a literal backslash, kept out of string literals on purpose

CITE_RE = re.compile(BS + BS + r"cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ENTRY_RE = re.compile(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)


def strip_comments(text):
    """Drop % comment lines and trailing % comments (but not \\%)."""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        # cut a trailing comment that is not an escaped percent
        cut = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != BS):
                cut = i
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def main():
    book = sys.argv[1] if len(sys.argv) > 1 else "."
    tex = sorted(
        f for f in os.listdir(book) if f.endswith(".tex")
    )

    cited = {}
    for name in tex:
        path = os.path.join(book, name)
        with open(path, encoding="utf-8", errors="replace") as fh:
            body = strip_comments(fh.read())
        for match in CITE_RE.finditer(body):
            for key in match.group(1).split(","):
                key = key.strip()
                if key:
                    cited.setdefault(key, set()).add(name)

    bib_path = os.path.join(book, "sdp.bib")
    with open(bib_path, encoding="utf-8", errors="replace") as fh:
        bib = fh.read()
    defined = {}
    for kind, key in ENTRY_RE.findall(bib):
        defined[key.strip()] = kind

    print("cited distinct keys : %d" % len(cited))
    print("entries in sdp.bib  : %d" % len(defined))
    print()

    missing = sorted(k for k in cited if k not in defined)
    print("CITED BUT NOT IN BIB (these print as [?]): %d" % len(missing))
    for key in missing:
        print("   ! %-32s  in %s" % (key, ", ".join(sorted(cited[key]))))
    print()

    unused = sorted(k for k in defined if k not in cited)
    print("IN BIB BUT NEVER CITED (will NOT appear): %d" % len(unused))
    for key in unused:
        print("   - %-32s  (%s)" % (key, defined[key]))
    print()

    printed = len([k for k in cited if k in defined])
    print("=> references that will PRINT: %d" % printed)

    print()
    print("citations per file:")
    per = {}
    for key, files in cited.items():
        for f in files:
            per.setdefault(f, set()).add(key)
    for name in tex:
        if name in per:
            print("   %-24s %d" % (name, len(per[name])))

    return 0


if __name__ == "__main__":
    sys.exit(main())
