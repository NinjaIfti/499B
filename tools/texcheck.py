"""Find LaTeX row separators that lost a backslash.

A tabular row must end with a double backslash. A single trailing backslash is
a control-space instead, which silently merges rows and produces a cascade of
"Misplaced \\noalign" and "Extra alignment tab" errors far from the real cause.

This class of damage is easy to introduce with shell heredocs, which halve
backslashes, so it is worth checking for explicitly.

Run:  py -3 tools/texcheck.py Book
"""
import os
import sys

BS = chr(92)
NL = chr(10)


def scan(path):
    """Yield (lineno, text) for lines ending in exactly one backslash."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for num, raw in enumerate(fh, 1):
            line = raw.rstrip(NL).rstrip()
            if not line.endswith(BS):
                continue
            # count the run of trailing backslashes
            run = 0
            for ch in reversed(line):
                if ch == BS:
                    run += 1
                else:
                    break
            if run == 1:
                yield num, line


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    names = sorted(f for f in os.listdir(root) if f.endswith(".tex"))
    total = 0
    for name in names:
        hits = list(scan(os.path.join(root, name)))
        if not hits:
            continue
        print("%s: %d suspicious line(s)" % (name, len(hits)))
        for num, line in hits:
            shown = line if len(line) <= 88 else line[:85] + "..."
            print("   %5d | %s" % (num, shown))
        total += len(hits)
    print()
    print("total single-trailing-backslash lines: %d" % total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
