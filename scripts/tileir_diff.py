#!/usr/bin/env python3
"""Compare two Tile IR text dumps and report the first difference.

Usage:
    tileir_diff.py <left.tileir> <right.tileir>

Exit status is 0 when the files are byte-identical, 1 when they differ, and 2
when a file cannot be read. On a difference the report names the first line
that differs and shows both sides of it, so a reader does not have to open a
diff viewer to know what broke.
"""

import sys


def read_lines(path):
    with open(path, "rb") as handle:
        return handle.read()


def first_difference(left, right):
    """Return (line number, left line, right line) of the first difference."""
    left_lines = left.split(b"\n")
    right_lines = right.split(b"\n")
    for index in range(min(len(left_lines), len(right_lines))):
        if left_lines[index] != right_lines[index]:
            return index + 1, left_lines[index], right_lines[index]
    if len(left_lines) != len(right_lines):
        index = min(len(left_lines), len(right_lines))
        left_line = left_lines[index] if index < len(left_lines) else b"<end of file>"
        right_line = right_lines[index] if index < len(right_lines) else b"<end of file>"
        return index + 1, left_line, right_line
    return None


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: tileir_diff.py <left.tileir> <right.tileir>\n")
        return 2
    try:
        left = read_lines(argv[1])
        right = read_lines(argv[2])
    except OSError as err:
        sys.stderr.write("tileir_diff: %s\n" % err)
        return 2

    if left == right:
        return 0

    found = first_difference(left, right)
    line, left_line, right_line = found
    sys.stdout.write("DIFF %s %s\n" % (argv[1], argv[2]))
    sys.stdout.write("first difference at line %d\n" % line)
    sys.stdout.write("  left : %s\n" % left_line.decode("utf-8", "backslashreplace"))
    sys.stdout.write("  right: %s\n" % right_line.decode("utf-8", "backslashreplace"))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
