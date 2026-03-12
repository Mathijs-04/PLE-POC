import argparse
import os
from typing import Iterable, List


CHAR_REPLACEMENTS = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "–": "-",
    "—": "-",
    "−": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "…": "...",
    "®": "",
    "™": "",
    "\u00a0": " ",  # non‑breaking space
}


def normalize_line(s: str) -> str:
    for bad, good in CHAR_REPLACEMENTS.items():
        s = s.replace(bad, good)
    return s


def reflow_paragraphs(lines: Iterable[str]) -> List[str]:
    """
    Very simple Markdown-friendly reflow:
    - Keeps headings, lists, code fences, and horizontal rules as-is.
    - Merges consecutive "body" lines into single paragraphs.
    """
    out: List[str] = []
    buf: str = ""

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            out.append(buf)
            buf = ""

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Structural markers: keep exact line break
        is_heading = stripped.startswith("#")
        is_hr = stripped == "---"
        is_list = stripped.startswith("- ") or stripped.startswith("* ")
        is_code_fence = stripped.startswith("```")
        is_quote = stripped.startswith("> ")

        if stripped == "":
            flush_buf()
            out.append("")
            continue

        if is_heading or is_hr or is_list or is_code_fence or is_quote:
            flush_buf()
            out.append(line)
            continue

        # Normal paragraph text: accumulate into a single line
        if not buf:
            buf = stripped
        else:
            buf += " " + stripped

    flush_buf()
    return out


def clean_markdown(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # 1) Normalize broken characters
    normalized = [normalize_line(line) for line in raw_lines]

    # 2) Reflow paragraphs
    cleaned_lines = reflow_paragraphs(normalized)

    with open(out_path, "w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lightly clean a Markdown file: fix common ligatures/symbols and "
            "reflow plain text paragraphs while preserving headings/lists."
        )
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input Markdown file to clean.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        help=(
            "Output Markdown path. If omitted, writes <input_basename>_clean.md "
            "in the same directory."
        ),
    )

    args = parser.parse_args()

    in_path = args.in_path
    if not os.path.isfile(in_path):
        raise SystemExit(f"Input file not found: {in_path}")

    out_path = args.out_path
    if not out_path:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_clean{ext or '.md'}"

    print(f"Cleaning Markdown: {in_path} -> {out_path}")
    clean_markdown(in_path, out_path)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

