import argparse
import os

import fitz  # PyMuPDF
from tqdm import tqdm


def convert_pdf_to_markdown(pdf_path: str, out_path: str) -> None:
    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    lines: list[str] = []
    lines.append("# PDF Conversion (text only)\n")
    lines.append(f"- Source: `{os.path.basename(pdf_path)}`\n")
    lines.append(f"- Pages: 1-{page_count} of {page_count}\n")
    lines.append("\n---\n")

    for page_number in tqdm(range(1, page_count + 1), desc="Pages"):
        page = doc.load_page(page_number - 1)

        lines.append("\n---\n")
        lines.append(f"## Page {page_number}\n")

        text = (page.get_text("text") or "").strip()
        if text:
            lines.append(text)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PDF(s) to Markdown using text-only extraction (no AI, no images)."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdf",
        help="Path to a single PDF file to convert.",
    )
    group.add_argument(
        "--input-dir",
        help=(
            "Convert all .pdf files in this directory (non-recursive). "
            "Each PDF becomes a .md with the same base name."
        ),
    )
    parser.add_argument(
        "--out",
        help=(
            "Output Markdown file path (only valid with --pdf). "
            "If omitted with --pdf, uses <pdf_basename>.md next to the PDF."
        ),
    )

    args = parser.parse_args()

    if args.pdf:
        pdf_path = args.pdf
        if not os.path.isfile(pdf_path):
            raise SystemExit(f"PDF not found: {pdf_path}")

        out_path = args.out
        if not out_path:
            base, _ = os.path.splitext(pdf_path)
            out_path = base + ".md"

        print(f"Converting single PDF to Markdown (text only): {pdf_path} -> {out_path}")
        convert_pdf_to_markdown(pdf_path, out_path)
        print("Done.")
        return 0

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Directory not found: {input_dir}")

    pdf_files = [
        f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print("No .pdf files found in directory:", input_dir)
        return 0

    print(f"Converting {len(pdf_files)} PDF(s) in {input_dir} (text only).")

    for name in pdf_files:
        pdf_path = os.path.join(input_dir, name)
        base, _ = os.path.splitext(pdf_path)
        out_path = base + ".md"

        print(f"\n=== {name} -> {os.path.basename(out_path)} ===")
        convert_pdf_to_markdown(pdf_path, out_path)

    print("\nAll conversions complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

