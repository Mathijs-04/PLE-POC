import argparse
import base64
import hashlib
import json
import os
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def _response_text(response: Any) -> str:
    """
    Best-effort extraction of text from OpenAI Responses API return object.
    """
    try:
        return response.output_text
    except Exception:
        pass

    try:
        return response.output[0].content[0].text
    except Exception:
        return str(response)


def _mime_from_ext(ext: str) -> str:
    ext = (ext or "").lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")


def _safe_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def classify_image_informational(
    *,
    client: OpenAI,
    model: str,
    image_bytes: bytes,
    mime_type: str,
    page_number: int,
    image_index: int,
    page_text_context: str,
) -> tuple[bool, str]:
    """
    Returns (should_describe, reason).

    Goal: skip decorative content (logos, backgrounds, artwork) and keep
    informational visuals (diagrams, charts, tables, rules-reference icons).
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")

    context = (page_text_context or "").strip()
    if len(context) > 1000:
        context = context[:1000] + "…"

    prompt = f"""Classify the image from Page {page_number}, Image {image_index} for a RAG dataset.

We ONLY want informational visuals: diagrams, flowcharts, tables, charts, labelled rules-reference figures.
We want to SKIP: logos, decorative borders, background textures, full-page artwork/illustrations, purely aesthetic photos.

Use the nearby page text only as weak context:
{context if context else "[no page text available]"}

Return ONLY valid JSON with this schema:
{{
  "informational": true|false,
  "category": "diagram|table|chart|map|annotated_figure|icon|logo|border|background|artwork|photo|other",
  "reason": "short reason"
}}
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{b64}",
                    },
                ],
            }
        ],
    )

    text = _response_text(response).strip()
    obj = _safe_parse_json_object(text)
    if not obj:
        # If the model didn't comply, default to describing rather than dropping data.
        return True, "classifier_unparseable_json"

    informational = bool(obj.get("informational"))
    category = str(obj.get("category") or "other")
    reason = str(obj.get("reason") or "")
    return informational, f"{category}: {reason}".strip(": ").strip() or category


def _image_rect_coverage_fraction(page: fitz.Page, xref: int) -> float:
    """
    Computes the fraction of the page area covered by this image's rectangles.
    If unknown, returns 0.0.
    """
    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return 0.0
    if not rects:
        return 0.0

    page_rect = page.rect
    page_area = float(page_rect.width * page_rect.height) if page_rect else 0.0
    if page_area <= 0:
        return 0.0

    covered = 0.0
    for r in rects:
        covered += float(r.width * r.height)
    return max(0.0, min(1.0, covered / page_area))


def describe_image(
    *,
    client: OpenAI,
    model: str,
    image_bytes: bytes,
    mime_type: str,
    page_number: int,
    image_index: int,
    page_text_context: str,
) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")

    context = (page_text_context or "").strip()
    if len(context) > 1500:
        context = context[:1500] + "…"

    prompt = f"""You are converting a PDF into high-quality Markdown for vector search / RAG.

Task: Describe the image from Page {page_number}, Image {image_index}.

Requirements:
- Be very detailed and literal about what is visible (layout, labels, arrows, callouts, icons, table structure).
- If there is any text in the image, transcribe it exactly (preserve numbers/keywords).
- If it’s a diagram/flowchart, explain the relationships and the meaning of connectors.
- If it’s a table, output a clean Markdown table (when feasible) plus a short explanation.
- If it’s decorative (e.g., logo, border), say so briefly.
- Do NOT assume rules context that isn’t visible in the image.

Nearby page text (may help disambiguate):
{context if context else "[no page text available]"}
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{b64}",
                    },
                ],
            }
        ],
    )

    return _response_text(response).strip()


def main() -> int:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in your environment or a .env file."
        )

    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown with detailed AI image descriptions."
    )
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")
    parser.add_argument(
        "--out",
        required=True,
        help="Path to output Markdown file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Vision-capable model name.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-based start page (inclusive).",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="1-based end page (inclusive). Omit for last page.",
    )
    parser.add_argument(
        "--min-image-area",
        type=int,
        default=20_000,
        help="Skip very small images by area in pixels (width*height).",
    )
    parser.add_argument(
        "--max-page-coverage",
        type=float,
        default=0.35,
        help="Skip images covering more than this fraction of the page (likely background/artwork).",
    )
    parser.add_argument(
        "--ai-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AI to filter out logos/backgrounds/artwork before describing.",
    )
    parser.add_argument(
        "--filter-model",
        default="gpt-4.1-mini",
        help="Vision-capable model name for filtering (can be cheaper/faster).",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional JSON cache file for image descriptions (dedupe by sha256).",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=api_key)

    cache: dict[str, str] = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                cache = {str(k): str(v) for k, v in loaded.items()}

    doc = fitz.open(args.pdf)
    page_count = doc.page_count
    start_page = max(1, args.start_page)
    end_page = min(page_count, args.end_page if args.end_page else page_count)
    if start_page > end_page:
        raise ValueError(f"Invalid page range: {start_page}..{end_page}")

    md: list[str] = []
    md.append(f"# PDF Conversion\n")
    md.append(f"- Source: `{os.path.basename(args.pdf)}`\n")
    md.append(f"- Pages: {start_page}-{end_page} of {page_count}\n")
    md.append("\n---\n")

    for page_number in tqdm(range(start_page, end_page + 1), desc="Pages"):
        page = doc.load_page(page_number - 1)

        md.append(f"\n---\n")
        md.append(f"## Page {page_number}\n")

        page_text = (page.get_text("text") or "").strip()
        if page_text:
            md.append(page_text)

        images = page.get_images(full=True)
        if not images:
            continue

        md.append("\n### Images\n")

        for i, img in enumerate(images, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes: bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            mime_type = _mime_from_ext(ext)

            w = int(base_image.get("width") or 0)
            h = int(base_image.get("height") or 0)
            if w > 0 and h > 0 and (w * h) < int(args.min_image_area):
                continue

            coverage = _image_rect_coverage_fraction(page, xref)
            if coverage > float(args.max_page_coverage):
                # likely background texture / full-page art
                continue

            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest in cache:
                description = cache[digest]
            else:
                if args.ai_filter:
                    should_describe, reason = classify_image_informational(
                        client=client,
                        model=args.filter_model,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        page_number=page_number,
                        image_index=i,
                        page_text_context=page_text,
                    )
                    if not should_describe:
                        # Skip decorative content entirely.
                        continue

                description = describe_image(
                    client=client,
                    model=args.model,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    page_number=page_number,
                    image_index=i,
                    page_text_context=page_text,
                )
                cache[digest] = description

            md.append(f"\n#### Image {i}\n")
            if w and h:
                md.append(f"- Size: {w}×{h}\n")
            md.append(description)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md).rstrip() + "\n")

    if args.cache:
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    print("Markdown generated:", args.out)
    if args.cache:
        print("Cache updated:", args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())