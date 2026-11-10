"""Parse a PDF or image and save Markdown, JSON and a picture of the layout for each page.

    python examples/parse_pdf.py report.pdf --base-url http://localhost:8000/v1 --model typhoon-ai/typhoon-ocr2-2b

Without --base-url it uses $TYPHOON_OCR_BASE_URL, else http://localhost:8000/v1, like the typhoon-ocr2 command.
"""
import argparse
import json
import sys
from pathlib import Path

from typhoon_ocr2 import PartialParseError, TyphoonOCR, draw_layout, iter_pages

ap = argparse.ArgumentParser()
ap.add_argument("file", help="PDF or image (a multi-page TIFF is read page by page)")
ap.add_argument("--base-url", default=None, help="default: $TYPHOON_OCR_BASE_URL or http://localhost:8000/v1")
ap.add_argument("--model", default=None, help="served model name (default: as TyphoonOCR chooses)")
ap.add_argument("--pages", default=None, help="e.g. 1-3")
ap.add_argument("--out", default="out")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
failed = None
with TyphoonOCR(base_url=a.base_url, model=a.model) as ocr:
    try:
        results = ocr.parse(a.file, pages=a.pages)
    except PartialParseError as e:              # a page failed: save the pages that were parsed
        results, failed = e.results, e
# The page images again, one at a time, by the page numbers of the results (PDF pages, TIFF frames, an image).
for r, (_, image) in zip(results, iter_pages(a.file, [r.page for r in results])):
    stem = out / f"page{r.page:03d}"
    stem.with_suffix(".md").write_text(r.markdown, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(r.elements, ensure_ascii=False, indent=1), encoding="utf-8")
    draw_layout(image, r.elements).save(stem.with_suffix(".png"))
    note = (" (stopped at the token limit)" if r.truncated else "") + (f" ({r.parse_error})" if r.parse_error else "")
    print(f"page {r.page}: {len(r.elements)} elements{note}")
if failed:
    sys.exit(f"error: {failed}")
