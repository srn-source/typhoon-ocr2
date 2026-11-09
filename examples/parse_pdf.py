"""Parse a PDF or image and save Markdown, JSON and a picture of the layout for each page.

    python examples/parse_pdf.py report.pdf --base-url http://localhost:8000/v1
"""
import argparse
import json
from pathlib import Path

from typhoon_ocr2 import TyphoonOCR, draw_layout, load_pages

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--pages", default=None, help="e.g. 1-3")
ap.add_argument("--out", default="out")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(exist_ok=True)
with TyphoonOCR(base_url=a.base_url) as ocr:
    results = ocr.parse(a.file, pages=a.pages)
images = load_pages(a.file, [r.page for r in results] if a.file.lower().endswith(".pdf") else None)
for r, image in zip(results, images):
    stem = out / f"page{r.page:03d}"
    stem.with_suffix(".md").write_text(r.markdown, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(r.elements, ensure_ascii=False, indent=1), encoding="utf-8")
    draw_layout(image, r.elements).save(stem.with_suffix(".png"))
    print(f"page {r.page}: {len(r.elements)} elements" + (" (stopped at the token limit)" if r.truncated else ""))
