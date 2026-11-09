"""Load pages from images and PDFs, and prepare them for the model."""
from __future__ import annotations

import base64
import io
import os
from typing import Iterable, List, Optional, Sequence, Union

from PIL import Image

ImageInput = Union[str, os.PathLike, bytes, Image.Image]

# Images whose long side is below this are enlarged before they are sent. On ThaiOCRBench this raised
# Typhoon OCR V2 2B from 59.9 to 62.9 and 0.8B from 37.3 to 41.3; on large pages it changes nothing.
UPSCALE_MIN_LONG = 1600
# PDF pages are rendered with this long side, as in our olmOCR-bench evaluation.
PDF_LONG_SIDE = 2048


def upscale_min_long(image: Image.Image, min_long: int = UPSCALE_MIN_LONG) -> Image.Image:
    """Enlarge the image so that its long side is at least `min_long` pixels, keeping the aspect
    ratio. Images that are already large enough are returned unchanged."""
    scale = min_long / max(image.size)
    if scale <= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def is_pdf(source: ImageInput) -> bool:
    if isinstance(source, (str, os.PathLike)):
        return str(source).lower().endswith(".pdf")
    if isinstance(source, bytes):
        return source[:5] == b"%PDF-"
    return False


def pdf_page_count(source: Union[str, os.PathLike, bytes]) -> int:
    pdfium = _pdfium()
    doc = pdfium.PdfDocument(source)
    try:
        return len(doc)
    finally:
        doc.close()


def render_pdf_pages(source: Union[str, os.PathLike, bytes], pages: Optional[Iterable[int]] = None,
                     long_side: int = PDF_LONG_SIDE) -> List[Image.Image]:
    """Render PDF pages (1-based numbers; all pages when `pages` is None) to RGB images whose long side
    is `long_side` pixels."""
    pdfium = _pdfium()
    doc = pdfium.PdfDocument(source)
    try:
        n = len(doc)
        numbers = list(range(1, n + 1)) if pages is None else list(pages)
        out = []
        for p in numbers:
            if not 1 <= p <= n:
                raise ValueError(f"page {p} is out of range: the PDF has {n} page(s)")
            page = doc[p - 1]
            w, h = page.get_size()                      # in points (1/72 inch)
            bitmap = page.render(scale=long_side / max(w, h))
            out.append(bitmap.to_pil().convert("RGB"))
            page.close()
        return out
    finally:
        doc.close()


def load_pages(source: ImageInput, pages: Optional[Iterable[int]] = None,
               pdf_long_side: int = PDF_LONG_SIDE) -> List[Image.Image]:
    """Return the pages of a PDF, or a single image, as RGB PIL images.

    `source` can be a file path, raw bytes, or a PIL image. `pages` (1-based) applies to PDFs only."""
    if isinstance(source, Image.Image):
        return [source.convert("RGB")]
    if is_pdf(source):
        return render_pdf_pages(source, pages, pdf_long_side)
    if pages is not None and list(pages) not in ([], [1]):
        raise ValueError("`pages` can only be used with PDFs; an image has one page")
    data = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(data) as im:
        return [im.convert("RGB")]


def to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def to_data_url(image: Image.Image) -> str:
    return "data:image/png;base64," + to_base64_png(image)


def parse_pages(spec: Optional[Union[str, int, Sequence[int]]]) -> Optional[List[int]]:
    """'1-3,5' -> [1, 2, 3, 5]; an int -> [int]; None -> None (all pages)."""
    if spec is None or isinstance(spec, (list, tuple)):
        return None if spec is None else [int(p) for p in spec]
    if isinstance(spec, int):
        return [spec]
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _pdfium():
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover - it is a dependency, but give a clear message anyway
        raise ImportError("PDF support needs pypdfium2: pip install pypdfium2") from e
    return pdfium
