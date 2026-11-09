"""Turn Typhoon OCR V2 output into layout elements, Markdown and pixel boxes."""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw

CATEGORIES = ("header", "title", "text", "figure", "table", "formula", "figure_caption", "table_caption",
              "formula_caption", "figure_footnote", "table_footnote", "page_footnote", "footer")
HEADER_FOOTER = frozenset({"header", "footer", "page_footnote"})

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)(?:\n```|$)", re.DOTALL)


def parse_elements(raw: str) -> List[Dict]:
    """Parse the model output, a JSON list of elements, usually inside a ```json fence.

    If the output stopped at the token limit, the last, incomplete element is dropped and the complete
    ones are returned. Raises ValueError when no element can be recovered."""
    m = _FENCE.search(raw)
    text = (m.group(1) if m else raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        cut = text.rfind('{"bbox":')
        if cut <= 0:
            raise ValueError("the model output is not a JSON list of elements") from None
        try:
            data = json.loads(text[:cut].rstrip().rstrip(",") + "]")
        except json.JSONDecodeError:
            raise ValueError("the model output is not a JSON list of elements") from None
    if isinstance(data, dict):           # the prompt asks for "a JSON object"; accept {"...": [elements]}
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if len(lists) == 1 else [data]
    if not isinstance(data, list):
        raise ValueError("the model output is not a JSON list of elements")
    return [e for e in data if isinstance(e, dict)]


def to_markdown(elements: Sequence[Dict], keep_header_footer: bool = False,
                exclude: Optional[Iterable[str]] = None) -> str:
    """Join the text of the elements in reading order.

    By default, header, footer and page_footnote elements are left out (we evaluated the model this way).
    `keep_header_footer=True` keeps every element. `exclude` sets exactly which categories to leave out,
    e.g. exclude=["footer"] keeps headers but drops footers, and exclude=[] keeps everything."""
    if exclude is not None:
        skip = set(exclude)
        unknown = skip - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown categories {sorted(unknown)}; choose from {list(CATEGORIES)}")
    else:
        skip = set() if keep_header_footer else HEADER_FOOTER
    return "\n\n".join(e["text"] for e in elements if e.get("text") and e.get("category") not in skip)


def to_pixels(bbox: Sequence[float], width: int, height: int) -> List[float]:
    """Map a box on the model's 0-1000 grid to pixels of an image of the given size."""
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000]


_COLORS = {"title": (228, 87, 46), "text": (91, 84, 214), "table": (42, 157, 85), "figure": (192, 138, 0),
           "formula": (181, 23, 158)}


def draw_layout(image: Image.Image, elements: Sequence[Dict], numbers: bool = True) -> Image.Image:
    """Return a copy of `image` with the element boxes drawn on it, numbered in reading order."""
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    lw = max(2, round(max(w, h) / 600))
    for i, e in enumerate(elements, 1):
        if not e.get("bbox") or len(e["bbox"]) != 4:
            continue
        box = to_pixels(e["bbox"], w, h)
        c = _COLORS.get(e.get("category"), (127, 127, 127))
        d.rectangle(box, outline=c + (255,), width=lw, fill=c + (28,))
        if numbers:
            d.text((box[0] + lw, box[1] + lw), str(i), fill=c + (255,))
    return out


def parse_json(raw: str) -> Optional[object]:
    """Parse a JSON answer (key-information extraction), with or without a ```json fence.
    Returns None when the answer is not valid JSON."""
    m = _FENCE.search(raw)
    try:
        return json.loads((m.group(1) if m else raw).strip())
    except json.JSONDecodeError:
        return None
