"""Turn Typhoon OCR V2 output into layout elements, Markdown and pixel boxes."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw

CATEGORIES = ("header", "title", "text", "figure", "table", "formula", "figure_caption", "table_caption",
              "formula_caption", "figure_footnote", "table_footnote", "page_footnote", "footer")
HEADER_FOOTER = frozenset({"header", "footer", "page_footnote"})

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)(?:\n```|$)", re.DOTALL)
_DECODER = json.JSONDecoder()
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"?', re.DOTALL)    # a JSON string, also one cut off (linear time)


def parse_elements(raw: str) -> List[Dict]:
    """Parse the model output, a JSON list of elements, usually inside a ```json fence.

    Text after the JSON (a stop token, a note, a closing fence on the same line) is ignored. If the output
    stopped at the token limit, the last, incomplete element is dropped and the complete ones are returned.
    Raises ValueError when no element can be recovered, also from a list that holds no element, such as [1, 2]
    ([] is a page without elements)."""
    text = _json_text(raw)
    try:
        data = _decode(text)
    except ValueError as e:
        if isinstance(e, _TooDeep) and _closed(text):   # complete, not cut off: no element can be left out
            raise ValueError("the model output is nested too deeply to read") from None
        cut = text.rfind('{"bbox":')
        if cut <= 0:
            raise ValueError("the model output is not a JSON list of elements") from None
        try:
            data = json.loads(text[:cut].rstrip().rstrip(",") + "]")
        except (json.JSONDecodeError, RecursionError):
            raise ValueError("the model output is not a JSON list of elements") from None
    if isinstance(data, dict):           # the prompt asks for "a JSON object"; accept {"...": [elements]}
        lists = [v for v in data.values() if isinstance(v, list)]
        if ("bbox" in data or "category" in data) and all(k == "bbox" for k, v in data.items() if isinstance(v, list)):
            data = [data]                # a single element on its own (its only list is its bbox)
        else:
            data = lists[0] if len(lists) == 1 else [data]
    if not isinstance(data, list):
        raise ValueError("the model output is not a JSON list of elements")
    elements = [e for e in data if isinstance(e, dict)]
    if data and not elements:            # e.g. [1, 2]: numbers or strings, not layout elements
        raise ValueError("the model output is not a JSON list of elements")
    return elements


def validate_exclude(exclude: Optional[Iterable[str]]) -> Optional[FrozenSet[str]]:
    """Check an `exclude` list of categories and return it as a set (None stays None).

    Raises ValueError for a plain string (pass a list, e.g. ["footer"]) and for names not in CATEGORIES. An empty
    string names no category, as in 0.1.0: like [], it keeps everything."""
    if exclude is None:
        return None
    if isinstance(exclude, str):
        if not exclude:
            return frozenset()
        example = [exclude.strip() if exclude.strip() in CATEGORIES else "footer"]
        raise ValueError(f"exclude must be a list of category names, e.g. {example}, not a string")
    names = list(exclude)
    unknown = {str(c) for c in names if not isinstance(c, str) or c not in CATEGORIES}
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}; choose from {list(CATEGORIES)}")
    return frozenset(names)


def to_markdown(elements: Sequence[Dict], keep_header_footer: bool = False,
                exclude: Optional[Iterable[str]] = None) -> str:
    """Join the text of the elements in reading order.

    By default, header, footer and page_footnote elements are left out (we evaluated the model this way).
    `keep_header_footer=True` keeps every element. `exclude` sets exactly which categories to leave out,
    e.g. exclude=["footer"] keeps headers but drops footers, and exclude=[] keeps everything (see
    validate_exclude). Malformed elements do not raise: a number becomes its digits, a list or object its
    JSON, and an element without text is skipped."""
    skip = validate_exclude(exclude)
    if skip is None:
        skip = frozenset() if keep_header_footer else HEADER_FOOTER
    parts = []
    for e in elements:
        if not isinstance(e, Mapping) or not e.get("text"):
            continue
        category = e.get("category")
        if isinstance(category, str) and category in skip:
            continue
        text = e["text"]
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False, default=str) if isinstance(text, (list, dict)) else str(text)
        parts.append(text)
    return "\n\n".join(parts)


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
    """Parse a JSON answer (key-information extraction), with or without a ```json fence. Text after the
    JSON is ignored. Returns None when the answer is not valid JSON."""
    try:
        return _decode(_json_text(raw))
    except ValueError:
        return None


def _json_text(raw: str) -> str:
    m = _FENCE.search(raw)
    text = (m.group(1) if m else raw).strip()
    return text[:-3].rstrip() if text.endswith("```") else text      # a closing fence on the same line: "[]```"


class _TooDeep(ValueError):
    """JSON nested too deeply for this Python to read (3.10 stops at a depth of about 1,000; later versions go deeper,
    as far as the C stack of the platform allows)."""


def _closed(text: str) -> bool:
    """True when every [ and { in the JSON text is closed (those in strings left out): a complete answer, not one
    cut off at the token limit."""
    bare = _STRING.sub("", text)
    return bare.count("[") + bare.count("{") <= bare.count("]") + bare.count("}")


def _decode(text: str):
    """The JSON value in `text`: the whole text when it is valid JSON, else the object, or the array holding an
    object, that the text starts with, ignoring the text after it (a stop token, a note, a closing fence). Raises
    ValueError when the text does not start with one (prose such as "[1] A. Smith, ..." or "[ ] Check the totals"),
    when more JSON follows it (e.g. objects not inside a list), and when it is nested too deeply to read."""
    try:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if not text.startswith(("[", "{")):
                raise
            data, end = _DECODER.raw_decode(text)
            if text[end:].lstrip().startswith((",", "[", "{")) or not (
                    isinstance(data, dict) or (isinstance(data, list) and any(isinstance(e, dict) for e in data))):
                raise
            return data
    except RecursionError:               # e.g. a repetition loop of "[" (Python 3.10 fails at a depth of 1,000)
        raise _TooDeep("the JSON is nested too deeply to read") from None
