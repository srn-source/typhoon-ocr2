import json

import pytest
from PIL import Image

from typhoon_ocr2 import draw_layout, parse_elements, parse_json, to_markdown, to_pixels
from conftest import ELEMENTS

RAW = json.dumps(ELEMENTS, ensure_ascii=False)


def test_parse_fenced():
    assert parse_elements("```json\n" + RAW + "\n```") == ELEMENTS


def test_parse_without_fence():           # Ollama sometimes returns the bare list
    assert parse_elements(RAW) == ELEMENTS


def test_parse_truncated_keeps_complete_elements():
    raw = "```json\n" + RAW[:-1] + ', {"bbox": [1, 2, 3, 4], "category": "text", "text": "cut of'
    assert parse_elements(raw) == ELEMENTS


def test_parse_object_wrapping_a_list():
    assert parse_elements(json.dumps({"elements": ELEMENTS})) == ELEMENTS


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_elements("I cannot read this page.")


def test_markdown_skips_header_footer_by_default():
    assert to_markdown(ELEMENTS) == "Annual Report\n\nรายได้เพิ่มขึ้น"
    assert to_markdown(ELEMENTS, keep_header_footer=True).startswith("Page 1\n\n")


def test_markdown_skips_empty_figures():
    assert to_markdown([{"bbox": [0, 0, 1, 1], "category": "figure", "text": ""}]) == ""


def test_to_pixels_and_swapped_corners():
    assert to_pixels([100, 200, 500, 1000], 2000, 1000) == [200, 200, 1000, 1000]
    assert to_pixels([500, 1000, 100, 200], 2000, 1000) == [200, 200, 1000, 1000]


def test_draw_layout_keeps_size():
    im = Image.new("RGB", (300, 200), "white")
    out = draw_layout(im, ELEMENTS)
    assert out.size == im.size and out is not im


def test_parse_json():
    assert parse_json('```json\n{"Total": "355.00"}\n```') == {"Total": "355.00"}
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json("not json") is None


def test_markdown_exclude_selects_categories():
    els = ELEMENTS + [{"bbox": [100, 950, 900, 980], "category": "footer", "text": "Confidential"}]
    assert to_markdown(els, exclude=["footer"]) == "Page 1\n\nAnnual Report\n\nรายได้เพิ่มขึ้น"   # header kept
    assert to_markdown(els, exclude=[]).endswith("Confidential")                                   # everything
    assert to_markdown(els, exclude=["header", "footer", "title"]) == "รายได้เพิ่มขึ้น"


def test_markdown_exclude_unknown_category():
    with pytest.raises(ValueError):
        to_markdown(ELEMENTS, exclude=["footers"])
