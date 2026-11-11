import json

import pytest
from PIL import Image

from typhoon_ocr2 import draw_layout, output, parse_elements, parse_json, to_markdown, to_pixels
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
    assert parse_elements(json.dumps({"category": "page", "elements": ELEMENTS})) == ELEMENTS     # as in 0.1.0


def test_parse_ignores_text_after_the_json():
    for raw in ("```json\n" + RAW + "```", RAW + "<|im_end|>", RAW + "\nThat is all.",
                "Here:\n```json\n" + RAW + "\n```\nDone."):
        assert parse_elements(raw) == ELEMENTS


def test_parse_single_element_object():
    assert parse_elements(json.dumps(ELEMENTS[1])) == [ELEMENTS[1]]
    assert parse_elements('{"category": "text", "text": "x"}') == [{"category": "text", "text": "x"}]


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_elements("I cannot read this page.")


def test_parse_takes_only_json_that_starts_the_answer_and_is_complete():
    for raw in ("See note [1] for details. " + RAW,                                # a bracket in prose
                ", ".join(json.dumps(e, ensure_ascii=False) for e in ELEMENTS)):   # objects not inside a list
        with pytest.raises(ValueError):
            parse_elements(raw)
    assert parse_json("I found [1] item") is None and parse_json('{"a": 1}, {"b": 2}') is None


def test_prose_that_starts_with_a_bracket_is_not_layout_json():
    # A references page or a checklist answered in Markdown: as in 0.1.0, no JSON (so the page gets parse_error).
    for raw in ("[1] A. Smith, Deep OCR for Thai, 2020.\n[2] B. Lee, Layout Models, 2021.",
                "[ ] Check the totals\n[x] Sign the form", "[1] See the note. " + RAW, "[1, 2]", '{"pages": [1, 2]}'):
        with pytest.raises(ValueError, match="not a JSON list of elements"):
            parse_elements(raw)
    assert parse_json("[1] A. Smith, 2020.") is None and parse_json("[ ] Check the totals") is None
    assert parse_elements("[]") == [] and parse_json("[1, 2]") == [1, 2]             # whole JSON: as before
    assert parse_elements(RAW + " The last table is cut off.") == ELEMENTS           # elements, then a note


def test_deeply_nested_output_is_not_json_and_does_not_raise_recursion_error():
    # A repetition loop of "[" at the token limit: Python 3.10 cannot decode 1,000 levels, Python 3.14 100,000.
    deep = "[" * 100_000
    raw = "```json\n[" + json.dumps(ELEMENTS[0], ensure_ascii=False) + ', {"bbox": [1, 2, 3, 4], "text": ' + deep
    assert parse_elements(raw) == [ELEMENTS[0]]                                      # the complete element is kept
    with pytest.raises(ValueError):
        parse_elements(deep)
    assert parse_json(deep) is None and parse_json('{"a": ' + deep) is None


def test_a_complete_answer_nested_too_deeply_to_read_is_not_cut_short(monkeypatch):
    # How deep Python reads JSON depends on its version and on the C stack of the platform (3.10 stops at about 1,000
    # levels; 3.14 below 100,000 on macOS, above it on Linux), so the decoder's RecursionError is simulated. An answer
    # whose brackets all close is complete: it is not layout JSON this Python can read (parse_error), not an answer
    # cut off at the token limit, whose last element would be dropped without a word.
    deep, real_loads = "[" * 50 + "]" * 50, json.loads

    def loads(text, *args, **kwargs):
        if deep in text:
            raise RecursionError("maximum recursion depth exceeded")
        return real_loads(text, *args, **kwargs)

    class Decoder(json.JSONDecoder):
        def raw_decode(self, text, *args, **kwargs):
            if deep in text:
                raise RecursionError("maximum recursion depth exceeded")
            return super().raw_decode(text, *args, **kwargs)
    monkeypatch.setattr(output.json, "loads", loads)
    monkeypatch.setattr(output, "_DECODER", Decoder())
    ok, too_deep = json.dumps(ELEMENTS[1], ensure_ascii=False), '{"bbox": [1, 2, 3, 4], "text": ' + deep + "}"
    for raw in ("[" + too_deep + "]", "[" + ok + ", " + too_deep + "]", "[" + too_deep + ", " + ok + "]"):
        with pytest.raises(ValueError, match="nested too deeply"):
            parse_elements("```json\n" + raw + "\n```")
    assert parse_elements("```json\n[" + ok + "]\n```") == [ELEMENTS[1]]    # the decoder still reads the rest


@pytest.mark.parametrize("raw", ["```json\n[]```", "[]```", "```json\n[] ```", "```json\n[]\n```"])
def test_a_page_without_elements_whose_closing_fence_is_on_the_same_line(raw):
    assert parse_elements(raw) == [] and parse_json(raw) == []
    with pytest.raises(ValueError):                          # prose is still not JSON
        parse_elements(raw.replace("[]", "[ ] Check the totals"))


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


def test_parse_json_ignores_text_after_the_json():
    assert parse_json('```json\n{"Total": "355.00"}```') == {"Total": "355.00"}
    assert parse_json('{"a": 1}\nNote: VAT included.') == {"a": 1}
    assert parse_json("The total is 42.") is None


def test_markdown_exclude_selects_categories():
    els = ELEMENTS + [{"bbox": [100, 950, 900, 980], "category": "footer", "text": "Confidential"}]
    assert to_markdown(els, exclude=["footer"]) == "Page 1\n\nAnnual Report\n\nรายได้เพิ่มขึ้น"   # header kept
    assert to_markdown(els, exclude=[]).endswith("Confidential")                                   # everything
    assert to_markdown(els, exclude=["header", "footer", "title"]) == "รายได้เพิ่มขึ้น"


def test_markdown_exclude_unknown_category():
    with pytest.raises(ValueError):
        to_markdown(ELEMENTS, exclude=["footers"])


def test_markdown_tolerates_malformed_elements():
    els = [{"category": "text", "text": "ok"}, {"category": "text", "text": 2568},
           {"category": "table", "text": ["a", "ข"]}, {"text": {"a": 1}}, {"category": "text", "text": None},
           {"category": ["text"], "text": "x"}, "not an element"]
    assert to_markdown(els) == 'ok\n\n2568\n\n["a", "ข"]\n\n{"a": 1}\n\nx'


def test_validate_exclude():
    from typhoon_ocr2 import validate_exclude
    assert validate_exclude(None) is None and validate_exclude(["footer", "header"]) == {"footer", "header"}
    with pytest.raises(ValueError, match="not a string"):
        validate_exclude("footer")
    with pytest.raises(ValueError, match=r"unknown categories \['footers'\]"):
        validate_exclude(["footers"])
    with pytest.raises(ValueError, match="not a string"):
        to_markdown(ELEMENTS, exclude="footer")
    with pytest.raises(ValueError, match=r"e\.g\. \['footer'\], not a string"):     # an example that is accepted
        validate_exclude(" ")
    everything = to_markdown(ELEMENTS, exclude=[])                  # the header too
    assert validate_exclude("") == frozenset() and to_markdown(ELEMENTS, exclude="") == everything   # as in 0.1.0
    assert everything.startswith("Page 1\n\n")
