"""Runs against a real server: TYPHOON_OCR2_TEST_URL=http://localhost:8000/v1 pytest -m integration"""
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from typhoon_ocr2 import TyphoonOCR

URL = os.getenv("TYPHOON_OCR2_TEST_URL")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not URL, reason="set TYPHOON_OCR2_TEST_URL")]
FONT = os.getenv("TYPHOON_OCR2_TEST_FONT")    # a Thai .ttf, e.g. Sarabun-Regular.ttf


def _page():
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    big = ImageFont.truetype(FONT, 56) if FONT else ImageFont.load_default()
    small = ImageFont.truetype(FONT, 34) if FONT else ImageFont.load_default()
    d.text((120, 120), "รายงานประจำปี 2568", font=big, fill="black")
    d.text((120, 260), "บริษัทมีรายได้รวม 1,250 ล้านบาท เพิ่มขึ้นร้อยละ 12", font=small, fill="black")
    d.text((120, 320), "Total revenue: 1,250 million baht", font=small, fill="black")
    return im


@pytest.fixture(scope="module")
def ocr():
    with TyphoonOCR(base_url=URL, model=os.getenv("TYPHOON_OCR2_TEST_MODEL")) as o:
        yield o


def test_parse_page(ocr):
    [r] = ocr.parse(_page())
    assert r.finish_reason == "stop" and r.elements
    assert "1,250" in r.markdown


def test_ask(ocr):
    assert "1250" in ocr.ask(_page(), "What is the total revenue? Answer with the number only.").replace(",", "")


def test_pdf(ocr, tmp_path):
    p = tmp_path / "doc.pdf"
    _page().save(p, format="PDF", resolution=150)
    [r] = ocr.parse(str(p))
    assert "1,250" in r.markdown
