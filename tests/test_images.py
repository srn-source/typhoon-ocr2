import pytest
from PIL import Image

from typhoon_ocr2 import load_pages, render_pdf_pages, upscale_min_long
from typhoon_ocr2.images import parse_pages


def test_upscale_small_image_keeps_aspect_ratio():
    out = upscale_min_long(Image.new("RGB", (417, 602)))
    assert out.size == (1108, 1600)


def test_upscale_leaves_large_image_alone():
    im = Image.new("RGB", (2480, 3508))
    assert upscale_min_long(im) is im


def test_load_image_from_path_bytes_and_pil(tmp_path, page_image):
    p = tmp_path / "page.png"
    page_image.save(p)
    assert load_pages(str(p))[0].size == (400, 600)
    assert load_pages(p.read_bytes())[0].size == (400, 600)
    assert load_pages(page_image)[0].size == (400, 600)


def test_pages_rejected_for_images(page_image):
    with pytest.raises(ValueError):
        load_pages(_png(page_image), pages=[2])


def test_render_pdf_pages(pdf_bytes):
    pages = render_pdf_pages(pdf_bytes, long_side=1000)
    assert len(pages) == 3
    assert max(pages[0].size) == 1000 and pages[0].mode == "RGB"
    assert len(render_pdf_pages(pdf_bytes, [2])) == 1


def test_render_pdf_page_out_of_range(pdf_bytes):
    with pytest.raises(ValueError):
        render_pdf_pages(pdf_bytes, [4])


def test_load_pdf_from_path(tmp_path, pdf_bytes):
    p = tmp_path / "doc.pdf"
    p.write_bytes(pdf_bytes)
    assert len(load_pages(str(p), [1, 3])) == 2


@pytest.mark.parametrize("spec, expected", [(None, None), (2, [2]), ("1-3,5", [1, 2, 3, 5]), ([4, 1], [4, 1])])
def test_parse_pages(spec, expected):
    assert parse_pages(spec) == expected


def _png(im):
    import io
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
