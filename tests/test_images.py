import gc
import io
import logging
import os
import struct
import threading
import tracemalloc
import warnings
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, TiffImagePlugin

from typhoon_ocr2 import images, load_pages, render_pdf_pages, upscale_min_long
from typhoon_ocr2.images import parse_pages

DATA = Path(__file__).parent / "data"


def test_upscale_small_image_keeps_aspect_ratio():
    out = upscale_min_long(Image.new("RGB", (417, 602)))
    assert out.size == (1108, 1600)


def test_upscale_leaves_large_image_alone():
    im = Image.new("RGB", (2480, 3508))
    assert upscale_min_long(im) is im


def test_upscale_leaves_empty_image_alone():
    for size in ((0, 0), (0, 10)):
        im = Image.new("RGB", size)
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


def test_pdf_without_extension(tmp_path, pdf_bytes):
    p = tmp_path / "upload_7d1e"
    p.write_bytes(pdf_bytes)
    assert images.is_pdf(str(p)) and images.is_pdf(p) and len(load_pages(p)) == 3
    assert not images.is_pdf(str(tmp_path / "missing")) and not images.is_pdf(tmp_path)


def test_iter_pages_checks_every_page_first_and_renders_one_at_a_time(monkeypatch, pdf_bytes):
    rendered = []
    real = images._render_page
    monkeypatch.setattr(images, "_render_page", lambda doc, n, side: rendered.append(n) or real(doc, n, side))
    with pytest.raises(ValueError, match="page 4 is out of range"):
        next(images.iter_pages(pdf_bytes, [1, 4]))
    assert rendered == []                         # page 4 was refused before page 1 was rendered
    pages = images.iter_pages(pdf_bytes, [3, 1], pdf_long_side=200)
    n, page = next(pages)
    assert (n, max(page.size), rendered) == (3, 200, [3])
    pages.close()                                 # stopping early closes the document
    assert [n for n, _ in images.iter_pages(pdf_bytes, pdf_long_side=100)] == [1, 2, 3]


def test_pdfium_calls_hold_the_lock(monkeypatch, pdf_bytes):
    # PDFium crashes the process when two threads call it at once, even for different documents.
    import pypdfium2 as pdfium
    seen = []

    def checked(f):
        def call(*args, **kwargs):
            seen.append(images._PDFIUM_LOCK._is_owned())          # held by this thread
            return f(*args, **kwargs)
        return call
    for cls, names in ((pdfium.PdfDocument, ("__init__", "init_forms", "__len__", "__getitem__", "close")),
                       (pdfium.PdfPage, ("get_size", "render", "close")), (pdfium.PdfBitmap, ("to_pil", "close"))):
        for name in names:
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, checked(getattr(cls, name)))
    load_pages(pdf_bytes, pdf_long_side=100)
    images.pdf_page_count(pdf_bytes)
    assert len(seen) > 10 and all(seen)


def test_an_abandoned_iterator_closed_in_the_middle_of_a_render_does_not_deadlock(monkeypatch, pdf_bytes):
    # The garbage collector closes an unfinished iter_pages() held in a reference cycle at any allocation, also in
    # the middle of a render, in the thread that holds the PDFium lock.
    monkeypatch.setattr(images, "_PDFIUM_LOCK", type(images._PDFIUM_LOCK)())   # a deadlock stays in this test
    real = images._render_page

    def render(*args):
        gc.collect()                                # the allocation that happens to start a collection
        return real(*args)
    monkeypatch.setattr(images, "_render_page", render)

    class Holder:
        pass
    enabled = gc.isenabled()
    gc.disable()                                    # so that the cycle is collected inside the render
    try:
        holder = Holder()
        holder.pages, holder.me = images.iter_pages(pdf_bytes, pdf_long_side=50), holder
        next(holder.pages)
        del holder
        loader = threading.Thread(target=load_pages, args=(pdf_bytes, [1], 50), daemon=True)
        loader.start()
        loader.join(10)
    finally:
        if enabled:
            gc.enable()
    assert not loader.is_alive()


def test_closing_an_iterator_never_waits_for_the_pdfium_lock(monkeypatch, pdf_bytes):
    # The garbage collector can close an abandoned iter_pages() in a thread that holds logging's lock while another
    # thread holds the PDFium lock and waits for logging's (pypdfium2 logs inside init_forms()): waiting for the
    # PDFium lock there deadlocks both threads. The document is closed by the next holder of the lock instead.
    monkeypatch.setattr(images, "_PDFIUM_LOCK", type(images._PDFIUM_LOCK)())   # a deadlock stays in this test
    pages = images.iter_pages(pdf_bytes, pdf_long_side=50)
    next(pages)
    held, done = threading.Event(), threading.Event()

    def hold():
        with images._pdfium_locked():
            held.set()
            done.wait(10)
    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(10)
    closer = threading.Thread(target=pages.close, daemon=True)
    closer.start()
    closer.join(5)
    assert not closer.is_alive() and len(images._UNCLOSED) == 1     # left to the thread that holds the lock
    done.set()
    holder.join(10)
    assert not holder.is_alive() and not images._UNCLOSED          # closed when that thread released the lock


def test_no_logger_is_looked_up_under_the_pdfium_lock(monkeypatch, pdf_bytes):
    # getLogger() takes logging's module lock: under the PDFium lock, it is the other half of the deadlock above.
    real, under_lock = logging.getLogger, []

    def get_logger(*args, **kwargs):
        under_lock.append(images._PDFIUM_LOCK._is_owned())
        return real(*args, **kwargs)
    monkeypatch.setattr(logging, "getLogger", get_logger)
    load_pages(pdf_bytes, pdf_long_side=50)
    images.pdf_page_count(pdf_bytes)
    assert not any(under_lock)


def test_pdfs_rendered_from_many_threads_at_once(pdf_bytes):
    docs = [pdf_bytes, (DATA / "form.pdf").read_bytes(), _pdf(["gray", "white"]), _pdf(["white"] * 4, (300, 500))]
    expected = [[p.tobytes() for p in load_pages(d, pdf_long_side=120)] for d in docs]

    def job(i):
        return i % len(docs), [p.tobytes() for p in load_pages(docs[i % len(docs)], pdf_long_side=120)]
    with ThreadPoolExecutor(8) as pool:
        for d, got in pool.map(job, range(240)):
            assert got == expected[d]


def test_filled_form_fields_are_drawn(pdf_bytes):
    # tests/data/form.pdf: a 240 x 60 pt page holding only a text field filled in with "SOMCHAI 1,250.00", with its
    # appearance stream, made once with reportlab 5.1 (canvas.acroForm.textfield(value=..., borderWidth=0)).
    # Without form rendering it comes out blank.
    page = load_pages(DATA / "form.pdf", pdf_long_side=480)[0]
    assert page.size == (480, 120) and page.convert("L").getextrema()[0] < 100
    import pypdfium2 as pdfium                      # a PDF without forms renders exactly as without init_forms()
    doc = pdfium.PdfDocument(pdf_bytes)
    page = doc[1]
    before = page.render(scale=500 / max(page.get_size())).to_pil().convert("RGB")
    page.close()
    doc.close()
    assert load_pages(pdf_bytes, [2], 500)[0].tobytes() == before.tobytes()


def test_an_xfa_form_is_drawn_without_pypdfium2s_notice(caplog):
    # Adobe LiveCycle and AEM forms carry XFA, which the PyPI builds of PDFium cannot render: pypdfium2 5 logs, for
    # each one, how to build PDFium with XFA (with no logging set up, Python prints it on stderr). The page is drawn
    # as it is without forms set up, as 0.1.0 drew it, and nothing is logged.
    import pypdfium2 as pdfium
    data = _xfa_pdf()
    doc = pdfium.PdfDocument(data)
    try:
        if hasattr(doc, "get_formtype"):                # pypdfium2 >= 4.2: the test reaches the XFA case
            assert doc.get_formtype() in (pdfium.raw.FORMTYPE_XFA_FULL, pdfium.raw.FORMTYPE_XFA_FOREGROUND)
        page = doc[0]
        expected = page.render(scale=500 / 792).to_pil().convert("RGB").tobytes()
        page.close()
    finally:
        doc.close()
    [page] = load_pages(data, pdf_long_side=500)
    assert page.tobytes() == expected and page.convert("L").getextrema()[0] < 100     # the text is there
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def _xfa_pdf() -> bytes:
    """A one-page PDF with a line of text whose AcroForm has an XFA packet, as static XFA forms have."""
    content = b"BT /F1 24 Tf 72 700 Td (Invoice 2568/001) Tj ET"
    xfa = (b'<xdp:xdp xmlns:xdp="http://ns.adobe.com/xdp/"><template '
           b'xmlns="http://www.xfa.org/schema/xfa-template/3.3/"/></xdp:xdp>')
    objects = [b"<< /Type /Catalog /Pages 2 0 R /AcroForm 6 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", b"<< /Fields [] /XFA 7 0 R >>",
               b"<< /Length %d >>\nstream\n%s\nendstream" % (len(xfa), xfa)]
    out, offsets = bytearray(b"%PDF-1.7\n"), []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (n, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_pdfs_given_as_other_path_like_objects(tmp_path, pdf_bytes):
    # os.scandir() entries and other os.PathLike objects: pypdfium2 itself takes a str or pathlib.Path only.
    class FsPath:
        def __init__(self, path):
            self.path = path

        def __fspath__(self):
            return self.path
    (tmp_path / "report.pdf").write_bytes(pdf_bytes)
    (tmp_path / "upload_7d1e").write_bytes(pdf_bytes)              # no extension: a PDF by its first bytes
    expected = [p.tobytes() for p in load_pages(str(tmp_path / "report.pdf"), [2], pdf_long_side=100)]
    for source in sorted(os.scandir(tmp_path), key=lambda e: e.name) + [FsPath(str(tmp_path / "report.pdf"))]:
        assert images.is_pdf(source) and images.pdf_page_count(source) == 3
        assert [p.tobytes() for p in load_pages(source, [2], pdf_long_side=100)] == expected


def test_password_protected_or_damaged_pdf_gives_value_error():
    # tests/data/locked.pdf: a one-page PDF encrypted with pypdf 7.0 (AES-256, user password "1234").
    with pytest.raises(ValueError, match="password-protected or damaged"):
        load_pages(DATA / "locked.pdf")
    with pytest.raises(ValueError, match="cannot open the PDF"):
        load_pages(b"%PDF-1.7 cut off")


def test_a_missing_pdf_says_why(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"No such file or directory: .*missing\.pdf"):
        load_pages(tmp_path / "missing.pdf")


def test_exif_orientation_is_applied(tmp_path):
    upright = Image.new("RGB", (60, 80), "white")
    ImageDraw.Draw(upright).rectangle((0, 0, 59, 15), fill="black")      # a dark band at the top
    exif = Image.Exif()
    exif[0x0112] = 6                                                     # stored sideways, as phones do
    path = tmp_path / "photo.jpg"
    upright.rotate(90, expand=True).save(path, quality=95, exif=exif.tobytes())
    for source in (str(path), path.read_bytes(), Image.open(path)):
        page = load_pages(source)[0]
        assert page.size == (60, 80)
        assert page.crop((0, 2, 60, 12)).convert("L").getextrema()[1] < 80
        assert page.crop((0, 30, 60, 80)).convert("L").getextrema()[0] > 170
    plain = tmp_path / "plain.jpg"                                       # no orientation tag: exactly as before
    upright.save(plain, quality=95)
    with Image.open(plain) as im:
        assert load_pages(str(plain))[0].tobytes() == im.convert("RGB").tobytes()
    by_hand = Image.open(path).rotate(-90, expand=True)                  # turned upright by the caller, tag still 6:
    assert by_hand.getexif().get(0x0112) == 6                            # sent as given, not turned again
    assert load_pages(by_hand)[0].tobytes() == by_hand.convert("RGB").tobytes()
    tiff = io.BytesIO()                                                  # Pillow turns TIFFs upright itself
    upright.convert("L").rotate(90, expand=True).save(tiff, "TIFF", tiffinfo={0x0112: 6})
    page = load_pages(tiff.getvalue())[0]
    assert page.size == (60, 80) and page.crop((0, 2, 60, 12)).convert("L").getextrema()[1] < 80


def test_transparent_images_are_sent_as_before():
    # Transparent images get the plain convert("RGB") of 0.1.0: a transparent area shows the colour stored under it.
    # No background fits every image (black text needs white, white text needs black), so none is chosen.
    clear = Image.new("RGBA", (40, 20), (0, 0, 0, 0))                    # transparent black, as canvas exports
    ImageDraw.Draw(clear).rectangle((5, 5, 15, 15), fill=(0, 0, 0, 255))
    palette = Image.new("P", (40, 20), 0)
    palette.putpalette([0, 0, 0] * 256)
    ImageDraw.Draw(palette).rectangle((5, 5, 15, 15), fill=1)
    palette.info["transparency"] = 0
    edges = Image.new("RGBA", (256, 20), (0, 0, 0, 0))
    edges.putalpha(Image.linear_gradient("L").rotate(90).resize((256, 20)))     # every alpha from 0 to 255
    blue = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    ImageDraw.Draw(blue).rectangle((5, 5, 15, 15), fill=(0, 0, 255, 255))
    opaque = Image.new("RGBA", (40, 20), (200, 100, 50, 255))
    for source in (clear, clear.convert("LA"), palette, _png(clear), _png(palette, transparency=0), edges, blue,
                   opaque):
        before = source if isinstance(source, Image.Image) else Image.open(io.BytesIO(source))
        assert images.to_base64_png(load_pages(source)[0]) == images.to_base64_png(before.convert("RGB"))


def test_16_bit_images_keep_their_text():
    grey = Image.new("L", (40, 20), 235)                                 # a 16-bit scan: paper 235, ink 30, times 257
    ImageDraw.Draw(grey).rectangle((5, 5, 15, 15), fill=30)
    scan = Image.frombytes("I;16", grey.size, b"".join((v * 257).to_bytes(2, "little") for v in grey.tobytes()))
    for source in (scan, _png(scan)):
        page = load_pages(source)[0]
        assert page.getpixel((0, 0)) == (235, 235, 235) and page.getpixel((10, 10)) == (30, 30, 30)
    small = Image.new("I", (4, 4), 200)                                  # values that fit in 8 bits: as before
    assert load_pages(small)[0].tobytes() == small.convert("RGB").tobytes()
    assert load_pages(Image.new("I;16", (0, 5)))[0].size == (0, 5)      # empty: no extrema to look at


def _grey16(paper, ink, mode="I;16", border=None):
    """A 400 x 120 greyscale page in `mode` ("I;16" or "I"): `paper` with a block of `ink`, and a 1 px frame of value
    `border`. Made from bytes: Pillow 9 turns "I" values above 32767 into 32767 when it converts them to "I;16"."""
    def value(x, y):
        if border is not None and (x in (0, 399) or y in (0, 119)):
            return border
        return ink if 40 <= x <= 360 and 40 <= y <= 80 else paper
    values = [value(x, y) for y in range(120) for x in range(400)]
    return Image.frombytes(mode, (400, 120), struct.pack(f"<{len(values)}" + ("H" if mode == "I;16" else "i"), *values),
                           "raw", "I;16" if mode == "I;16" else "I;32S")


def test_16_bit_greyscale_whose_ink_is_below_256_is_sent_as_before():
    # 0.1.0 turned every value above 255 white. When the ink is below 256 that page was readable, so it is sent
    # byte for byte as before: 8-bit values stored in 16 bits with one value above 255 (a sharpening overshoot),
    # 10-bit data whose paper is above 255, and an 8-bit scan brightened in Pillow's 32-bit "I" mode.
    overshoot = _grey16(240, 20)
    overshoot.putpixel((0, 0), 256)
    brightened = _grey16(230, 40, "I").point(lambda v: v * 1.15)        # paper 264
    for page in (overshoot, _grey16(980, 60), brightened):
        for source in (page, _png(page) if page.mode == "I;16" else page):
            before = source if isinstance(source, Image.Image) else Image.open(io.BytesIO(source))
            sent = load_pages(source)[0]
            assert images.to_base64_png(sent) == images.to_base64_png(before.convert("RGB"))
            assert sent.getpixel((200, 60))[0] < 100                    # the ink is still there


def test_16_bit_greyscale_whose_ink_is_above_255_is_scaled_by_its_depth():
    # 0.1.0 sent these as blank white pages. They are scaled by the depth of the data: 12-bit data by 4095, not by
    # 65535, which would make it black; a black frame around a 16-bit scan does not keep it white.
    for paper, ink, border, expected in ((3900, 300, None, (242, 18)), (60000, 3000, None, (233, 11)),
                                         (60000, 3000, 0, (233, 11))):
        for source in (_grey16(paper, ink, border=border), _png(_grey16(paper, ink, border=border))):
            sent = load_pages(source)[0]
            assert (sent.getpixel((20, 20))[0], sent.getpixel((200, 60))[0]) == expected


def test_light_text_on_a_transparent_background_is_sent_as_before():
    # White or light text on transparent black (a caption, a logo for dark backgrounds): a white background would
    # hide it, so it is sent as 0.1.0 sent it, light on black, byte for byte.
    mask = Image.new("1", (40, 20), 0)
    ImageDraw.Draw(mask).rectangle((5, 5, 15, 15), fill=1)
    rgba = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    rgba.paste((255, 255, 255, 255), mask=mask)
    la = Image.new("LA", (40, 20), (0, 0))
    la.paste((200, 255), mask=mask)
    palette = Image.new("P", (40, 20), 0)                                # index 0: black and transparent
    palette.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    palette.paste(1, mask=mask)
    gif = io.BytesIO()
    palette.save(gif, "GIF", transparency=0)
    for source in (rgba, la, _png(rgba), _png(la), _png(palette, transparency=0), gif.getvalue()):
        before = source if isinstance(source, Image.Image) else Image.open(io.BytesIO(source))
        page = load_pages(source)[0]
        assert images.to_base64_png(page) == images.to_base64_png(before.convert("RGB"))
        assert page.getpixel((0, 0)) == (0, 0, 0) and page.getpixel((10, 10))[0] >= 200


def test_light_text_next_to_darker_content_on_a_transparent_background_is_sent_as_before():
    # A chart made for dark slides (white labels next to coloured bars) and outlined subtitles (white text, black
    # edge): on white their light text would vanish, so they are sent as 0.1.0 sent them, light on black.
    labels = Image.new("1", (400, 200), 0)
    ImageDraw.Draw(labels).rectangle((10, 10, 150, 30), fill=1)         # "Revenue 2568"
    for x in (20, 120, 220, 320):
        ImageDraw.Draw(labels).rectangle((x, 180, x + 50, 195), fill=1)  # "Q1 300"
    chart = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
    for x, height, colour in ((20, 130, (40, 90, 200)), (120, 100, (200, 60, 30)), (220, 140, (40, 120, 60)),
                              (320, 90, (120, 40, 140))):
        chart.paste(colour + (255,), (x, 170 - height, x + 70, 170))
    chart.paste((255, 255, 255, 255), mask=labels)
    subtitle = Image.new("RGBA", (400, 60), (0, 0, 0, 0))
    for y in (8, 24, 40):                                               # thin white strokes in a black outline:
        subtitle.paste((0, 0, 0, 255), (10, y, 390, y + 8))             # mostly dark, as text with a 3 px edge is
        subtitle.paste((255, 255, 255, 255), (13, y + 3, 387, y + 5))
    for source in (chart, _png(chart), subtitle, _png(subtitle)):
        before = source if isinstance(source, Image.Image) else Image.open(io.BytesIO(source))
        assert images.to_base64_png(load_pages(source)[0]) == images.to_base64_png(before.convert("RGB"))


def test_transparent_images_with_nothing_hidden_are_sent_as_before():
    # Transparent areas that are already white, and an alpha channel that hides every pixel (BGRX data read as
    # BGRA): the PNG sent is the one 0.1.0 sent, a palette image's transparent colour (tRNS) included.
    white = Image.new("P", (40, 20), 0)
    white.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
    ImageDraw.Draw(white).rectangle((5, 5, 15, 15), fill=1)
    gif = io.BytesIO()
    white.save(gif, "GIF", transparency=0)
    hidden = Image.new("RGBA", (40, 20), (200, 30, 30, 0))
    for source in (_png(white, transparency=0), gif.getvalue(), hidden):
        before = source if isinstance(source, Image.Image) else Image.open(io.BytesIO(source))
        assert images.to_base64_png(load_pages(source)[0]) == images.to_base64_png(before.convert("RGB"))


def test_multi_page_tiff_frames_are_pages(tmp_path):
    frames = [Image.new("RGB", (60 + i, 80), c) for i, c in enumerate(("white", "gray", "black"))]
    tiff = tmp_path / "scan.tif"
    frames[0].save(tiff, save_all=True, append_images=frames[1:])
    assert [p.size for p in load_pages(str(tiff))] == [(60, 80), (61, 80), (62, 80)]
    assert [p.size for p in load_pages(tiff.read_bytes(), [3, 1])] == [(62, 80), (60, 80)]
    with pytest.raises(ValueError, match="the TIFF has 3 page"):
        load_pages(str(tiff), [4])
    gif = tmp_path / "animated.gif"                                      # other multi-frame images: frame 1, as before
    frames[0].save(gif, save_all=True, append_images=frames[1:])
    assert [p.size for p in load_pages(str(gif))] == [(60, 80)]


def test_a_tiff_read_from_a_stream_or_pipe_that_cannot_seek(tmp_path):
    # A pipe, an HTTP response (urlopen) or a named pipe (also <(...) in a shell) can be read once. Image.open() reads
    # such a stream into memory; a TIFF's frames are looked through in a second copy of the file, which found the
    # stream already read (or, for a named pipe, waited for another writer).
    class Pipe(io.RawIOBase):
        def __init__(self, data):
            self._data = io.BytesIO(data)

        def readable(self):
            return True

        def readinto(self, buffer):
            return self._data.readinto(buffer)
    frames = [Image.new("L", (60 + i, 80), 60 * i) for i in range(3)]
    many, one, png = io.BytesIO(), io.BytesIO(), io.BytesIO()
    frames[0].save(many, "TIFF", save_all=True, append_images=frames[1:], compression="tiff_lzw")
    frames[0].save(one, "TIFF", compression="tiff_lzw")
    frames[2].save(png, "PNG")
    for data, sizes in ((many.getvalue(), [(60, 80), (61, 80), (62, 80)]), (one.getvalue(), [(60, 80)]),
                        (png.getvalue(), [(62, 80)])):
        with Image.open(io.BytesIO(data)) as im:
            first = im.convert("RGB").tobytes()                 # as 0.1.0 read it
        pages = load_pages(Pipe(data))
        assert [p.size for p in pages] == sizes and pages[0].tobytes() == first
    assert images._one_page(Pipe(many.getvalue()), 2, 100).size == (61, 80)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "scan.tif"
        os.mkfifo(fifo)
        threading.Thread(target=fifo.write_bytes, args=(many.getvalue(),), daemon=True).start()
        read = []
        reader = threading.Thread(target=lambda: read.append([p.size for p in load_pages(fifo)]), daemon=True)
        reader.start()
        reader.join(20)
        if reader.is_alive():                                   # it waits for a second writer: let it end
            fifo.write_bytes(b"")
        assert read == [[(60, 80), (61, 80), (62, 80)]]


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd")
def test_a_tiff_read_through_dev_fd_keeps_its_pages(tmp_path):
    # On macOS every file opened on /dev/stdin or /dev/fd/N shares one read position, so a TIFF opened twice at once
    # (once to look through its frames, once for the page) read the other copy's data: page 1 came out as page 3.
    frames = [Image.new("RGB", (300 + i, 400), (60 * i, 10 * i, 255 - 60 * i)) for i in range(3)]
    path = tmp_path / "scan.tif"
    frames[0].save(path, "TIFF", save_all=True, append_images=frames[1:])     # uncompressed, as scanners write
    expected = [page.tobytes() for page in load_pages(path)]
    fd = os.open(path, os.O_RDONLY)
    try:
        assert [page.tobytes() for page in load_pages(f"/dev/fd/{fd}")] == expected
    finally:
        os.close(fd)


def test_a_photoshop_file_is_read_as_its_merged_image(tmp_path):
    # Pillow numbers the frames of a PSD from 1, and opens it at its merged image: that is the page, as in 0.1.0.
    width, height = 120, 80
    plane = bytearray([255]) * (width * height)
    plane[30 * width:40 * width] = bytes(10 * width)                   # a black band
    psd = (b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, height, width, 8, 3)   # version 1, RGB, 8 bits, 3 channels
           + struct.pack(">IIIH", 0, 0, 0, 0) + bytes(plane) * 3)         # no extra sections; raw planes R, G, B
    path = tmp_path / "page.psd"
    path.write_bytes(psd)
    with Image.open(path) as im:
        assert im.format == "PSD"
        expected = im.convert("RGB").tobytes()
    for source in (psd, path, str(path), Image.open(path)):
        assert [p.tobytes() for p in load_pages(source)] == [expected]


def test_tiff_thumbnails_and_masks_are_not_pages():
    # NewSubfileType (tag 254) marks reduced-resolution copies (1, thumbnails and pyramid levels) and transparency
    # masks (4); pages are 0 or 2. Pillow 9 writes the first frame's tags on every frame, hence AppendingTiffWriter.
    def tiff(*frames):
        buf = io.BytesIO()
        with TiffImagePlugin.AppendingTiffWriter(buf) as out:
            for image, kind in frames:
                image.save(out, format="TIFF", tiffinfo={254: kind})
                out.newFrame()
        return buf.getvalue()
    page, back = Image.new("L", (120, 170), 230), Image.new("L", (100, 170), 90)
    scan = tiff((page, 0), (page.resize((12, 17)), 1))
    assert [p.size for p in load_pages(scan)] == [(120, 170)]                       # one page, as before
    with pytest.raises(ValueError, match="an image has one page"):
        load_pages(scan, [2])
    fax = tiff((page, 2), (page.resize((12, 17)), 1), (back, 2), (back.convert("1"), 4))
    assert [p.size for p in load_pages(fax)] == [(120, 170), (100, 170)]
    assert [p.size for p in load_pages(fax, [2])] == [(100, 170)]


def test_tiff_pages_do_not_take_the_palette_or_colour_profile_of_other_pages():
    # Pillow 12 and older keep the palette of a palette frame when they go to a frame without one, and every version
    # keeps an ICC profile: page 1 came out black or failed ("unrecognized image mode"), or carried page 2's profile.
    # Page 1 is now sent byte for byte as 0.1.0 sent it, and every page has the pixels and profile of its own frame.
    from PIL import ImageCms
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    def tiff(*frames):
        buf = io.BytesIO()
        with TiffImagePlugin.AppendingTiffWriter(buf) as out:
            for image, extra in frames:
                image.save(out, format="TIFF", **extra)
                out.newFrame()
        return buf.getvalue()

    def page_1(data):                                                   # as 0.1.0 read it
        with Image.open(io.BytesIO(data)) as im:
            return images.to_base64_png(im.convert("RGB"))
    text = Image.new("RGB", (120, 80), "white")
    ImageDraw.Draw(text).rectangle((10, 30, 110, 50), fill=(200, 30, 30))
    palette = text.convert("P", palette=Image.ADAPTIVE, colors=4)
    palette.putpalette(palette.getpalette()[:12][::-1] + [0] * 756)    # its first entries are not grey
    for first in (text, text.convert("L"), text.convert("1"), text.convert("CMYK")):
        frames = [first, palette, text.convert("L")]
        data = tiff(*((frame, {}) for frame in frames))
        expected = [frame.convert("RGB").tobytes() for frame in frames]
        pages = load_pages(data)
        assert [p.tobytes() for p in pages] == expected and images.to_base64_png(pages[0]) == page_1(data)
        assert [p.tobytes() for p in load_pages(data, [3, 1])] == [expected[2], expected[0]]
    data = tiff((palette, {}), (text, {}))                              # page 2 alone, after a palette page 1
    assert load_pages(data, [2])[0].tobytes() == text.tobytes()
    for profiles in ((None, srgb), (srgb, None)):
        data = tiff(*((text, {"icc_profile": p} if p else {}) for p in profiles))
        pages = load_pages(data)
        assert [p.info.get("icc_profile") for p in pages] == list(profiles)
        assert images.to_base64_png(pages[0]) == page_1(data)
        assert [p.info.get("icc_profile") for p in load_pages(data, [2, 1])] == [profiles[1], profiles[0]]


def test_exif_orientation_comes_from_the_exif_block_only():
    # Pillow also reads tiff:Orientation from XMP metadata when the EXIF block has none (for PNG on every version,
    # for JPEG on newer ones); photo viewers do not, and neither did 0.1.0. A malformed tag in an EXIF block without
    # an orientation does not give Pillow's warnings either.
    page = Image.new("RGB", (60, 80), "white")
    ImageDraw.Draw(page).rectangle((0, 0, 59, 15), fill="black")
    xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           '<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>')
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", xmp)
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=95)
    segment = b"http://ns.adobe.com/xap/1.0/\x00" + xmp.encode()
    jpeg = buf.getvalue()[:2] + b"\xff\xe1" + struct.pack(">H", len(segment) + 2) + segment + buf.getvalue()[2:]
    bad_make = b"Exif\x00\x00II*\x00" + struct.pack("<IHHHII", 8, 1, 0x010F, 2, 64, 4000) + struct.pack("<I", 0)
    for data in (_png(page, pnginfo=info), jpeg, _png(page, exif=bad_make)):
        with Image.open(io.BytesIO(data)) as im:
            expected = images.to_base64_png(im.convert("RGB"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert images.to_base64_png(load_pages(data)[0]) == expected
    big_endian = b"Exif\x00\x00MM\x00*" + struct.pack(">IHHHIHH", 8, 1, 0x0112, 3, 1, 6, 0) + struct.pack(">I", 0)
    assert load_pages(_png(page.rotate(90, expand=True), exif=big_endian))[0].size == (60, 80)


def test_a_tiff_page_over_pillows_size_limit_is_a_page(monkeypatch):
    # Pillow 9 checks the size of every frame as it moves to it, later versions as they read it: the page is counted
    # on every version, and reading it raises Pillow's DecompressionBombError.
    frames = [Image.new("L", (60, 80), 255), Image.new("L", (400, 300), 128), Image.new("L", (60, 80), 0)]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50_000)              # page 2 is over twice the limit
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # not "damaged after page 1"
        assert [p.getpixel((0, 0)) for p in load_pages(buf.getvalue(), [1, 3])] == [(255,) * 3, (0,) * 3]
        with pytest.raises(Image.DecompressionBombError):
            load_pages(buf.getvalue())


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")                     # Pillow's own, as it reads the damage
def test_a_tiff_whose_page_list_is_damaged_keeps_the_pages_before_the_damage():
    frames = [Image.new("L", (60, 80), c) for c in (255, 128, 0)]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    data = buf.getvalue()
    with pytest.warns(UserWarning, match="damaged after page 2"):            # frame 3's directory is past the end
        assert [p.getpixel((0, 0)) for p in load_pages(_next_ifd_past_end(data, 2))] == [(255,) * 3, (128,) * 3]
    cut = _next_ifd_past_end(data, 1)                                        # only frame 1 can be read
    with pytest.warns(UserWarning, match="damaged after page 1"):
        assert [p.getpixel((0, 0)) for p in load_pages(cut)] == [(255,) * 3]
    with pytest.warns(UserWarning, match="damaged after page 1"):
        with pytest.raises(ValueError, match="page 2 is out of range: the TIFF has 1 page"):
            load_pages(cut, [2])


def _next_ifd_past_end(tiff: bytes, k: int) -> bytes:
    """`tiff` with the "next directory" offset of frame k pointing past the end of the file."""
    data = bytearray(tiff)
    ifd = struct.unpack_from("<I", data, 4)[0]
    for _ in range(k):
        at = ifd + 2 + 12 * struct.unpack_from("<H", data, ifd)[0]
        ifd = struct.unpack_from("<I", data, at)[0]
    struct.pack_into("<I", data, at, len(data) + 5000)
    return bytes(data)


@pytest.mark.parametrize("spec, expected", [
    (None, None), (2, [2]), ("1-3,5", [1, 2, 3, 5]), ([4, 1], [4, 1]), (range(1, 4), [1, 2, 3]),
    ((p for p in (3, 1)), [3, 1]), ("1,1,2", [1, 2]), ([2, 2, 1], [2, 1]), (" 1 - 2 ,", [1, 2]), ([1.0, 3.0], [1, 3]),
    (Decimal("2"), [2]), ([Fraction(4, 2), Decimal("3")], [2, 3]), (range(5, 0, -2), [5, 3, 1]),
    ("2-3,1-4", [2, 3, 1, 4])])
def test_parse_pages(spec, expected):
    assert parse_pages(spec) == expected


@pytest.mark.parametrize("spec", ["3-1", "-2", "1-", "abc", "0", "", " ", ",", [], [0], [1.5], range(0, 3),
                                  range(4, 4), [Decimal("2.5")], [float("nan")], [float("inf")], b"1-3",
                                  bytearray(b"7")])
def test_parse_pages_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_pages(spec)


def test_a_huge_page_range_is_checked_against_the_page_count_first(pdf_bytes):
    # A typo such as "1-1000000000", or range(2, 10**7) for "page 2 to the end", is out of range as soon as the page
    # count is known: no list of all those page numbers is made. (10**7, so that a regression costs only 400 MB.)
    tracemalloc.start()
    try:
        for pages in (range(2, 10**7), images._selection(range(2, 10**7)), images._selection("2-9999999")):
            with pytest.raises(ValueError, match="page 4 is out of range: the PDF has 3 page"):
                load_pages(pdf_bytes, pages)
        with pytest.raises(ValueError, match="an image has one page"):
            load_pages(_png(Image.new("RGB", (4, 4))), images._selection("1-9999999"))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 10_000_000                    # 10 million page numbers in a list alone take 80 MB


def _png(im, **kwargs):
    buf = io.BytesIO()
    im.save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def _pdf(colors, size=(612, 792)):
    pages = [Image.new("RGB", size, c) for c in colors]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:], resolution=72)
    return buf.getvalue()
