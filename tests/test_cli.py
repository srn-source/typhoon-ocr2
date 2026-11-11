"""The typhoon-ocr2 command: cli.main() against fake servers (httpx.MockTransport), plus subprocess runs whose
stdout is a file in a Windows code page."""
import base64
import inspect
import io
import json
import os
import subprocess
import sys
import tracemalloc
import warnings
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

import typhoon_ocr2
from typhoon_ocr2 import TyphoonOCR, cli
from typhoon_ocr2 import images as im
from conftest import ELEMENTS, FakeServer, attach

DATA = Path(__file__).resolve().parent / "data"
BASE = ["--base-url", "http://x/v1", "--model", "m"]
REPLY = "```json\n" + json.dumps(ELEMENTS, ensure_ascii=False) + "\n```"
MARKDOWN = "Annual Report\n\nรายได้เพิ่มขึ้น"               # ELEMENTS without the header


def run(monkeypatch, capsys, args, server=None):
    """cli.main(args) with every TyphoonOCR talking to `server`; retries do not wait.
    Returns (exit code, stdout, stderr, [(keyword arguments, TyphoonOCR), ...])."""
    server = server or FakeServer()
    made = []

    def make(**kwargs):
        ocr = attach(TyphoonOCR(**kwargs), server)
        ocr._sleep = lambda seconds: None
        made.append((kwargs, ocr))
        return ocr
    monkeypatch.setattr(cli, "TyphoonOCR", make)
    code = cli.main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err, made


def serve(reply):
    """A server that answers reply(number of the page it was sent): the model's output, or an httpx.Response."""
    def handler(request):
        url = json.loads(request.content)["messages"][0]["content"][0]["image_url"]["url"]
        answer = reply(round(sent(url).getpixel((5, 5))[0] / 20))
        return answer if isinstance(answer, httpx.Response) else httpx.Response(
            200, json={"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]})
    return SimpleNamespace(handler=handler)


def sent(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def grey_pdf(path, n):
    """An n-page PDF whose page k is a flat grey of level 20 * k, so the server can tell the pages apart."""
    pages = [Image.new("RGB", (150, 200), (20 * k,) * 3) for k in range(1, n + 1)]
    pages[0].save(path, format="PDF", save_all=True, append_images=pages[1:], resolution=72)
    return path


@pytest.fixture
def png(tmp_path, page_image):
    page_image.save(tmp_path / "page.png")
    return tmp_path / "page.png"


def test_markdown_is_the_default_output(monkeypatch, capsys, png):
    assert run(monkeypatch, capsys, [png, *BASE])[:3] == (0, MARKDOWN + "\n", "")
    assert run(monkeypatch, capsys, [png, *BASE, "--exclude"])[1] == "Page 1\n\n" + MARKDOWN + "\n"
    assert run(monkeypatch, capsys, [png, *BASE, "--exclude", "title", "header"])[1] == "รายได้เพิ่มขึ้น\n"
    assert run(monkeypatch, capsys, [png, *BASE, "--pages", ""])[:3] == (0, MARKDOWN + "\n", "")   # an image's page


def test_exclude_names_are_checked_before_anything_is_sent(monkeypatch, capsys, png, fake):
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, capsys, [png, *BASE, "--exclude", "footers"], fake)
    assert e.value.code == 2 and "invalid choice: 'footers'" in capsys.readouterr().err
    assert fake.requests == []


def test_pages_whose_output_is_not_layout_json(monkeypatch, capsys, tmp_path):
    pdf = grey_pdf(tmp_path / "doc.pdf", 2)
    server = serve(lambda page: REPLY if page == 1 else "Sorry, I cannot read this page.")
    code, out, err, _ = run(monkeypatch, capsys, [pdf, *BASE, "--format", "json"], server)
    page1 = ('  {\n    "page": 1, "width": 1536, "height": 2048,\n    "truncated": false,\n    "elements": [\n'
             + ",\n".join("      " + json.dumps(e, ensure_ascii=False) for e in ELEMENTS)
             + f'\n    ],\n    "markdown": {json.dumps(MARKDOWN, ensure_ascii=False)}\n  }}')
    assert out.startswith("[\n" + page1 + ",\n")              # a page that parsed: the 0.1.0 format, byte for byte
    assert json.loads(out)[1] == {"page": 2, "width": 1536, "height": 2048, "truncated": False, "elements": [],
                                  "markdown": "", "raw": "Sorry, I cannot read this page."}
    assert code == 0 and err == "warning: page 2: the model output is not layout JSON; nothing was extracted\n"
    code, out, err, _ = run(monkeypatch, capsys, [pdf, *BASE, "--pages", "2"], server)
    assert (code, out) == (0, "\n") and "page 2: the model output is not layout JSON" in err


def test_a_failed_page_keeps_the_pages_already_parsed(monkeypatch, capsys, tmp_path):
    pdf = grey_pdf(tmp_path / "doc.pdf", 4)
    server = serve(lambda page: httpx.Response(500, text="page 3 broke the server") if page == 3 else REPLY)
    code, out, err, made = run(monkeypatch, capsys, [pdf, *BASE, "--max-workers", "1"], server)
    assert code == 1 and out == MARKDOWN + "\n\n" + MARKDOWN + "\n"              # pages 1 and 2
    assert err.splitlines() == [
        "error: page 3 failed: http://x/v1/chat/completions returned 500 after 4 attempts: page 3 broke the server",
        "error: page(s) not sent: 4",
        "error: 2 page(s) were parsed and written out; to parse the missing ones, run again with --pages 3-4"]
    assert made[0][1]._client.is_closed
    code, out, err, _ = run(monkeypatch, capsys, [pdf, *BASE, "--max-workers", "1", "--format", "json"], server)
    assert code == 1 and [p["page"] for p in json.loads(out)] == [1, 2]


def test_a_page_that_cannot_be_read_from_the_file_is_left_out_of_the_hint(monkeypatch, capsys, tmp_path):
    # Parsing it again would fail again: the hint names the pages that can still be parsed.
    real = im._render_page

    def render(doc, n, long_side):
        if n == 2:
            raise im._pdfium().PdfiumError("Failed to load page.")
        return real(doc, n, long_side)
    monkeypatch.setattr(im, "_render_page", render)
    pdf = grey_pdf(tmp_path / "doc.pdf", 4)
    code, out, err, _ = run(monkeypatch, capsys, [pdf, *BASE, "--max-workers", "1"], serve(lambda page: REPLY))
    assert code == 1 and out == MARKDOWN + "\n"                                  # page 1
    assert err.splitlines() == [
        f"error: page 2 failed: cannot render page 2 of the PDF {str(pdf)!r}: Failed to load page",
        "error: page(s) not sent: 3-4",
        "error: 1 page(s) were parsed and written out; to parse the missing ones, run again with --pages 3-4; "
        "page(s) 2 cannot be read from the file"]
    frames = [Image.new("L", (150, 200), 20 * n) for n in (1, 2)]               # the last frame of a TIFF
    frames[0].save(tmp_path / "scan.tif", save_all=True, append_images=frames[1:])
    data = (tmp_path / "scan.tif").read_bytes()
    (tmp_path / "scan.tif").write_bytes(data[:len(data) - 2000])                 # frame 2's image data is cut off
    code, out, err, _ = run(monkeypatch, capsys, [tmp_path / "scan.tif", *BASE, "--max-workers", "1"],
                            serve(lambda page: REPLY))
    assert code == 1 and out == MARKDOWN + "\n" and err.splitlines()[-1] == (
        "error: 1 page(s) were parsed and written out; page(s) 2 cannot be read from the file")


def test_an_image_over_pillows_size_limit_is_reported_without_a_traceback(monkeypatch, capsys, png):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)       # page.png: 240,000 pixels, over twice the limit
    code, out, err, _ = run(monkeypatch, capsys, [png, *BASE])
    assert (code, out) == (1, "") and err.count("\n") == 1
    assert err.startswith("error: Image size (240000 pixels) exceeds limit of 2000 pixels")
    assert err.endswith("; scale the image down or cut it into parts first\n")


def test_ask_and_extract_read_the_first_page_of_pages(monkeypatch, capsys, tmp_path):
    asked = []
    server = serve(lambda page: asked.append(page) or '{"Total": "1,250"}')
    pdf = grey_pdf(tmp_path / "doc.pdf", 3)
    code, out, err, _ = run(monkeypatch, capsys, [pdf, *BASE, "--pages", "2-3", "--ask", "Total?"], server)
    assert (code, out, asked) == (0, '{"Total": "1,250"}\n', [2])
    assert err == "warning: --ask and --extract read one page: page 2\n"
    code, out, err, _ = run(monkeypatch, capsys, [pdf, *BASE, "--pages", "3", "--extract", "Total"], server)
    assert (code, json.loads(out), err, asked) == (0, {"Total": "1,250"}, "", [2, 3])
    tracemalloc.start()
    try:                                    # a long range is not expanded to find its first page
        code, out, err, _ = run(monkeypatch, capsys, [pdf, *BASE, "--pages", "1-9999999", "--ask", "Total?"], server)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert (code, asked, err) == (0, [2, 3, 1], "warning: --ask and --extract read one page: page 1\n")
    assert peak < 50_000_000                # 10 million page numbers in a list alone take 80 MB


def test_ask_reads_a_frame_of_a_multi_page_tiff(monkeypatch, capsys, tmp_path, fake):
    frames = [Image.new("L", (100 + 10 * i, 150), 255) for i in range(3)]
    frames[0].save(tmp_path / "fax.tif", save_all=True, append_images=frames[1:])
    args = [tmp_path / "fax.tif", *BASE, "--no-upscale", "--pages", "2", "--ask", "Total?"]
    assert run(monkeypatch, capsys, args, fake)[0] == 0
    assert sent(fake.requests[0][1]["messages"][0]["content"][0]["image_url"]["url"]).size == (110, 150)


@pytest.mark.parametrize("source, args, answer, message", [
    ("doc.pdf", ["--pages", "3-1"], None, "error: page range '3-1' is reversed: write it as '1-3'"),
    ("doc.pdf", ["--pages", "abc", "--ask", "Total?"], None, "error: 'abc' is not a page number"),
    ("doc.pdf", ["--pages", "", "--ask", "Total?"], None, "error: no page selected: pages=''"),
    ("doc.pdf", ["--pages", "9"], None, "error: page 9 is out of range: the PDF has 3 page(s)"),
    ("doc.pdf", ["--base-url", "http://localhost:99999/v1", "--model", "m"], None, "error: Port out of range"),
    ("page.png", ["--ask", "Total?", "--base-url", "http://localhost:abc/v1", "--model", "m", "--backend", "openai"],
     None, "error: cannot reach http://localhost:abc/v1/chat/completions: Invalid port"),
    ("page.png", BASE, b'{"choices": []}', "error: page 1 failed: unexpected answer from http://x/v1/chat/completions"),
    ("page.png", ["--ask", "Total?", *BASE], b"<html>Sign in</html>",
     "error: http://x/v1/chat/completions did not answer with JSON"),
    # tests/data/locked.pdf needs the password "1234" (AES-256); made with reportlab 5.1 and pypdf 7.0's encrypt().
    ("locked.pdf", BASE, None, "error: cannot open the PDF"),
    # A page read from the file that fails later (1.5's resize to 1800 px makes it 0 px wide): not "cannot be read".
    ("strip.png", [*BASE, "--version", "v1.5"], None, "error: page 1 failed: height and width must be > 0"),
])
def test_errors_are_reported_without_a_traceback(monkeypatch, capsys, tmp_path, page_image, source, args, answer,
                                                 message):
    path = DATA / source if source == "locked.pdf" else tmp_path / source
    if source == "page.png":
        page_image.save(path)
    elif source == "strip.png":
        Image.new("RGB", (1, 20000), "white").save(path)
    elif source == "doc.pdf":
        grey_pdf(path, 3)
    server = SimpleNamespace(handler=lambda request: httpx.Response(200, content=answer)) if answer else None
    code, out, err, made = run(monkeypatch, capsys, [path, *args], server)
    assert (code, out) == (1, "") and err.startswith(message) and err.count("\n") == 1
    assert all(ocr._client.is_closed for _, ocr in made)


def test_options_reach_typhoon_ocr(monkeypatch, capsys, png, fake):
    _, _, _, [(kwargs, _)] = run(monkeypatch, capsys, [png, *BASE], fake)
    defaults = inspect.signature(TyphoonOCR).parameters
    for name in ("backend", "version", "upscale", "max_workers", "timeout", "figure_language"):
        assert kwargs[name] == defaults[name].default             # the CLI defaults are the library defaults
    args = ["--version", "v1.5", "--max-workers", "2", "--timeout", "30", "--figure-language", "English"]
    _, _, _, [(kwargs, ocr)] = run(monkeypatch, capsys, [png, *BASE, *args], fake)
    assert (kwargs["max_workers"], kwargs["timeout"], kwargs["figure_language"]) == (2, 30.0, "English")
    assert ocr.max_workers == 2 and "Describe in English." in fake.requests[-1][1]["messages"][0]["content"][1]["text"]
    for bad in (["--max-workers", "0"], ["--timeout", "-1"], ["--timeout", "soon"], ["--figure-language", "Lao"]):
        with pytest.raises(SystemExit) as e:
            run(monkeypatch, capsys, [png, *BASE, *bad], fake)
        assert e.value.code == 2


def test_library_warnings_are_printed_like_the_commands_own(monkeypatch, capsys, png, fake):
    code, out, err, _ = run(monkeypatch, capsys, [png, "--base-url", "http://10.0.0.5:8000/v1", "--model", "m",
                                                  "--api-key", "k"], fake)
    assert (code, out) == (0, MARKDOWN + "\n") and err.count("\n") == 1     # one line, without Python's source line
    assert err.startswith("warning: the API key will be sent unencrypted to 10.0.0.5 (plain http)")
    monkeypatch.setattr(cli, "TyphoonOCR", lambda **kwargs: warnings.warn("old option", DeprecationWarning) or
                        attach(TyphoonOCR(**kwargs), fake))
    with pytest.warns(DeprecationWarning, match="old option"):          # other warnings: as before (here, to pytest)
        assert cli.main([str(png), *BASE]) == 0
    assert capsys.readouterr().err == ""


def test_output_that_the_stdout_encoding_cannot_hold_is_written_as_utf8(tmp_path, png):
    # On Windows before Python 3.15, a redirected stdout uses the ANSI code page: cp1252 (with Windows' default
    # error handler) has no Thai letters, so the output is UTF-8; cp874 has them, so it is cp874, as before.
    driver = ("import sys; from typhoon_ocr2 import cli; from conftest import FakeServer, attach; "
              "real = cli.TyphoonOCR; cli.TyphoonOCR = lambda **kw: attach(real(**kw), FakeServer()); "
              "sys.exit(cli.main(sys.argv[1:]))")
    paths = [str(Path(typhoon_ocr2.__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, paths + [os.environ.get("PYTHONPATH")]))}
    for encoding, written in (("cp1252", "utf-8"), ("cp874", "cp874")):
        env["PYTHONIOENCODING"] = encoding + ":surrogateescape"
        with open(tmp_path / "out.md", "wb") as stdout:
            p = subprocess.run([sys.executable, "-c", driver, str(png), *BASE], stdout=stdout, stderr=subprocess.PIPE,
                               env=env, timeout=120)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        assert (tmp_path / "out.md").read_bytes().splitlines() == MARKDOWN.encode(written).splitlines()
    p = subprocess.run([sys.executable, "-c", driver, str(tmp_path / "ไม่มี.png"), *BASE], capture_output=True,
                       env={**env, "PYTHONIOENCODING": "cp1252:surrogateescape"}, timeout=120)
    assert p.returncode == 1 and p.stderr.startswith(b"error: ") and b"Traceback" not in p.stderr
