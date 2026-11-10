import _thread
import base64
import io
import json
import pickle
import struct
import threading
import time
import tracemalloc
import warnings
import weakref
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import format_datetime

import httpx
import pytest
from PIL import Image

import typhoon_ocr2
from typhoon_ocr2 import EXTRACT_ALL_PROMPT, PARSE_PROMPT, TyphoonOCR, TyphoonOCRError, images
from typhoon_ocr2.client import _backoff, _detect_backend, _version_of
from conftest import ELEMENTS, FakeServer, attach

OK = {"choices": [{"message": {"content": json.dumps(ELEMENTS)}, "finish_reason": "stop"}]}


@pytest.fixture(autouse=True)
def no_env_config(monkeypatch):
    for k in ("TYPHOON_OCR_MODEL", "TYPHOON_OCR_BASE_URL", "TYPHOON_API_KEY", "TYPHOON_OCR_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def client(handler, sleeps=None, **kwargs):
    """A TyphoonOCR whose requests go to `handler`; the waits between retries are recorded in `sleeps`."""
    ocr = TyphoonOCR(**{"base_url": "http://x/v1", "model": "m", **kwargs})
    ocr._client = httpx.Client(transport=httpx.MockTransport(handler))
    ocr._sleep = sleeps.append if sleeps is not None else (lambda seconds: None)
    return ocr


def serving(models, reply="```json\n" + json.dumps(ELEMENTS) + "\n```"):
    """A server that lists `models` at /models and answers every chat request with `reply`."""
    bodies = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]})
    return handler, bodies


def sent_image(body):
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def grey_pdf(n, size=(150, 200)):
    """An n-page PDF whose page k is a flat grey of level 20 * k, so a server can tell the pages apart."""
    pages = [Image.new("RGB", size, (20 * k,) * 3) for k in range(1, n + 1)]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:], resolution=72)
    return buf.getvalue()


def page_of(request):
    return round(sent_image(json.loads(request.content)).getpixel((5, 5))[0] / 20)


def test_backend_detection():
    assert _detect_backend("http://localhost:11434") == "ollama"
    assert _detect_backend("http://localhost:11434/v1") == "openai"
    assert _detect_backend("http://localhost:8000/v1") == "openai"


def test_parse_openai_request_and_result(fake, page_image):
    ocr = attach(TyphoonOCR(base_url="http://x/v1"), fake)
    [r] = ocr.parse(page_image)
    path, body = fake.requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "typhoon-ai/typhoon-ocr2-2b"           # first served model
    assert body["temperature"] == 0.0 and body["max_tokens"] == 16384
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[1]["text"] == PARSE_PROMPT
    assert r.elements == ELEMENTS and r.markdown == "Annual Report\n\nรายได้เพิ่มขึ้น"
    assert r.image_size == (400, 600) and not r.truncated and r.parse_error is None


def test_small_image_is_enlarged_before_sending(fake, page_image):
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), fake)
    seen = []
    orig = ocr._generate
    ocr._generate = lambda image, prompt: (seen.append(image.size), orig(image, prompt))[1]
    ocr.parse(page_image)
    assert seen == [(1067, 1600)]
    ocr2 = attach(TyphoonOCR(base_url="http://x/v1", model="m", upscale=False), fake)
    assert ocr2._prepare(page_image).size == (400, 600)


def test_parse_ollama_request(page_image):
    srv = FakeServer()
    ocr = attach(TyphoonOCR(base_url="http://localhost:11434"), srv)
    [r] = ocr.parse(page_image)
    path, body = srv.requests[0]
    assert path == "/api/chat" and body["think"] is False
    assert body["model"] == "SuraponOCR/typhoon-ocr2-2b"
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] == 16384
    assert r.elements == ELEMENTS


def test_parse_pdf_pages_in_order(pdf_bytes):
    srv = FakeServer()
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m", max_workers=3), srv)
    res = ocr.parse(pdf_bytes, pages="1-3")
    assert [r.page for r in res] == [1, 2, 3] and len(srv.requests) == 3


def test_truncated_output(page_image):
    raw = "```json\n" + json.dumps(ELEMENTS)[:-1] + ', {"bbox": [1, 2'
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), FakeServer(reply=raw, finish="length"))
    [r] = ocr.parse(page_image)
    assert r.truncated and r.elements == ELEMENTS


def test_unparseable_output_gives_empty_elements(page_image):
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), FakeServer(reply="sorry"))
    [r] = ocr.parse(page_image)
    assert r.elements == [] and r.markdown == "" and r.raw == "sorry"
    assert "not a JSON list of elements" in r.parse_error


def test_v15_returns_markdown_with_v15_settings(page_image):
    srv = FakeServer(reply="# Title\n\ntext")
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="typhoon-ocr", version="v1.5"), srv)
    [r] = ocr.parse(page_image)
    body = srv.requests[0][1]
    assert r.markdown == "# Title\n\ntext" and r.elements == [] and r.parse_error is None
    assert body["temperature"] == 0.1 and body["repetition_penalty"] == 1.1
    assert "Describe in Thai." in body["messages"][0]["content"][1]["text"]


def test_ask_and_extract(page_image):
    srv = FakeServer(reply='```json\n{"Total": "355.00"}\n```')
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), srv)
    assert ocr.extract(page_image) == {"Total": "355.00"}
    assert "open-schema" in srv.requests[0][1]["messages"][0]["content"][1]["text"]
    assert ocr.ask(page_image, "Total?").startswith("```json")


def test_extract_one_field_given_as_a_string(page_image):
    srv = FakeServer(reply='{"Total": "355.00"}')
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), srv)
    assert ocr.extract(page_image, "Total") == {"Total": "355.00"}
    assert srv.requests[0][1]["messages"][0]["content"][1]["text"].endswith("Return only JSON.\nTotal")
    for blank in ("", " "):                                 # no field named: every field, as before
        ocr.extract(page_image, blank)
        assert srv.requests[-1][1]["messages"][0]["content"][1]["text"] == EXTRACT_ALL_PROMPT


def test_extract_invalid_json_raises(page_image):
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), FakeServer(reply="no idea"))
    with pytest.raises(TyphoonOCRError):
        ocr.extract(page_image)


def test_server_error_is_reported(page_image):
    ocr = client(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(TyphoonOCRError, match="500 after 4 attempts: boom") as e:
        ocr.parse(page_image)
    assert str(e.value).endswith("0 of 1 page(s) were parsed")            # nothing in .results to point at


def test_a_damaged_image_raises_oserror_before_anything_is_sent(fake, page_image, tmp_path):
    buf = io.BytesIO()
    page_image.save(buf, format="PNG")
    good = buf.getvalue()
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), fake)
    with pytest.raises(OSError, match="truncated"):
        ocr.parse(good[:len(good) // 2])
    # Damaged inside, not cut short: Pillow raises once and keeps what it decoded, which a second load() returns as a
    # partly black page without an error. A broken chunk length gives Pillow's SyntaxError, raised as OSError.
    at = good.index(b"IDAT")
    length = struct.unpack(">I", good[at - 4:at])[0]
    middle = at + 4 + length // 2
    flipped = good[:middle] + bytes(b ^ 0xFF for b in good[middle:middle + 16]) + good[middle + 16:]
    short = good[:at - 4] + struct.pack(">I", length - 16) + good[at:]
    for data, match in ((flipped, "data stream"), (short, "broken PNG file")):
        path = tmp_path / f"damaged_{len(match)}.png"
        path.write_bytes(data)
        for source in (data, path, Image.open(path)):
            with pytest.raises(OSError, match=match):
                ocr.parse(source)
            with pytest.raises(OSError, match=match):
                ocr.ask(Image.open(path) if isinstance(source, Image.Image) else source, "Total?")
    assert fake.requests == []


def damaged_tiff(k):
    """A 3-page TIFF (page n a flat grey of level 20 * n) whose frame k points to image data past the end of the
    file."""
    frames = [Image.new("L", (60, 80), 20 * n) for n in (1, 2, 3)]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    data = bytearray(buf.getvalue())
    ifd = struct.unpack_from("<I", data, 4)[0]
    for _ in range(k - 1):
        ifd = struct.unpack_from("<I", data, ifd + 2 + 12 * struct.unpack_from("<H", data, ifd)[0])[0]
    for entry in range(ifd + 2, ifd + 2 + 12 * struct.unpack_from("<H", data, ifd)[0], 12):
        tag, kind = struct.unpack_from("<HH", data, entry)
        if tag == 273:                                      # StripOffsets
            struct.pack_into("<I" if kind == 4 else "<H", data, entry + 8, 0xFFFF if kind == 3 else 1 << 30)
    return bytes(data)


def test_a_first_page_that_cannot_be_read_raises_before_anything_is_sent(monkeypatch, pdf_bytes):
    # As for an image: the first page of a PDF or multi-page TIFF is read before the model is looked up.
    sent = []

    def handler(request):
        sent.append(request.method)
        return httpx.Response(200, json={"data": [{"id": "typhoon-ai/typhoon-ocr2-2b"}]} if request.method == "GET"
                              else OK)
    with pytest.raises(OSError, match="truncated"):
        client(handler, model=None).parse(damaged_tiff(1))
    real = images._render_page

    def render(doc, n, long_side):
        if n == 2:
            raise images._pdfium().PdfiumError("Failed to load page.")
        return real(doc, n, long_side)
    monkeypatch.setattr(images, "_render_page", render)
    with pytest.raises(ValueError, match="cannot render page 2 of the PDF: Failed to load page"):
        client(handler, model=None).parse(pdf_bytes, pages="2-3")
    with pytest.raises(ValueError, match="cannot render page 2"):
        client(handler, model=None).ask(pdf_bytes, "Total?", page=2)
    assert sent == []
    assert [r.page for r in client(handler, model=None).parse(pdf_bytes, pages=[3, 1])] == [3, 1]


def test_a_huge_page_selection_is_checked_against_the_page_count_first(fake, pdf_bytes):
    # A typo such as --pages 1-1000000000, or range(3, 10**7) for "page 3 to the end", is out of range as soon as the
    # page count is known, without a list of all those page numbers. (10**7: a regression costs only 400 MB.)
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), fake)
    tracemalloc.start()
    try:
        for pages in (range(3, 10**7), "2-9999999", [2, 1, 7]):
            with pytest.raises(ValueError, match=r"page (4|7) is out of range: the PDF has 3 page"):
                ocr.parse(pdf_bytes, pages=pages)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 10_000_000 and fake.requests == []


def test_retries_rate_limits_and_server_errors(page_image):
    answers = [httpx.Response(429, headers={"Retry-After": "2"}, text="slow down"), httpx.Response(503, text="busy"),
               httpx.Response(200, json=OK)]
    sleeps = []
    [r] = client(lambda request: answers.pop(0), sleeps).parse(page_image)
    assert r.elements == ELEMENTS and not answers
    assert sleeps[0] == 2.0 and 0 <= sleeps[1] <= 2.0       # Retry-After, then backoff with jitter (0-2 s)


def test_retry_after_header():
    from typhoon_ocr2.client import _retry_after
    soon = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    assert _retry_after("5") == 5 and _retry_after("0") == 0 and _retry_after("3600") == 60
    assert 20 <= _retry_after(soon) <= 30
    assert _retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0                # a date in the past: retry now
    assert _retry_after("soon") is None and _retry_after("-1") is None and _retry_after(None) is None


def test_retry_at_the_time_a_typhoon_api_429_gives(page_image):
    # The Typhoon API's documented 429 has no Retry-After header, only the time in the message.
    from typhoon_ocr2.client import _retry_at
    soon = (datetime.now(timezone.utc) + timedelta(seconds=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
    limited = {"error": {"message": f"Rate limit exceeded. Please retry after {soon}", "type": "rate_limit_error",
                         "param": None, "code": "rate_limit_exceeded"}}
    answers = [httpx.Response(429, json=limited), httpx.Response(200, json=OK)]
    sleeps = []
    [r] = client(lambda request: answers.pop(0), sleeps).parse(page_image)
    assert r.elements == ELEMENTS and 30 < sleeps[0] <= 40
    later = datetime.now(timezone(timedelta(hours=7))) + timedelta(seconds=20)       # e.g. 2026-11-10T04:12:34.5+07:00
    assert 10 < _retry_at(f"retry after {later.isoformat()}") <= 20
    assert _retry_at("Please retry after 2999-01-01T00:00:00Z") == 60
    assert _retry_at("Please retry after 2020-01-01T00:00:00Z") is None      # passed (or a clock ahead): back off
    assert _retry_at("slow down") is None and _retry_at("") is None


def test_gives_up_after_max_retries(page_image):
    calls = []
    ocr = client(lambda r: calls.append(1) or httpx.Response(503, text="overloaded"), max_retries=2)
    with pytest.raises(TyphoonOCRError, match="503 after 3 attempts: overloaded"):
        ocr.parse(page_image)
    assert len(calls) == 3
    calls.clear()
    with pytest.raises(TyphoonOCRError, match="400: bad request"):          # not retried
        client(lambda r: calls.append(1) or httpx.Response(400, text="bad request")).parse(page_image)
    assert len(calls) == 1


def test_dropped_connections_are_retried_but_timeouts_are_not(page_image):
    calls = []

    def flaky(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        if len(calls) == 2:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        if len(calls) == 3:
            raise httpx.WriteError("broken pipe", request=request)         # dropped while the page was sent
        return httpx.Response(200, json=OK)
    assert client(flaky).parse(page_image)[0].elements == ELEMENTS and len(calls) == 4
    calls.clear()

    def slow(request):
        calls.append(1)
        raise httpx.ReadTimeout("timed out", request=request)
    with pytest.raises(TyphoonOCRError, match="no answer .* in time"):
        client(slow).parse(page_image)
    assert len(calls) == 1                                  # the server may still be working on the page
    calls.clear()

    def unreachable(request):
        calls.append(1)
        raise httpx.ConnectTimeout("timed out", request=request)
    with pytest.raises(TyphoonOCRError, match="cannot reach http://x/v1/chat/completions: timed out"):
        client(unreachable).parse(page_image)
    assert len(calls) == 1                                  # it has already waited `timeout` seconds


def test_backoff_stays_under_30_seconds():
    assert 0 <= _backoff(0) <= 1 and all(0 <= _backoff(a) <= 30 for a in (5, 1023, 1024, 10**6))   # max_retries=10**6


def test_an_api_key_that_cannot_be_sent_is_not_repeated_in_the_error(page_image):
    def reject(request):                    # what h11 raises for a header value with a line break in it
        raise httpx.LocalProtocolError(f"Illegal header value {request.headers['authorization'].encode()!r}")
    ocr = TyphoonOCR(base_url="https://x/v1", model="m", api_key="sk-SECRET\nsk-OTHER")
    ocr._client = httpx.Client(transport=httpx.MockTransport(reject), headers=ocr._client.headers)
    with pytest.raises(TyphoonOCRError, match="a header is not valid; check the API key") as e:
        ocr.ask(page_image, "Total?")
    assert "SECRET" not in str(e.value)
    ocr = TyphoonOCR(base_url="https://x/v1", api_key="sk-SECRET \n")     # read from a file: the line break goes
    assert ocr.api_key == "sk-SECRET" and ocr._client.headers["Authorization"] == "Bearer sk-SECRET"


def test_a_server_that_cannot_be_reached_is_not_reported_as_a_missing_model(page_image):
    def refused(request):
        raise httpx.ConnectError("connection refused", request=request)
    with pytest.raises(TyphoonOCRError) as e:
        client(refused, model=None).parse(page_image)
    assert str(e.value) == "cannot reach http://x/v1/models: connection refused"
    with pytest.raises(TyphoonOCRError, match=r"pass model=\.\.\.: .*/models returned 404"):   # no model list
        client(lambda request: httpx.Response(404, text="not found"), model=None).parse(page_image)


def test_a_failing_model_lookup_is_not_repeated_by_each_waiting_thread(page_image):
    # Threads sharing one TyphoonOCR look the model up at the same time; when the lookup fails, each fails after one
    # lookup, not one after the other (the last of N threads after N lookups, each up to `timeout`).
    met = threading.Barrier(4, timeout=10)

    def handler(request):
        met.wait()                          # all 4 lookups are under way at once
        raise httpx.ReadTimeout("timed out", request=request)
    ocr = client(handler, model=None)
    errors = []

    def ask():
        try:
            ocr.ask(page_image, "Total?")
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=ask) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(errors) == 4 and all("no answer from http://x/v1/models in time" in str(e) for e in errors)


def test_a_model_list_without_names(page_image):
    listing = {"data": [{"id": 7}, {"id": "typhoon-ai/typhoon-ocr2-2b"}]}
    assert client(lambda r: httpx.Response(200, json=listing), model=None).model == "typhoon-ai/typhoon-ocr2-2b"
    for listing in ({"data": [{"id": 7}]}, {"data": [{"id": None}]}, {"data": []}, {"data": "m"}):
        with pytest.raises(TyphoonOCRError, match=r"pass model=\.\.\.: .*\(the list has no model name\)"):
            client(lambda r, listing=listing: httpx.Response(200, json=listing), model=None).parse(page_image)


@pytest.mark.parametrize("response, match", [
    (httpx.Response(302, headers={"Location": "https://x/v1/chat/completions"}),
     "redirected (302) to https://x/v1/chat/completions; use that address as base_url, i.e. base_url='https://x/v1'"),
    (httpx.Response(305, headers={"Location": "http://[::1"}), "answered 305 with an invalid Location 'http://[::1'"),
    (httpx.Response(200, text="<html>proxy login</html>"), "did not answer with JSON"),
    (httpx.Response(200, text="[" * 100_000), "did not answer with JSON"),          # nested too deeply to read
    (httpx.Response(200, json={"choices": []}), "unexpected answer"),
    (httpx.Response(200, json={"error": {"message": "upstream failed"}}), "upstream failed")])
def test_unusable_answers_raise_typhoon_ocr_error(page_image, response, match):
    with pytest.raises(TyphoonOCRError) as e:
        client(lambda r: response).parse(page_image)
    assert match in str(e.value)


def test_a_redirect_to_an_invalid_location_is_not_retried(page_image):
    # httpx reads the Location of a 301, 302, 303, 307 or 308 also when it does not follow it, and raises
    # RemoteProtocolError, the error of a dropped connection, when it is not a URL.
    calls = []
    ocr = client(lambda request: calls.append(1) or httpx.Response(302, headers={"Location": "http://[::1"}))
    with pytest.raises(TyphoonOCRError, match=r"completions answered a redirect with an invalid Location \(Invalid URL"
                                              r" in location header: .*\): check base_url"):
        ocr.ask(page_image, "Total?")
    assert len(calls) == 1


def test_ollama_error_answer_and_empty_content(page_image):
    with pytest.raises(TyphoonOCRError, match="model not loaded"):
        client(lambda r: httpx.Response(200, json={"error": "model not loaded"}),
               base_url="http://localhost:11434").parse(page_image)
    ocr = client(lambda r: httpx.Response(200, json={"message": {"content": None}, "done_reason": "stop"}),
                 base_url="http://localhost:11434")
    assert ocr.ask(page_image, "Total?") == ""


def test_bad_version():
    with pytest.raises(ValueError):
        TyphoonOCR(version="v3")


def test_typhoon_api_default_models(page_image):
    srv = FakeServer()
    ocr = attach(TyphoonOCR(base_url="https://api.opentyphoon.ai/v1", api_key="k"), srv)
    assert ocr.model == "typhoon-ocr-v2"
    ocr.parse(page_image)
    assert srv.requests[0][0] == "/v1/chat/completions" and srv.requests[0][1]["model"] == "typhoon-ocr-v2"
    assert TyphoonOCR(base_url="https://api.opentyphoon.ai/v1", version="v1.5").model == "typhoon-ocr-v1.5"


def test_models_list_as_bare_list():
    ocr = TyphoonOCR(base_url="http://x/v1")
    ocr._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[{"id": "m1"}, {"id": "m2"}])))
    with pytest.warns(UserWarning, match="pass model="):
        assert ocr.model == "m1"


def test_a_typhoon_model_is_picked_from_a_long_list():
    handler, _ = serving(["mlx-community/Qwen2.5-VL-7B-Instruct-4bit", "typhoon-ai/typhoon-ocr2-2b-mlx-8bit",
                          "typhoon-ai/typhoon-ocr1.5-2b"])
    assert client(handler, model=None).model == "typhoon-ai/typhoon-ocr2-2b-mlx-8bit"
    assert client(handler, model=None, version="v1.5").model == "typhoon-ai/typhoon-ocr1.5-2b"
    with pytest.warns(UserWarning, match=r"none is named like Typhoon OCR.*'qwen3-vl:8b', 'llava:7b'"):
        assert client(serving(["qwen3-vl:8b", "llava:7b"])[0], model=None).model == "qwen3-vl:8b"
    others = ["allenai/olmOCR-2-7B-1025", "nanonets/Nanonets-OCR2-3B"]   # other OCR models with names like ours
    assert client(serving(others + ["typhoon-ai/typhoon-ocr2-2b"])[0], model=None).model == "typhoon-ai/typhoon-ocr2-2b"
    with pytest.warns(UserWarning, match="none is named like Typhoon OCR"):
        assert client(serving(others)[0], model=None).model == "allenai/olmOCR-2-7B-1025"


@pytest.mark.parametrize("listing", [                  # sorted, 1.5 first; Ollama's /v1, newest first
    ["typhoon-ai/typhoon-ocr1.5-2b", "typhoon-ai/typhoon-ocr2-2b"],
    ["scb10x/typhoon-ocr1.5-3b:latest", "SuraponOCR/typhoon-ocr2-2b:latest", "llama3.2:latest"]])
def test_v2_is_picked_from_a_list_with_both_versions(page_image, listing):
    handler, bodies = serving(listing)
    ocr = client(handler, model=None)
    [r] = ocr.parse(page_image)
    assert "typhoon-ocr2-2b" in bodies[0]["model"] and ocr.version == "v2" and r.elements == ELEMENTS
    assert bodies[0]["messages"][0]["content"][1]["text"] == PARSE_PROMPT
    assert "ocr1.5" in client(serving(listing)[0], model=None, version="v1.5").model


def test_version_follows_the_served_model(page_image):
    # vLLM serves the --model argument as typed, e.g. a directory with the "/" that tab completion adds.
    for served in ("typhoon-ai/typhoon-ocr1.5-2b", "/models/typhoon-ocr1.5-2b/"):
        handler, bodies = serving([served], reply="# Title\n\ntext")
        ocr = client(handler, model=None)
        [r] = ocr.parse(page_image)
        assert ocr.version == "v1.5" and r.markdown == "# Title\n\ntext" and bodies[0]["model"] == served
        assert bodies[0]["temperature"] == 0.1 and bodies[0]["repetition_penalty"] == 1.1


@pytest.mark.parametrize("model, expected", [
    ("/models/typhoon-ocr1.5-2b/", "v1.5"), ("./typhoon-ocr1.5-2b/", "v1.5"), ("/models/typhoon-ocr2-2b/", "v2"),
    ("typhoon_ocr_v2", "v2"), ("typhoon_ocr_2", "v2"), ("typhoon-ocr_v2", "v2"), ("typhoon_ocr_1_5", "v1.5"),
    ("custom-model/", None), ("/", None)])
def test_version_of_served_paths_and_underscored_names(model, expected):
    assert _version_of(model) == expected


def test_version_that_does_not_match_the_default_model_warns(page_image):
    ocr = attach(TyphoonOCR(base_url="http://localhost:11434", version="v1.5"), FakeServer(reply="# x"))
    with pytest.warns(UserWarning, match="'SuraponOCR/typhoon-ocr2-2b' looks like Typhoon OCR v2, but version='v1.5'"):
        ocr.parse(page_image)


@pytest.mark.parametrize("status", [400, 500])             # 500: e.g. a proxy whose upstream fails on the field
def test_retry_without_chat_template_kwargs(page_image, status):
    seen, sleeps = [], []

    def handler(request):
        body = json.loads(request.content)
        seen.append("chat_template_kwargs" in body)
        if "chat_template_kwargs" in body:
            return httpx.Response(status, json={"error": "unknown field chat_template_kwargs"})
        return httpx.Response(200, json=OK)

    ocr = client(handler, sleeps)
    [r] = ocr.parse(page_image)
    assert seen == [True, False] and r.elements == ELEMENTS and sleeps == []   # sent again at once, without it
    ocr.parse(page_image)
    ocr.ask(page_image, "Total?")
    assert seen == [True, False, False, False]              # not sent again after the server rejected it


@pytest.mark.parametrize("model, expected", [
    ("typhoon-ocr-v2", "v2"), ("typhoon-ai/typhoon-ocr2-2b", "v2"), ("SuraponOCR/typhoon-ocr2-0.8b", "v2"),
    ("typhoon-ocr2-0.8b", "v2"), ("SuraponOCR/typhoon-ocr2-2b:int8", "v2"), ("typhoon-ocr2-2b-MLX-8bit", "v2"),
    ("typhoon-ocr2-2b-GGUF", "v2"), ("typhoon-ocr-v1.5", "v1.5"), ("typhoon-ocr", "v1.5"),
    ("typhoon-ai/typhoon-ocr1.5-2b", "v1.5"), ("typhoon-ocr-1-5", "v1.5"), ("typhoon_ocr_1_5", "v1.5"),
    ("typhoon-ocr1.5-2b", "v1.5"), ("scb10x/typhoon-ocr1.5-3b", "v1.5"), (None, "v2")])
def test_version_follows_model_name(model, expected):
    assert TyphoonOCR(base_url="http://x/v1", model=model).version == expected


def test_mismatched_version_warns():
    with pytest.warns(UserWarning):
        TyphoonOCR(base_url="http://x/v1", model="typhoon-ocr-v1.5", version="v2")


def test_api_key_over_plain_http_warns():
    with pytest.warns(UserWarning, match="unencrypted to 10.0.0.5"):
        TyphoonOCR(base_url="http://10.0.0.5:8000/v1", api_key="k")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for url in ("https://api.opentyphoon.ai/v1", "http://localhost:8000/v1", "http://127.0.0.1:11434",
                    "http://[::1]:8000/v1", "http://gpu.localhost/v1", "http://0.0.0.0:8000/v1", "http://[::]:8000/v1"):
            TyphoonOCR(base_url=url, api_key="k")
        TyphoonOCR(base_url="http://10.0.0.5:8000/v1")      # no key, nothing to warn about


def test_default_max_workers():
    assert TyphoonOCR(base_url="http://localhost:11434").max_workers == 1    # Ollama answers one request at a time
    assert TyphoonOCR(base_url="http://x/v1").max_workers == 4
    assert TyphoonOCR(base_url="http://localhost:11434", max_workers=3).max_workers == 3


@pytest.mark.parametrize("upscale, expected", [(True, 1600), (False, 0), (1800, 1800), (0, 0)])
def test_upscale_flag(upscale, expected):
    assert TyphoonOCR(base_url="http://x/v1", model="m", upscale=upscale).upscale == expected


def test_parse_exclude_keeps_all_elements(fake, page_image):
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), fake)
    [r] = ocr.parse(page_image, exclude=["title"])
    assert r.markdown == "Page 1\n\nรายได้เพิ่มขึ้น" and len(r.elements) == 3
    assert ocr.parse(page_image, exclude="")[0].markdown == "Page 1\n\nAnnual Report\n\nรายได้เพิ่มขึ้น"   # as []


def test_bad_exclude_and_pages_fail_before_any_request(fake, pdf_bytes):
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), fake)
    for kwargs in ({"exclude": ["footers"]}, {"exclude": "footer"}, {"pages": "3-1"}, {"pages": [1, 4]}):
        with pytest.raises(ValueError):
            ocr.parse(pdf_bytes, **kwargs)
    assert fake.requests == []


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_page_keeps_the_pages_already_parsed(workers):
    sent = []

    def handler(request):
        n = page_of(request)
        sent.append(n)
        if n == 3:
            return httpx.Response(500, text="page 3 broke the server")
        time.sleep(0.05)
        return httpx.Response(200, json=OK)
    ocr = client(handler, max_workers=workers, max_retries=0, upscale=False, pdf_long_side=200)
    with pytest.raises(typhoon_ocr2.PartialParseError, match="page 3 failed: .*500") as e:
        ocr.parse(grey_pdf(6))
    err = e.value
    assert isinstance(err, TyphoonOCRError) and list(err.errors) == [3] and "page 3 broke" in str(err.errors[3])
    assert str(err).endswith("(the parsed pages are in .results)")
    assert [r.page for r in err.results][:2] == [1, 2] and all(r.elements == ELEMENTS for r in err.results)
    assert sorted([r.page for r in err.results] + list(err.errors) + err.skipped) == [1, 2, 3, 4, 5, 6]
    assert 6 not in sent and set(err.skipped) == {1, 2, 3, 4, 5, 6} - set(sent)
    if workers == 1:
        assert sent == [1, 2, 3] and err.skipped == [4, 5, 6]
    copy = pickle.loads(pickle.dumps(err))                  # e.g. raised in a multiprocessing worker
    assert str(copy) == str(err) and copy.skipped == err.skipped
    assert [r.page for r in copy.results] == [r.page for r in err.results] and list(copy.errors) == [3]
    assert err.unreadable == copy.unreadable == []          # page 3 was read; the server failed it


@pytest.mark.parametrize("workers", [1, 2])
def test_a_page_that_cannot_be_rendered_stops_the_parse(monkeypatch, fake, pdf_bytes, workers):
    real = images._render_page

    def render(doc, n, long_side):
        if n == 2:
            raise images._pdfium().PdfiumError("Failed to load page.")
        return real(doc, n, long_side)
    monkeypatch.setattr(images, "_render_page", render)
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m", max_workers=workers), fake)
    with pytest.raises(typhoon_ocr2.PartialParseError, match="page 2 failed: cannot render page 2 of the PDF") as e:
        ocr.parse(pdf_bytes)
    assert [r.page for r in e.value.results] == [1] and list(e.value.errors) == [2] and e.value.skipped == [3]
    assert e.value.unreadable == pickle.loads(pickle.dumps(e.value)).unreadable == [2]
    assert len(fake.requests) == 1


def test_ctrl_c_stops_the_retries_of_the_pages_being_sent():
    # Ctrl-C while worker threads wait to retry pages the server rate-limited (Retry-After: 30): parse() comes out
    # at once, and no request is sent after it. _thread.interrupt_main() is a Ctrl-C that does not interrupt a wait
    # of the main thread, as on Windows.
    sent, first = [], threading.Event()

    def handler(request):
        sent.append(time.monotonic())
        first.set()
        return httpx.Response(429, headers={"Retry-After": "30"}, json={"error": {"message": "Rate limit exceeded"}})

    def ctrl_c():
        first.wait(10)
        time.sleep(0.2)                                     # the workers now wait to send their pages again
        _thread.interrupt_main()
    ocr = TyphoonOCR(base_url="http://x/v1", model="m", max_workers=4, max_retries=1, upscale=False,
                     pdf_long_side=100)
    ocr._client = httpx.Client(transport=httpx.MockTransport(handler))      # the retries wait for real
    threading.Thread(target=ctrl_c, daemon=True).start()
    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        ocr.parse(grey_pdf(6))
    assert time.monotonic() - start < 10                    # not the 30 s the server asked for
    assert 1 <= len(sent) <= 4 and max(sent) < start + 1    # pages 1-4 were sent once each, none after Ctrl-C


def test_pages_are_rendered_while_workers_are_busy(monkeypatch):
    rendered, in_main_thread = [], []
    real = images._render_page

    def render(*args):
        in_main_thread.append(threading.current_thread() is threading.main_thread())
        image = real(*args)
        rendered.append(weakref.ref(image))
        return image
    monkeypatch.setattr(images, "_render_page", render)
    alive = []

    def handler(request):
        alive.append(sum(r() is not None for r in rendered))
        time.sleep(0.02)
        return httpx.Response(200, json=OK)
    res = client(handler, max_workers=2, upscale=False, pdf_long_side=200).parse(grey_pdf(8))
    assert [r.page for r in res] == list(range(1, 9)) and all(in_main_thread)
    assert max(alive) <= 3                                  # max_workers + 1 page images at most, not all 8


def test_multi_page_tiff(tmp_path, fake):
    frames = [Image.new("L", (100 + 10 * i, 150), 255) for i in range(3)]
    tiff = tmp_path / "fax.tif"
    frames[0].save(tiff, save_all=True, append_images=frames[1:])
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m", upscale=False), fake)
    assert [(r.page, r.image_size) for r in ocr.parse(str(tiff))] == [(1, (100, 150)), (2, (110, 150)), (3, (120, 150))]
    ocr.ask(str(tiff), "Total?", page=2)
    assert sent_image(fake.requests[-1][1]).size == (110, 150)


def test_ask_reads_whole_page_numbers_as_parse_does(fake, page_image):
    frames = [Image.new("L", (100 + 10 * i, 150), 255) for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m", upscale=False), fake)
    for page in (2.0, Decimal("2"), "2"):
        ocr.ask(buf.getvalue(), "Total?", page=page)
        assert sent_image(fake.requests[-1][1]).size == (110, 150)
    for source in (buf.getvalue(), grey_pdf(3)):
        for page in (2.5, None, "two", float("nan")):
            with pytest.raises(ValueError, match="is not a page number"):
                ocr.ask(source, "Total?", page=page)
    for page in (None, 0, 2.5):                             # an image is its one page, whatever `page` is, as before
        ocr.ask(page_image, "Total?", page=page)
    assert len(fake.requests) == 6


def test_a_selection_without_pages_is_refused_for_a_pdf_or_tiff_only(fake, page_image, pdf_bytes):
    # A PDF or multi-page TIFF gave no page at all in 0.1.0; an image was parsed as its one page, and still is.
    frames = [Image.new("L", (60, 80), 20 * n) for n in (1, 2)]
    tiff = io.BytesIO()
    frames[0].save(tiff, "TIFF", save_all=True, append_images=frames[1:])
    png = io.BytesIO()
    page_image.save(png, "PNG")
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), fake)
    for pages in ("", [], (), " , "):
        for source in (png.getvalue(), page_image):
            assert [(r.page, r.elements) for r in ocr.parse(source, pages=pages)] == [(1, ELEMENTS)]
        for source in (pdf_bytes, tiff.getvalue()):
            with pytest.raises(ValueError, match=r"no page selected: pages="):
                ocr.parse(source, pages=pages)
    assert len(fake.requests) == 8
    with pytest.raises(ValueError, match="reversed"):     # a range written the wrong way round is still refused
        ocr.parse(png.getvalue(), pages="3-1")
