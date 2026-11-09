import json

import pytest

from typhoon_ocr2 import PARSE_PROMPT, TyphoonOCR, TyphoonOCRError
from typhoon_ocr2.client import _detect_backend
from conftest import ELEMENTS, FakeServer, attach


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
    assert r.image_size == (400, 600) and not r.truncated


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
    assert r.elements == [] and r.raw == "sorry"


def test_v15_returns_markdown_with_v15_settings(page_image):
    srv = FakeServer(reply="# Title\n\ntext")
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="typhoon-ocr", version="v1.5"), srv)
    [r] = ocr.parse(page_image)
    body = srv.requests[0][1]
    assert r.markdown == "# Title\n\ntext" and r.elements == []
    assert body["temperature"] == 0.1 and body["repetition_penalty"] == 1.1
    assert "Describe in Thai." in body["messages"][0]["content"][1]["text"]


def test_ask_and_extract(page_image):
    srv = FakeServer(reply='```json\n{"Total": "355.00"}\n```')
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), srv)
    assert ocr.extract(page_image) == {"Total": "355.00"}
    assert "open-schema" in srv.requests[0][1]["messages"][0]["content"][1]["text"]
    assert ocr.ask(page_image, "Total?").startswith("```json")


def test_extract_invalid_json_raises(page_image):
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), FakeServer(reply="no idea"))
    with pytest.raises(TyphoonOCRError):
        ocr.extract(page_image)


def test_server_error_is_reported(page_image):
    import httpx
    ocr = TyphoonOCR(base_url="http://x/v1", model="m")
    ocr._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(TyphoonOCRError, match="500"):
        ocr.parse(page_image)


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
    import httpx
    ocr = TyphoonOCR(base_url="http://x/v1")
    ocr._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[{"id": "m1"}, {"id": "m2"}])))
    assert ocr.model == "m1"


def test_retry_without_chat_template_kwargs(page_image):
    import httpx
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append("chat_template_kwargs" in body)
        if "chat_template_kwargs" in body:
            return httpx.Response(400, json={"error": "unknown field chat_template_kwargs"})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(ELEMENTS)}, "finish_reason": "stop"}]})

    ocr = TyphoonOCR(base_url="http://x/v1", model="m")
    ocr._client = httpx.Client(transport=httpx.MockTransport(handler))
    [r] = ocr.parse(page_image)
    assert seen == [True, False] and r.elements == ELEMENTS


@pytest.mark.parametrize("model, expected", [
    ("typhoon-ocr-v2", "v2"), ("typhoon-ai/typhoon-ocr2-2b", "v2"), ("SuraponOCR/typhoon-ocr2-0.8b", "v2"),
    ("typhoon-ocr-v1.5", "v1.5"), ("typhoon-ocr", "v1.5"), ("typhoon-ai/typhoon-ocr1.5-2b", "v1.5"), (None, "v2")])
def test_version_follows_model_name(model, expected):
    assert TyphoonOCR(base_url="http://x/v1", model=model).version == expected


def test_mismatched_version_warns():
    with pytest.warns(UserWarning):
        TyphoonOCR(base_url="http://x/v1", model="typhoon-ocr-v1.5", version="v2")


@pytest.mark.parametrize("upscale, expected", [(True, 1600), (False, 0), (1800, 1800), (0, 0)])
def test_upscale_flag(upscale, expected):
    assert TyphoonOCR(base_url="http://x/v1", model="m", upscale=upscale).upscale == expected


def test_parse_exclude_keeps_all_elements(fake, page_image):
    ocr = attach(TyphoonOCR(base_url="http://x/v1", model="m"), fake)
    [r] = ocr.parse(page_image, exclude=["title"])
    assert r.markdown == "Page 1\n\nรายได้เพิ่มขึ้น" and len(r.elements) == 3
