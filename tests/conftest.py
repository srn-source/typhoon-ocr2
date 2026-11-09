import io
import json

import httpx
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def page_image():
    im = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(im).text((40, 40), "Hello", fill="black")
    return im


@pytest.fixture
def pdf_bytes():
    pages = [Image.new("RGB", (612, 792), c) for c in ("white", "lightgray", "white")]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:], resolution=72)
    return buf.getvalue()


ELEMENTS = [{"bbox": [100, 50, 900, 100], "category": "header", "text": "Page 1"},
            {"bbox": [100, 120, 900, 180], "category": "title", "text": "Annual Report"},
            {"bbox": [100, 200, 900, 400], "category": "text", "text": "รายได้เพิ่มขึ้น"}]


class FakeServer:
    """Records requests and answers like vLLM (OpenAI API) or Ollama."""

    def __init__(self, reply="```json\n" + json.dumps(ELEMENTS, ensure_ascii=False) + "\n```", finish="stop"):
        self.reply, self.finish, self.requests = reply, finish, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "typhoon-ai/typhoon-ocr2-2b"}]})
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": self.reply}, "done_reason": self.finish})
        return httpx.Response(200, json={"choices": [{"message": {"content": self.reply}, "finish_reason": self.finish}]})


@pytest.fixture
def fake():
    return FakeServer()


def attach(ocr, server):
    ocr._client = httpx.Client(transport=httpx.MockTransport(server.handler))
    return ocr
