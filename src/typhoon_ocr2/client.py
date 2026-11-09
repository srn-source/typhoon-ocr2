"""Client for Typhoon OCR models served by vLLM, llama.cpp, MLX (mlx-vlm), Ollama or the Typhoon API."""
from __future__ import annotations

import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import httpx
from PIL import Image

from . import images as im
from .output import parse_elements, parse_json, to_markdown
from .prompts import EXTRACT_ALL_PROMPT, PARSE_PROMPT, extract_fields_prompt, v15_prompt

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OLLAMA_MODEL = "SuraponOCR/typhoon-ocr2-2b"
TYPHOON_API_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_MODELS = {"v2": "typhoon-ocr-v2", "v1.5": "typhoon-ocr-v1.5"}   # model names on the Typhoon API


class TyphoonOCRError(RuntimeError):
    """The server returned an error, or the output could not be used."""


@dataclass
class PageResult:
    """One parsed page. `elements` is empty for Typhoon OCR 1.5, which returns Markdown only."""
    page: int
    markdown: str
    elements: List[Dict] = field(default_factory=list)
    raw: str = ""
    finish_reason: Optional[str] = None
    image_size: Tuple[int, int] = (0, 0)     # (width, height) of the page image, for to_pixels()

    @property
    def truncated(self) -> bool:
        """True when the output stopped at the token limit (usually a repetition loop)."""
        return self.finish_reason == "length"


class TyphoonOCR:
    """Parse documents with Typhoon OCR.

    >>> ocr = TyphoonOCR(base_url="https://api.opentyphoon.ai/v1", model="typhoon-ocr-v2", version="v2")
    >>> pages = ocr.parse("report.pdf", pages="1-3")
    >>> print(pages[0].markdown)

    Args:
        base_url: server URL. OpenAI-compatible servers (Typhoon API, vLLM, llama.cpp, mlx-vlm) end in "/v1";
            Ollama is "http://localhost:11434". Defaults to $TYPHOON_OCR_BASE_URL, then "http://localhost:8000/v1".
        model: the model's name on the server. When None: "typhoon-ocr-v2" / "typhoon-ocr-v1.5" on the Typhoon
            API (following `version`), "SuraponOCR/typhoon-ocr2-2b" on Ollama, else the server's first model.
        api_key: for the Typhoon API or another server that needs one. Defaults to $TYPHOON_API_KEY,
            $TYPHOON_OCR_API_KEY or $OPENAI_API_KEY.
        backend: "openai", "ollama", or "auto" (Ollama when the URL has no "/v1" path and port 11434).
        version: "v2" (Typhoon OCR V2: layout elements and Markdown) or "v1.5" (Typhoon OCR 1.5: Markdown).
            Sets the prompt and output format. When None, it follows the model name (v1.5 for 1.5 models,
            otherwise v2).
        upscale: True (default) enlarges images whose long side is under 1,600 px; False turns it off; an
            int sets another size. V2 only.
        max_tokens: output token limit per page.
        pdf_long_side: long side, in pixels, of rendered PDF pages.
        max_workers: pages sent in parallel when parsing a multi-page document.
        timeout: seconds to wait for one page.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None,
                 backend: str = "auto", version: Optional[str] = None, upscale: Union[bool, int] = True,
                 max_tokens: int = 16384, pdf_long_side: int = im.PDF_LONG_SIDE, max_workers: int = 4,
                 timeout: float = 600.0, figure_language: str = "Thai"):
        if version not in (None, "v2", "v1.5"):
            raise ValueError("version must be 'v2' or 'v1.5'")
        self.base_url = (base_url or os.getenv("TYPHOON_OCR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.backend = _detect_backend(self.base_url) if backend == "auto" else backend
        if self.backend not in ("openai", "ollama"):
            raise ValueError("backend must be 'openai', 'ollama' or 'auto'")
        self.api_key = api_key or os.getenv("TYPHOON_API_KEY") or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._model = model or os.getenv("TYPHOON_OCR_MODEL")
        implied = _version_of(self._model)
        if version and implied and version != implied:
            warnings.warn(f"model {self._model!r} looks like Typhoon OCR {implied}, but version={version!r}: "
                          f"the {version} prompt will be sent to it", stacklevel=2)
        self.version = version or implied or "v2"
        self.upscale = im.UPSCALE_MIN_LONG if upscale is True else int(upscale or 0)
        self.max_tokens = max_tokens
        self.pdf_long_side, self.max_workers, self.figure_language = pdf_long_side, max_workers, figure_language
        self._client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})

    # ---------------------------------------------------------------- public API
    @property
    def model(self) -> str:
        if not self._model:
            if self.backend == "ollama":
                self._model = DEFAULT_OLLAMA_MODEL
            elif _is_typhoon_api(self.base_url):
                self._model = TYPHOON_API_MODELS[self.version]
            else:
                self._model = self._first_served_model()
        return self._model

    def parse(self, source: im.ImageInput, pages: Union[None, int, str, Sequence[int]] = None,
              keep_header_footer: bool = False, exclude: Optional[Sequence[str]] = None) -> List[PageResult]:
        """Parse each page. `pages` selects PDF pages (1-based), e.g. 2, "1-3,5" or [1, 4].

        `page.elements` always holds every element the model returned. `page.markdown` leaves out header,
        footer and page_footnote elements unless `keep_header_footer=True`; `exclude` picks the categories to
        leave out yourself (e.g. ["footer"], or [] to keep everything)."""
        page_numbers = im.parse_pages(pages)
        page_images = im.load_pages(source, page_numbers, self.pdf_long_side)
        numbers = page_numbers or list(range(1, len(page_images) + 1))

        def one(args):
            n, page = args
            prompt = PARSE_PROMPT if self.version == "v2" else v15_prompt(self.figure_language)
            raw, finish = self._generate(self._prepare(page), prompt)
            if self.version == "v1.5":
                return PageResult(n, raw, [], raw, finish, page.size)
            try:
                elements = parse_elements(raw)
            except ValueError:
                elements = []
            return PageResult(n, to_markdown(elements, keep_header_footer, exclude), elements, raw, finish, page.size)

        jobs = list(zip(numbers, page_images))
        if len(jobs) == 1 or self.max_workers <= 1:
            return [one(j) for j in jobs]
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(one, jobs))

    def ocr(self, source: im.ImageInput, pages: Union[None, int, str, Sequence[int]] = None,
            keep_header_footer: bool = False, exclude: Optional[Sequence[str]] = None) -> str:
        """Markdown of the selected pages, separated by blank lines."""
        return "\n\n".join(p.markdown for p in self.parse(source, pages, keep_header_footer, exclude))

    def ask(self, source: im.ImageInput, question: str, page: int = 1) -> str:
        """Ask a question about one page (Typhoon OCR V2 2B). Returns the answer as text."""
        image = im.load_pages(source, [page] if im.is_pdf(source) else None, self.pdf_long_side)[0]
        return self._generate(self._prepare(image), question)[0].strip()

    def extract(self, source: im.ImageInput, fields: Optional[Sequence[str]] = None, page: int = 1):
        """Extract key information as JSON (Typhoon OCR V2 2B). Without `fields`, the model picks the
        fields itself, using the printed labels as keys. Raises TyphoonOCRError if the answer is not JSON."""
        prompt = EXTRACT_ALL_PROMPT if not fields else extract_fields_prompt(fields)
        raw = self.ask(source, prompt, page)
        data = parse_json(raw)
        if data is None:
            raise TyphoonOCRError(f"the answer is not valid JSON: {raw[:300]!r}")
        return data

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- internals
    def _prepare(self, image: Image.Image) -> Image.Image:
        if self.version == "v1.5":
            return _resize_v15(image)
        return im.upscale_min_long(image, self.upscale) if self.upscale else image

    def _generate(self, image: Image.Image, prompt: str) -> Tuple[str, Optional[str]]:
        if self.backend == "ollama":
            body = {"model": self.model, "stream": False, "think": False,
                    "messages": [{"role": "user", "content": prompt, "images": [im.to_base64_png(image)]}],
                    "options": {"temperature": 0, "top_p": 1.0, "num_predict": self.max_tokens, "num_ctx": 40960}}
            if self.version == "v1.5":
                body["options"].update(temperature=0.1, top_p=0.6, repeat_penalty=1.1)
            d = self._post(f"{self.base_url}/api/chat", body)
            return d["message"]["content"], d.get("done_reason")
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": im.to_data_url(image)}},
                    {"type": "text", "text": prompt}]}]}
        if self.version == "v2":
            body.update(temperature=0.0, top_p=1.0, chat_template_kwargs={"enable_thinking": False})
        else:  # Typhoon OCR 1.5 settings from the original typhoon-ocr package
            body.update(temperature=0.1, top_p=0.6, repetition_penalty=1.1)
        try:
            d = self._post(f"{self.base_url}/chat/completions", body)
        except TyphoonOCRError as e:
            # Some OpenAI-compatible gateways reject vLLM's chat_template_kwargs: retry once without it.
            if "chat_template_kwargs" not in body or "chat_template_kwargs" not in str(e):
                raise
            body.pop("chat_template_kwargs")
            d = self._post(f"{self.base_url}/chat/completions", body)
        d = d["choices"][0]
        return d["message"].get("content") or "", d.get("finish_reason")

    def _post(self, url: str, body: dict) -> dict:
        try:
            r = self._client.post(url, json=body)
        except httpx.HTTPError as e:
            raise TyphoonOCRError(f"cannot reach {url}: {e}") from e
        if r.status_code >= 400:
            raise TyphoonOCRError(f"{url} returned {r.status_code}: {r.text[:500]}")
        return r.json()

    def _first_served_model(self) -> str:
        try:
            r = self._client.get(f"{self.base_url}/models")
            r.raise_for_status()
            data = r.json()
            models = data["data"] if isinstance(data, dict) else data      # vLLM: {"data": [...]}; some APIs: [...]
            return models[0]["id"]
        except Exception as e:
            raise TyphoonOCRError(f"pass model=...: could not list the models served at {self.base_url} ({e})") from e


def _detect_backend(base_url: str) -> str:
    u = urlparse(base_url)
    return "ollama" if u.port == 11434 and not u.path.rstrip("/").endswith("/v1") else "openai"


def _version_of(model: Optional[str]) -> Optional[str]:
    """'v1.5' for Typhoon OCR 1.5 model names (typhoon-ocr, typhoon-ocr-v1.5, typhoon-ai/typhoon-ocr1.5-2b),
    'v2' for V2 names (typhoon-ocr-v2, typhoon-ai/typhoon-ocr2-2b), None when the name says nothing."""
    if not model:
        return None
    name = model.lower().rsplit("/", 1)[-1]
    if re.search(r"ocr-?v?1\.5", name) or name == "typhoon-ocr":
        return "v1.5"
    if re.search(r"ocr-?v?2", name):
        return "v2"
    return None


def _is_typhoon_api(base_url: str) -> bool:
    return urlparse(base_url).hostname == urlparse(TYPHOON_API_URL).hostname


def _resize_v15(image: Image.Image, long_side: int = 1800) -> Image.Image:
    """Typhoon OCR 1.5 resizing from the original package: the long side becomes 1800 px."""
    w, h = image.size
    if w <= 300 and h <= 300:
        return image
    s = long_side / max(w, h)
    return image.resize((round(w * s), round(h * s)), Image.LANCZOS)


def ocr_document(source: im.ImageInput, pages: Union[None, int, str, Sequence[int]] = None, **kwargs) -> str:
    """One call: parse a PDF or image and return Markdown. Keyword arguments go to TyphoonOCR."""
    with TyphoonOCR(**kwargs) as ocr:
        return ocr.ocr(source, pages)
