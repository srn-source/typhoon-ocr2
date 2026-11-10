"""Client for Typhoon OCR models served by vLLM, llama.cpp, MLX (mlx-vlm), Ollama or the Typhoon API."""
from __future__ import annotations

import ipaddress
import json
import math
import os
import random
import re
import threading
import time
import warnings
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image

from . import images as im
from .output import parse_elements, parse_json, to_markdown, validate_exclude
from .prompts import EXTRACT_ALL_PROMPT, PARSE_PROMPT, extract_fields_prompt, v15_prompt

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OLLAMA_MODEL = "SuraponOCR/typhoon-ocr2-2b"
TYPHOON_API_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_API_MODELS = {"v2": "typhoon-ocr-v2", "v1.5": "typhoon-ocr-v1.5"}   # model names on the Typhoon API

# Retried up to max_retries times: rate limits, server errors, and connections that failed or dropped (also while
# the page was being sent). Timeouts are not retried: the server may still be working on the page, and a
# connection has already waited `timeout`.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)
_RETRY_AFTER_MAX = 60.0                     # seconds: the longest Retry-After we wait for
_BACKOFF_BASE, _BACKOFF_MAX = 1.0, 30.0     # seconds: exponential backoff with full jitter
# The Typhoon API's 429 answer: {"error": {"message": "Rate limit exceeded. Please retry after 2024-01-01T12:00:00Z"}}
_RETRY_AT = re.compile(r"retry after (\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d)?)(?:\.\d+)?\s*(Z|[+-]\d\d:?\d\d)?", re.I)


class TyphoonOCRError(RuntimeError):
    """The server returned an error, or the output could not be used."""


@dataclass
class PageResult:
    """One parsed page. `elements` is empty for Typhoon OCR 1.5, which returns Markdown only, and when the
    V2 output could not be parsed: `parse_error` then says why, and `raw` keeps the model's answer."""
    page: int
    markdown: str
    elements: List[Dict] = field(default_factory=list)
    raw: str = ""
    finish_reason: Optional[str] = None
    image_size: Tuple[int, int] = (0, 0)     # (width, height) of the page image, for to_pixels()
    parse_error: Optional[str] = None        # why the V2 output is not a list of layout elements

    @property
    def truncated(self) -> bool:
        """True when the output stopped at the token limit (usually a repetition loop)."""
        return self.finish_reason == "length"


class PartialParseError(TyphoonOCRError):
    """parse() stopped because a page failed. `results` holds the pages that were parsed, in the requested
    order; `errors` maps each failed page number to its exception; `skipped` lists the pages never sent;
    `unreadable` lists the failed pages that could not be read from the file (a PDF or TIFF damaged there)."""

    def __init__(self, message: str, results: Sequence[PageResult] = (), errors: Optional[Dict] = None,
                 skipped: Sequence[int] = (), unreadable: Sequence[int] = ()):
        super().__init__(message)
        self.results, self.errors, self.skipped = list(results), dict(errors or {}), list(skipped)
        self.unreadable = list(unreadable)

    def __reduce__(self):
        return type(self), (str(self), self.results, self.errors, self.skipped, self.unreadable)


class TyphoonOCR:
    """Parse documents with Typhoon OCR.

    >>> ocr = TyphoonOCR(base_url="https://api.opentyphoon.ai/v1", model="typhoon-ocr-v2", version="v2")
    >>> pages = ocr.parse("report.pdf", pages="1-3")
    >>> print(pages[0].markdown)

    Args:
        base_url: server URL. OpenAI-compatible servers (Typhoon API, vLLM, llama.cpp, mlx-vlm) end in "/v1";
            Ollama is "http://localhost:11434". Defaults to $TYPHOON_OCR_BASE_URL, then "http://localhost:8000/v1".
        model: the model's name on the server. When None: $TYPHOON_OCR_MODEL, else "typhoon-ocr-v2" /
            "typhoon-ocr-v1.5" on the Typhoon API (following `version`), "SuraponOCR/typhoon-ocr2-2b" on Ollama,
            else the model the server lists (when it lists several, the first whose name looks like Typhoon OCR,
            preferring one of `version`, V2 by default).
        api_key: for the Typhoon API or another server that needs one. Defaults to $TYPHOON_API_KEY,
            $TYPHOON_OCR_API_KEY or $OPENAI_API_KEY. Spaces and line breaks at its end are dropped (a key read
            from a file). A UserWarning says when it would be sent over plain http to another machine.
        backend: "openai", "ollama", or "auto" (Ollama when the URL has no "/v1" path and port 11434).
        version: "v2" (Typhoon OCR V2: layout elements and Markdown) or "v1.5" (Typhoon OCR 1.5: Markdown).
            Sets the prompt and output format. When None, it follows the model name (v1.5 for 1.5 models,
            otherwise v2), also when the name comes from the server: the first parse(), ask() or extract()
            looks it up. A version that does not match the model name gives a UserWarning.
        upscale: True (default) enlarges images whose long side is under 1,600 px; False turns it off; an
            int sets another size. V2 only.
        max_tokens: output token limit per page.
        pdf_long_side: long side, in pixels, of rendered PDF pages.
        max_workers: pages sent in parallel when parsing a multi-page document. When None: 1 on Ollama, which
            answers one request at a time unless OLLAMA_NUM_PARALLEL is set (pages queued behind others count
            against `timeout`), and 4 elsewhere.
        timeout: seconds to wait for the answer to one request, including the time it waits in the
            server's queue.
        figure_language: "Thai" or "English", the language of figure descriptions. V1.5 only.
        max_retries: times a request is retried after HTTP 429, 500, 502, 503 or 504, or a connection that
            failed or dropped. It waits as the server asks (a Retry-After header, or the Typhoon API's "retry
            after <time>"; up to 60 s), else 0-1 s, 0-2 s, 0-4 s and so on (at most 30 s). Timeouts are not
            retried.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None,
                 backend: str = "auto", version: Optional[str] = None, upscale: Union[bool, int] = True,
                 max_tokens: int = 16384, pdf_long_side: int = im.PDF_LONG_SIDE, max_workers: Optional[int] = None,
                 timeout: float = 600.0, figure_language: str = "Thai", max_retries: int = 3):
        if version not in (None, "v2", "v1.5"):
            raise ValueError("version must be 'v2' or 'v1.5'")
        if max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        self.base_url = (base_url or os.getenv("TYPHOON_OCR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.backend = _detect_backend(self.base_url) if backend == "auto" else backend
        if self.backend not in ("openai", "ollama"):
            raise ValueError("backend must be 'openai', 'ollama' or 'auto'")
        self.api_key = api_key or os.getenv("TYPHOON_API_KEY") or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY")
        if isinstance(self.api_key, str):
            self.api_key = self.api_key.rstrip()      # a trailing line break or space is not valid in a header
        host = _remote_http_host(self.base_url)
        if self.api_key and host:
            warnings.warn(f"the API key will be sent unencrypted to {host} (plain http). Use https, or, if this "
                          "server needs no key, do not pass api_key and unset $TYPHOON_API_KEY, "
                          "$TYPHOON_OCR_API_KEY and $OPENAI_API_KEY", stacklevel=2)
        self._model = model or os.getenv("TYPHOON_OCR_MODEL")
        self._version_given = version is not None
        implied = _version_of(self._model)
        if version and implied and version != implied:
            warnings.warn(_mismatch(self._model, implied, version), stacklevel=2)
        self.version = version or implied or "v2"
        self._checked_model = self._model         # the model name `version` has been checked against
        self.upscale = im.UPSCALE_MIN_LONG if upscale is True else int(upscale or 0)
        self.max_tokens = max_tokens
        self.pdf_long_side, self.figure_language = pdf_long_side, figure_language
        self.max_workers = max_workers if max_workers is not None else 1 if self.backend == "ollama" else 4
        self.max_retries = max_retries
        self._sleep = None                        # waits between retries; None waits for real (tests replace it)
        self._template_kwargs = True              # False once a server has rejected chat_template_kwargs
        self._lock = threading.Lock()
        self._local = threading.local()           # .stop: the Event of the parse() a worker thread is sending for
        self._client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})

    # ---------------------------------------------------------------- public API
    @property
    def model(self) -> str:
        if not self._model:
            self._model = self._default_model()
        return self._model

    def parse(self, source: im.ImageInput, pages: Union[None, int, str, Sequence[int]] = None,
              keep_header_footer: bool = False, exclude: Optional[Sequence[str]] = None) -> List[PageResult]:
        """Parse each page. `pages` selects PDF pages or TIFF frames (1-based), e.g. 2, "1-3,5", [1, 4] or
        range(1, 4); a page listed twice is parsed once.

        `page.elements` always holds every element the model returned. `page.markdown` leaves out header,
        footer and page_footnote elements unless `keep_header_footer=True`; `exclude` picks the categories to
        leave out yourself (e.g. ["footer"], or [] to keep everything).

        Pages are rendered one at a time while earlier pages are with the server. When a page fails (after
        the retries), no further page is sent, the pages already sent are finished, and PartialParseError is
        raised: its `results` holds the pages that were parsed."""
        skip = validate_exclude(exclude)
        numbered = im._page_source(source, im._selection(pages, empty=True), self.pdf_long_side)
        try:
            numbers = next(numbered)                # every page number is checked before anything is sent
            if not numbers and pages is not None:   # a PDF or multi-page TIFF; an image is its one page, as before
                raise ValueError(f"no page selected: pages={pages!r}")
            self._resolve()
            prompt = PARSE_PROMPT if self.version == "v2" else v15_prompt(self.figure_language)
            return self._run(numbers, numbered,
                             lambda n, page: self._parse_page(n, page, prompt, keep_header_footer, skip))
        finally:
            numbered.close()

    def ocr(self, source: im.ImageInput, pages: Union[None, int, str, Sequence[int]] = None,
            keep_header_footer: bool = False, exclude: Optional[Sequence[str]] = None) -> str:
        """Markdown of the selected pages, separated by blank lines."""
        return "\n\n".join(p.markdown for p in self.parse(source, pages, keep_header_footer, exclude))

    def ask(self, source: im.ImageInput, question: str, page: int = 1) -> str:
        """Ask a question about one page (Typhoon OCR V2 2B): `page` of a PDF or multi-page TIFF, or the image
        itself. Returns the answer as text."""
        image = im._one_page(source, page, self.pdf_long_side)
        self._resolve()
        return self._generate(self._prepare(image), question)[0].strip()

    def extract(self, source: im.ImageInput, fields: Union[None, str, Sequence[str]] = None, page: int = 1):
        """Extract key information as JSON (Typhoon OCR V2 2B). `fields` names the fields to extract (one name
        can be a plain string). Without `fields`, the model picks the fields itself, using the printed labels
        as keys. Raises TyphoonOCRError if the answer is not JSON."""
        if isinstance(fields, str):
            fields = [fields] if fields.strip() else None      # "" names no field, as before
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
    def _default_model(self) -> str:
        if self.backend == "ollama":
            return DEFAULT_OLLAMA_MODEL
        if _is_typhoon_api(self.base_url):
            return TYPHOON_API_MODELS[self.version]
        return self._first_served_model()

    def _resolve(self) -> None:
        """Look up the model name once, in the calling thread, before any page is prepared. `version` then
        follows the name when it was not given; when it was given and the name says otherwise, warn once. The
        lookup runs outside the lock: threads that share this instance and start together each look the name up,
        as in 0.1.0, instead of repeating a lookup that fails one after the other."""
        name = self._model or self._default_model()
        with self._lock:
            if not self._model:
                self._model = name
            name = self._model
            if name == self._checked_model:
                return
            self._checked_model = name
            implied = _version_of(name)
            if implied and implied != self.version:
                if self._version_given:
                    warnings.warn(_mismatch(name, implied, self.version), stacklevel=3)
                else:
                    self.version = implied

    def _run(self, numbers: List[int], pages: Iterator[Tuple[int, Image.Image]],
             job: Callable[[int, Image.Image], PageResult]) -> List[PageResult]:
        """job(n, image) for every page, up to max_workers at a time. Pages are rendered here, in the calling
        thread, and only while a worker is free or about to be, so at most max_workers + 1 page images exist.
        After the first failure no page is started; the pages in flight finish; then PartialParseError. When this
        thread is interrupted (Ctrl-C), the pages in flight send no further request: their retries stop."""
        done: Dict[int, PageResult] = {}
        errors: Dict[int, Exception] = {}
        unreadable: List[int] = []                  # pages that could not be read from the file
        if len(numbers) <= 1 or self.max_workers <= 1:
            for n in numbers:
                try:
                    page = next(pages)
                except Exception as e:
                    errors[n] = e
                    unreadable.append(n)
                    break
                try:
                    done[n] = job(*page)
                except Exception as e:
                    errors[n] = e
                    break
                del page
        else:
            stop = threading.Event()
            with ThreadPoolExecutor(self.max_workers) as pool:
                running = {}

                def collect(futures):
                    for f in futures:
                        n = running.pop(f)
                        if not f.cancelled():
                            try:
                                done[n] = f.result()
                            except Exception as e:
                                errors[n] = e

                try:
                    for n in numbers:
                        collect([f for f in running if f.done()])
                        if errors:
                            break
                        try:
                            page = next(pages)
                        except Exception as e:      # the page could not be rendered
                            errors[n] = e
                            unreadable.append(n)
                            break
                        while len(running) >= self.max_workers and not errors:
                            collect(_finished(running, FIRST_COMPLETED))
                        if errors:
                            break
                        running[pool.submit(self._in_worker, stop, job, *page)] = n
                        del page
                    if errors:
                        for f in running:
                            f.cancel()              # not started yet: never sent
                    collect(_finished(running, ALL_COMPLETED))
                except BaseException:               # KeyboardInterrupt: the workers stop waiting to retry
                    stop.set()
                    for f in running:
                        f.cancel()
                    raise
        if not errors:
            return [done[n] for n in numbers]
        results = [done[n] for n in numbers if n in done]
        skipped = [n for n in numbers if n not in done and n not in errors]
        first, error = next(iter(errors.items()))
        message = f"page {first} failed: {error}; {len(results)} of {len(numbers)} page(s) were parsed"
        if len(errors) > 1:
            message += f", {len(errors) - 1} more failed"
        if skipped:
            message += f", {len(skipped)} were not sent"
        if results:
            message += " (the parsed pages are in .results)"
        raise PartialParseError(message, results, errors, skipped, unreadable) from error

    def _in_worker(self, stop: threading.Event, job: Callable[[int, Image.Image], PageResult], n: int,
                   page: Image.Image) -> PageResult:
        """job(n, page) in a worker thread of _run(); once `stop` is set, its requests are not retried."""
        self._local.stop = stop
        try:
            return job(n, page)
        finally:
            self._local.stop = None

    def _parse_page(self, n: int, page: Image.Image, prompt: str, keep_header_footer: bool, exclude) -> PageResult:
        raw, finish = self._generate(self._prepare(page), prompt)
        if self.version == "v1.5":
            return PageResult(n, raw, [], raw, finish, page.size)
        try:
            elements = parse_elements(raw)
        except ValueError as e:
            return PageResult(n, "", [], raw, finish, page.size, str(e))
        return PageResult(n, to_markdown(elements, keep_header_footer, exclude), elements, raw, finish, page.size)

    def _prepare(self, image: Image.Image) -> Image.Image:
        if self.version == "v1.5":
            return _resize_v15(image)
        return im.upscale_min_long(image, self.upscale) if self.upscale else image

    def _generate(self, image: Image.Image, prompt: str) -> Tuple[str, Optional[str]]:
        if self.backend == "ollama":
            url = f"{self.base_url}/api/chat"
            body = {"model": self.model, "stream": False, "think": False,
                    "messages": [{"role": "user", "content": prompt, "images": [im.to_base64_png(image)]}],
                    "options": {"temperature": 0, "top_p": 1.0, "num_predict": self.max_tokens, "num_ctx": 40960}}
            if self.version == "v1.5":
                body["options"].update(temperature=0.1, top_p=0.6, repeat_penalty=1.1)
            d = self._post(url, body)
            message = d.get("message") if isinstance(d, dict) else None
            if not isinstance(message, dict):           # e.g. {"error": "..."}
                raise TyphoonOCRError(f"unexpected answer from {url}: {_excerpt(d)}")
            return _content(message, url, d), d.get("done_reason")
        url = f"{self.base_url}/chat/completions"
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": im.to_data_url(image)}},
                    {"type": "text", "text": prompt}]}]}
        if self.version == "v2":
            body.update(temperature=0.0, top_p=1.0)
            if self._template_kwargs:
                body["chat_template_kwargs"] = {"enable_thinking": False}
        else:  # Typhoon OCR 1.5 settings from the original typhoon-ocr package
            body.update(temperature=0.1, top_p=0.6, repetition_penalty=1.1)
        try:
            d = self._post(url, body, final=_names_template_kwargs if "chat_template_kwargs" in body else None)
        except TyphoonOCRError as e:
            # Some OpenAI-compatible gateways reject vLLM's chat_template_kwargs: retry once without it, and
            # leave it out of later requests when that works. Such an answer is not retried with it, also a 5xx.
            if "chat_template_kwargs" not in body or "chat_template_kwargs" not in str(e):
                raise
            body.pop("chat_template_kwargs")
            d = self._post(url, body)
            self._template_kwargs = False
        choices = d.get("choices") if isinstance(d, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise TyphoonOCRError(f"unexpected answer from {url}: {_excerpt(d)}")
        return _content(choice["message"], url, d), choice.get("finish_reason")

    def _post(self, url: str, body: dict, final: Optional[Callable[[httpx.Response], bool]] = None):
        r = self._send("POST", url, final, json=body)
        try:
            return r.json()
        except (ValueError, RecursionError) as e:       # an HTML page, an empty body, JSON nested too deeply, ...
            raise TyphoonOCRError(f"{url} did not answer with JSON: {r.text[:300]!r}") from e

    def _send(self, method: str, url: str, final: Optional[Callable[[httpx.Response], bool]] = None,
              **kwargs) -> httpx.Response:
        """Send a request; retry HTTP 429/500/502/503/504 and failed or dropped connections up to max_retries
        times, but not an answer for which final(answer) is true. Raises TyphoonOCRError unless the final answer is
        a 2xx."""
        stop = getattr(self._local, "stop", None)   # set when the parse() this worker sends for was interrupted
        for attempt in range(self.max_retries + 1):
            if stop is not None and stop.is_set():
                raise TyphoonOCRError(f"not sent to {url}: the parse was interrupted")
            last = attempt == self.max_retries
            try:
                r = self._client.request(method, url, **kwargs)
            except _RETRY_ERRORS as e:
                if isinstance(e, httpx.RemoteProtocolError) and "location header" in str(e):
                    # httpx reads the Location of a redirect (301, 302, 303, 307, 308) also when it does not follow
                    # it: an answer, not a dropped connection
                    raise TyphoonOCRError(f"{url} answered a redirect with an invalid Location "
                                          f"({str(e).rstrip('.')}): check base_url") from e
                if last:
                    raise TyphoonOCRError(f"cannot reach {url}: {e}") from e
                self._wait(_backoff(attempt), stop)
                continue
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:   # it may still be working on it
                raise TyphoonOCRError(f"no answer from {url} in time ({e}); the server may be busy with this "
                                      "or other pages: raise timeout= or lower max_workers=") from e
            except httpx.LocalProtocolError as e:   # its message repeats the header value, the API key included
                raise TyphoonOCRError(f"cannot send the request to {url}: a header is not valid; check the API key "
                                      "for spaces, line breaks or other characters") from e
            except (httpx.HTTPError, httpx.InvalidURL) as e:     # a connect timeout; InvalidURL: e.g. a bad port
                raise TyphoonOCRError(f"cannot reach {url}: {e}") from e
            if r.status_code not in _RETRY_STATUSES or last or (final is not None and final(r)):
                break
            delay = _retry_after(r.headers.get("retry-after"))
            if delay is None:
                delay = _retry_at(r.text)
            self._wait(_backoff(attempt) if delay is None else delay, stop)
        if 300 <= r.status_code < 400:                  # httpx does not follow redirects
            if not r.headers.get("location"):
                raise TyphoonOCRError(f"{url} answered {r.status_code}, a redirect without a Location: check base_url")
            try:
                target = urljoin(url, r.headers["location"])
            except ValueError:                          # e.g. "http://[::1"
                raise TyphoonOCRError(f"{url} answered {r.status_code} with an invalid Location "
                                      f"{r.headers['location'][:200]!r}: check base_url") from None
            path = url[len(self.base_url):]             # "/chat/completions", "/api/chat" or "/models"
            base = target[:-len(path)] if path and target.endswith(path) else target
            raise TyphoonOCRError(f"{url} redirected ({r.status_code}) to {target}; use that address as base_url"
                                  + (f", i.e. base_url={base!r}" if base != target else ""))
        if r.status_code >= 400:
            tries = f" after {attempt + 1} attempts" if attempt else ""
            raise TyphoonOCRError(f"{url} returned {r.status_code}{tries}: {r.text[:500]}")
        return r

    def _wait(self, seconds: float, stop: Optional[threading.Event]) -> None:
        """Wait before a retry. In a parse() worker thread the wait ends as soon as that parse is interrupted: a sleep
        there would hold Ctrl-C up for as long as the server asked to wait, and the retry would still be sent."""
        if self._sleep is not None:
            self._sleep(seconds)
        elif stop is not None:
            stop.wait(seconds)
        else:
            time.sleep(seconds)

    def _first_served_model(self) -> str:
        try:
            data = self._send("GET", f"{self.base_url}/models").json()
            models = data["data"] if isinstance(data, dict) else data      # vLLM: {"data": [...]}; some APIs: [...]
            ids = [m["id"] for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)]
            if not ids:
                raise ValueError("the list has no model name")
            if len(ids) == 1:
                return ids[0]
        except Exception as e:
            if isinstance(e, TyphoonOCRError) and isinstance(e.__cause__, (httpx.HTTPError, httpx.InvalidURL)):
                raise                                   # the server cannot be reached: model= would not help
            raise TyphoonOCRError(f"pass model=...: could not list the models served at {self.base_url} ({e})") from e
        # Servers such as mlx-vlm and Ollama's /v1 list every model they have: take a Typhoon OCR one (other
        # OCR models, such as olmOCR-2 or Nanonets-OCR2, also have names like ours), preferring one of `version`
        # (V2 unless version= says 1.5), whatever order the server lists them in.
        known = [i for i in ids if "typhoon" in i.lower() and _version_of(i)]
        if known:
            same = [i for i in known if _version_of(i) == self.version]
            return (same or known)[0]
        listed = ", ".join(map(repr, ids[:5])) + (", ..." if len(ids) > 5 else "")
        warnings.warn(f"{self.base_url} serves {len(ids)} models and none is named like Typhoon OCR, so the first, "
                      f"{ids[0]!r}, is used; pass model= to choose another ({listed})", stacklevel=2)
        return ids[0]


def _detect_backend(base_url: str) -> str:
    u = urlparse(base_url)
    return "ollama" if u.port == 11434 and not u.path.rstrip("/").endswith("/v1") else "openai"


def _version_of(model: Optional[str]) -> Optional[str]:
    """'v1.5' for Typhoon OCR 1.5 model names (typhoon-ocr, typhoon-ocr-v1.5, typhoon-ocr-1-5, typhoon_ocr_1_5,
    typhoon-ai/typhoon-ocr1.5-2b, a served path such as /models/typhoon-ocr1.5-2b/), 'v2' for V2 names
    (typhoon-ocr-v2, typhoon_ocr_v2, typhoon-ai/typhoon-ocr2-2b), None when the name says nothing."""
    if not model:
        return None
    name = model.lower().rstrip("/").rsplit("/", 1)[-1]
    if re.search(r"ocr[-_]?v?1[._-]5", name) or name == "typhoon-ocr":
        return "v1.5"
    if re.search(r"ocr[-_]?v?2", name):
        return "v2"
    return None


def _mismatch(model: str, implied: str, version: str) -> str:
    return (f"model {model!r} looks like Typhoon OCR {implied}, but version={version!r}: "
            f"the {version} prompt will be sent to it")


def _is_typhoon_api(base_url: str) -> bool:
    return urlparse(base_url).hostname == urlparse(TYPHOON_API_URL).hostname


def _remote_http_host(base_url: str) -> Optional[str]:
    """The host of a plain-http URL that is not this machine (localhost, *.localhost, 127.x.x.x, ::1, and 0.0.0.0
    or ::, which servers print as their address), else None."""
    u = urlparse(base_url)
    host = (u.hostname or "").rstrip(".")
    if u.scheme != "http" or host == "localhost" or host.endswith(".localhost"):
        return None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    return None if ip.is_loopback or ip.is_unspecified else host


def _finished(futures, return_when: str) -> set:
    """concurrent.futures.wait(futures, return_when=return_when).done, waiting half a second at a time: Ctrl-C takes
    effect in this thread only when it runs Python code, and a wait without a time limit lasts until a page is done
    where the signal does not interrupt it (Windows, a signal that reached another thread, _thread.interrupt_main)."""
    while True:
        done, not_done = wait(futures, timeout=0.5, return_when=return_when)
        if not not_done or (done and return_when == FIRST_COMPLETED):
            return done


def _backoff(attempt: int) -> float:
    return random.uniform(0, min(_BACKOFF_MAX, _BACKOFF_BASE * 2 ** min(attempt, 16)))   # 2 ** 1024 overflows a float


def _retry_after(value: Optional[str]) -> Optional[float]:
    """Seconds to wait from a Retry-After header (seconds or an HTTP date), at most 60; None if unusable."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    if not (math.isfinite(seconds) and seconds >= 0):
        return None
    return min(seconds, _RETRY_AFTER_MAX)


def _retry_at(text: str) -> Optional[float]:
    """Seconds until the time in an answer that says "retry after <ISO 8601 time>" (the Typhoon API's 429), at
    most 60; None if there is none or it has passed (a clock that is ahead)."""
    m = _RETRY_AT.search(text or "")
    if not m:
        return None
    zone = (m.group(2) or "Z").upper().replace("Z", "+00:00")
    try:
        when = datetime.fromisoformat(m.group(1) + (zone if ":" in zone else zone[:3] + ":" + zone[3:]))
    except ValueError:
        return None
    seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return min(seconds, _RETRY_AFTER_MAX) if seconds > 0 else None


def _names_template_kwargs(answer: httpx.Response) -> bool:
    """True for an answer that names chat_template_kwargs, as a gateway that rejects the field does: _generate() sends
    the page again without it at once, as 0.1.0 did, instead of retrying it with the field."""
    return "chat_template_kwargs" in answer.text[:500]


def _content(message: dict, url: str, answer) -> str:
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise TyphoonOCRError(f"unexpected answer from {url}: {_excerpt(answer)}")
    return content


def _excerpt(answer) -> str:
    return json.dumps(answer, ensure_ascii=False)[:300]


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
