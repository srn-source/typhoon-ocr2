"""Command line: typhoon-ocr2 FILE [options]"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
import warnings

from PIL import Image

from . import __version__
from . import images as im
from .client import PartialParseError, TyphoonOCR, TyphoonOCRError
from .output import CATEGORIES


def _pages_json(results) -> str:
    """Pretty JSON with one element per line, so long pages stay readable. A page whose output is not layout
    JSON also gets "raw", the model's answer."""
    pages = []
    for r in results:
        els = ",\n".join("      " + json.dumps(e, ensure_ascii=False) for e in r.elements)
        fields = [f'"page": {r.page}, "width": {r.image_size[0]}, "height": {r.image_size[1]}',
                  f'"truncated": {json.dumps(r.truncated)}',
                  f'"elements": [\n{els}\n    ]' if r.elements else '"elements": []',
                  f'"markdown": {json.dumps(r.markdown, ensure_ascii=False)}']
        if r.parse_error:
            fields.append(f'"raw": {json.dumps(r.raw, ensure_ascii=False)}')
        pages.append("  {\n" + ",\n".join("    " + f for f in fields) + "\n  }")
    return "[\n" + ",\n".join(pages) + "\n]"


def _show(results, fmt: str) -> None:
    """Warnings for the pages on stderr, then the pages on stdout."""
    for r in results:
        if r.truncated:
            print(f"warning: page {r.page} stopped at the token limit"
                  + ("; only complete elements are kept" if r.elements else ""), file=sys.stderr)
        if r.parse_error:
            print(f"warning: page {r.page}: the model output is not layout JSON; nothing was extracted",
                  file=sys.stderr)
    _print(_pages_json(results) if fmt == "json" else "\n\n".join(r.markdown for r in results))


def _print(text: str) -> None:
    """print() to stdout. When its encoding cannot hold the text (Thai in cp1252, which Windows uses for a
    redirected stdout before Python 3.15, or ☑ in cp874), stdout switches to UTF-8 first instead of failing.
    Text that fits is written as before."""
    out = sys.stdout
    try:
        text.encode(out.encoding or "utf-8", out.errors or "strict")
    except (UnicodeEncodeError, LookupError):
        try:
            out.reconfigure(encoding="utf-8", errors=out.errors)
        except (AttributeError, ValueError):        # not a stream that can switch
            pass
    except (AttributeError, TypeError):             # no stream (pythonw) or no encoding: let print() decide
        pass
    print(text)


def _show_warning(default, message, category, *args, **kwargs) -> None:
    """UserWarnings, such as the package's own (e.g. an API key over plain http), in the command's "warning: ..."
    form; other warnings go to `default`, the hook that was there before."""
    if issubclass(category, UserWarning):
        print(f"warning: {message}", file=sys.stderr)
    else:
        default(message, category, *args, **kwargs)


def _report(e: PartialParseError) -> None:
    """Which pages failed or were not sent, and how to parse them again. Pages that cannot be read from the file
    are left out of the --pages hint: they would fail again."""
    sys.stdout.flush()                      # the parsed pages first, when both go to the same place
    for n, error in e.errors.items():
        print(f"error: page {n} failed: {error}", file=sys.stderr)
    if e.skipped:
        print(f"error: page(s) not sent: {_ranges(e.skipped)}", file=sys.stderr)
    unreadable = sorted(n for n in e.unreadable if n in e.errors)
    again = sorted(n for n in [*e.errors, *e.skipped] if n not in unreadable)
    notes = []
    if again and (e.results or unreadable):
        notes.append(f"to parse the missing ones, run again with --pages {_ranges(again)}")
    if unreadable:
        notes.append(f"page(s) {_ranges(unreadable)} cannot be read from the file")
    if notes:
        parsed = f"{len(e.results)} page(s) were parsed and written out" if e.results else "no page was parsed"
        print(f"error: {parsed}; " + "; ".join(notes), file=sys.stderr)


def _ranges(numbers) -> str:
    """[1, 2, 3, 5] -> "1-3,5", the --pages format."""
    runs = []
    for n in numbers:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _above_zero(cast):
    """argparse type: a number above 0, read with `cast`."""
    def read(text: str):
        value = cast(text)                  # a ValueError makes argparse say "invalid int value: 'x'"
        if not 0 < value < float("inf"):
            raise argparse.ArgumentTypeError(f"must be above 0, not {text}")
        return value
    read.__name__ = cast.__name__
    return read


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="typhoon-ocr2", description="Parse a PDF or image with Typhoon OCR.")
    ap.add_argument("file", help="PDF or image (a multi-page TIFF is read page by page)")
    ap.add_argument("--pages", help="PDF pages or TIFF frames, e.g. 1-3,5 (default: all; --ask and --extract "
                                    "read the first one)")
    ap.add_argument("--base-url", help="server URL (default: $TYPHOON_OCR_BASE_URL or http://localhost:8000/v1; "
                                       "Typhoon API: https://api.opentyphoon.ai/v1; Ollama: http://localhost:11434)")
    ap.add_argument("--model", help="served model name (default: $TYPHOON_OCR_MODEL; else on the Typhoon API "
                                    "typhoon-ocr-v2 or typhoon-ocr-v1.5, following --version; on Ollama "
                                    "SuraponOCR/typhoon-ocr2-2b; on other servers the Typhoon OCR model they "
                                    "serve, preferring --version, V2 by default)")
    ap.add_argument("--api-key", help="API key (default: $TYPHOON_API_KEY, then $TYPHOON_OCR_API_KEY, "
                                      "then $OPENAI_API_KEY)")
    ap.add_argument("--backend", choices=["auto", "openai", "ollama"], default="auto",
                    help="server API (default: auto, Ollama for port 11434 without /v1, else OpenAI-compatible)")
    ap.add_argument("--version", dest="model_version", choices=["v2", "v1.5"], default=None,
                    help="Typhoon OCR version (default: follows the model name; v2 otherwise)")
    ap.add_argument("--format", choices=["md", "json"], default="md",
                    help="md: Markdown; json: one object per page with the layout elements")
    ap.add_argument("--keep-header-footer", action="store_true", help="keep headers, footers and page footnotes in Markdown")
    ap.add_argument("--exclude", nargs="*", choices=CATEGORIES, metavar="CATEGORY",
                    help="categories to leave out of the Markdown, e.g. --exclude footer page_footnote "
                         "(default: header footer page_footnote; --exclude with no names keeps everything); "
                         "one of: %(choices)s")
    ap.add_argument("--ask", metavar="QUESTION", help="ask a question about the page instead of parsing it (2B)")
    ap.add_argument("--extract", nargs="*", metavar="FIELD",
                    help="extract key information as JSON (2B); list fields, or none to let the model choose")
    ap.add_argument("--no-upscale", action="store_true", help="do not enlarge images under 1,600 px (V2 only)")
    ap.add_argument("--figure-language", choices=["Thai", "English"], default="Thai",
                    help="language of figure descriptions (v1.5 only; default: Thai)")
    ap.add_argument("--max-workers", type=_above_zero(int), metavar="N",
                    help="pages sent to the server at once (default: 1 on Ollama, 4 elsewhere)")
    ap.add_argument("--timeout", type=_above_zero(float), default=600.0, metavar="SECONDS",
                    help="seconds to wait for each answer, including the time in the server's queue (default: 600)")
    ap.add_argument("-V", "--package-version", action="version", version=f"typhoon-ocr2 {__version__}")
    a = ap.parse_args(argv)

    ocr = None
    with warnings.catch_warnings():
        warnings.showwarning = functools.partial(_show_warning, warnings.showwarning)
        try:
            ocr = TyphoonOCR(base_url=a.base_url, model=a.model, api_key=a.api_key, backend=a.backend,
                             version=a.model_version, upscale=not a.no_upscale, max_workers=a.max_workers,
                             timeout=a.timeout, figure_language=a.figure_language)
            if a.ask or a.extract is not None:
                pages = list(itertools.islice(im._selection(a.pages), 2)) if a.pages is not None else [1]
                if len(pages) > 1:
                    print(f"warning: --ask and --extract read one page: page {pages[0]}", file=sys.stderr)
                if a.ask:
                    _print(ocr.ask(a.file, a.ask, pages[0]))
                else:
                    _print(json.dumps(ocr.extract(a.file, a.extract or None, pages[0]), ensure_ascii=False, indent=2))
            else:
                _show(ocr.parse(a.file, a.pages, a.keep_header_footer, a.exclude), a.format)
        except PartialParseError as e:      # a page failed after the retries: keep the pages that were parsed
            if e.results:
                _show(e.results, a.format)
            _report(e)
            return 1
        except (TyphoonOCRError, ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except Image.DecompressionBombError as e:   # over twice Image.MAX_IMAGE_PIXELS (Pillow only warns below)
            print(f"error: {str(e).rstrip('.')}; scale the image down or cut it into parts first", file=sys.stderr)
            return 1
        finally:
            if ocr is not None:
                ocr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
