"""Command line: typhoon-ocr2 FILE [options]"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .client import TyphoonOCR, TyphoonOCRError


def _pages_json(results) -> str:
    """Pretty JSON with one element per line, so long pages stay readable."""
    pages = []
    for r in results:
        els = ",\n".join("      " + json.dumps(e, ensure_ascii=False) for e in r.elements)
        pages.append("  {\n"
                     f'    "page": {r.page}, "width": {r.image_size[0]}, "height": {r.image_size[1]},\n'
                     f'    "truncated": {json.dumps(r.truncated)},\n'
                     f'    "elements": [\n{els}\n    ],\n' if r.elements else
                     "  {\n"
                     f'    "page": {r.page}, "width": {r.image_size[0]}, "height": {r.image_size[1]},\n'
                     f'    "truncated": {json.dumps(r.truncated)},\n'
                     '    "elements": [],\n')
        pages[-1] += f'    "markdown": {json.dumps(r.markdown, ensure_ascii=False)}\n  }}'
    return "[\n" + ",\n".join(pages) + "\n]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="typhoon-ocr2", description="Parse a PDF or image with Typhoon OCR.")
    ap.add_argument("file", help="PDF or image")
    ap.add_argument("--pages", help="PDF pages, e.g. 1-3,5 (default: all)")
    ap.add_argument("--base-url", help="server URL (default: $TYPHOON_OCR_BASE_URL or http://localhost:8000/v1; "
                                       "Typhoon API: https://api.opentyphoon.ai/v1; Ollama: http://localhost:11434)")
    ap.add_argument("--model", help="served model name (default: the server's first model; Ollama: SuraponOCR/typhoon-ocr2-2b)")
    ap.add_argument("--api-key", help="API key (default: $TYPHOON_OCR_API_KEY / $TYPHOON_API_KEY)")
    ap.add_argument("--backend", choices=["auto", "openai", "ollama"], default="auto")
    ap.add_argument("--version", dest="model_version", choices=["v2", "v1.5"], default=None,
                    help="Typhoon OCR version (default: follows --model; v2 otherwise)")
    ap.add_argument("--format", choices=["md", "json"], default="md",
                    help="md: Markdown; json: one object per page with the layout elements")
    ap.add_argument("--keep-header-footer", action="store_true", help="keep headers, footers and page footnotes in Markdown")
    ap.add_argument("--exclude", nargs="*", metavar="CATEGORY",
                    help="categories to leave out of the Markdown, e.g. --exclude footer page_footnote "
                         "(default: header footer page_footnote; --exclude with no names keeps everything)")
    ap.add_argument("--ask", metavar="QUESTION", help="ask a question about the page instead of parsing it (2B)")
    ap.add_argument("--extract", nargs="*", metavar="FIELD",
                    help="extract key information as JSON (2B); list fields, or none to let the model choose")
    ap.add_argument("--no-upscale", action="store_true", help="do not enlarge images under 1,600 px")
    ap.add_argument("-V", "--package-version", action="version", version=f"typhoon-ocr2 {__version__}")
    a = ap.parse_args(argv)

    ocr = TyphoonOCR(base_url=a.base_url, model=a.model, api_key=a.api_key, backend=a.backend,
                     version=a.model_version, upscale=not a.no_upscale)
    page = int(a.pages.split(",")[0].split("-")[0]) if a.pages else 1
    try:
        if a.ask:
            print(ocr.ask(a.file, a.ask, page))
        elif a.extract is not None:
            print(json.dumps(ocr.extract(a.file, a.extract or None, page), ensure_ascii=False, indent=2))
        else:
            results = ocr.parse(a.file, a.pages, a.keep_header_footer, a.exclude)
            for r in results:
                if r.truncated:
                    print(f"warning: page {r.page} stopped at the token limit; only complete elements are kept",
                          file=sys.stderr)
            if a.format == "json":
                print(_pages_json(results))
            else:
                print("\n\n".join(r.markdown for r in results))
    except (TyphoonOCRError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        ocr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
