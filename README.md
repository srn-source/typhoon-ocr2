# typhoon-ocr2

Read Thai and English documents (PDFs, images and multi-page TIFFs) with **Typhoon OCR**. For every page you get:

- the page as **Markdown** (tables as HTML, formulas as LaTeX), and
- with Typhoon OCR V2, the **layout elements** in reading order: each one has a category (title, text, table, …), a bounding box and its text.

Links: [Typhoon OCR V2 2B](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b) · [Typhoon OCR V2 0.8B](https://huggingface.co/typhoon-ai/typhoon-ocr2-0.8b) · [Playground](https://playground.opentyphoon.ai/ocr) · [Typhoon API docs](https://docs.opentyphoon.ai/)

This package continues [scb-10x/typhoon-ocr](https://github.com/scb-10x/typhoon-ocr) (`pip install typhoon-ocr`) with the Typhoon team's permission. It has a new name, so code that uses `typhoon-ocr` keeps working.

## Install

```bash
pip install typhoon-ocr2
```

PDF support is included; you do not need to install Poppler.

## Quick start: Typhoon API (no GPU needed)

```python
import os
from typhoon_ocr2 import TyphoonOCR

os.environ["TYPHOON_API_KEY"] = "YOUR_KEY"       # your Typhoon API key (or pass api_key="YOUR_KEY" below)

ocr = TyphoonOCR(
    base_url="https://api.opentyphoon.ai/v1",    # the Typhoon API
    version="v2",                                # Typhoon OCR V2 ("v1.5" for Typhoon OCR 1.5)
    upscale=True,                                # enlarge images under 1,600 px (recommended; see Settings)
)

pages = ocr.parse("report.pdf", pages="1-3")     # a PDF, an image or a TIFF; pages: "1-3", 2, [1, 4] or None for all
for page in pages:
    print(f"--- page {page.page}")
    print(page.markdown)
```

## Typhoon OCR V2 or 1.5

Switch with `version`; the rest of the code stays the same.

```python
ocr_v15 = TyphoonOCR(
    base_url="https://api.opentyphoon.ai/v1",
    version="v1.5",
    model="typhoon-ocr",                         # Typhoon OCR 1.5's name in the Typhoon API documentation
)
print(ocr_v15.parse("page.jpg")[0].markdown)
```

| | `version="v2"` | `version="v1.5"` |
|---|---|---|
| model used on the Typhoon API when `model` is not given | `typhoon-ocr-v2` | `typhoon-ocr-v1.5`; the API documentation lists Typhoon OCR 1.5 as `typhoon-ocr`, so pass `model="typhoon-ocr"` |
| `page.markdown` | yes | yes |
| `page.elements` (category, box, text) | yes | no (empty list) |
| `ask()` / `extract()` | yes | not tested |

**`version` or `model`?** `version` tells the package which prompt to send and how to read the answer. `model` is the model's name on the server. On the Typhoon API, `version="v2"` is enough: the package picks the matching model for you; for 1.5, pass `model="typhoon-ocr"` as above (a 1.5 model name also selects `version="v1.5"`). You need `model` when you run the model yourself (next section), because there the name depends on how you started the server.

## Run the model yourself

Start a server, then point `base_url` and `model` at it:

```bash
# vLLM 0.30.0 or newer (GPU)
vllm serve typhoon-ai/typhoon-ocr2-2b --max-model-len 40960 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 2048, "max_pixels": 16777216}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'

# Ollama 0.40.15 or newer
ollama pull SuraponOCR/typhoon-ocr2-2b

# llama.cpp (GGUF)
llama-server -hf typhoon-ai/typhoon-ocr2-2b-GGUF:Q8_0 --jinja --reasoning off -c 40960 --image-max-tokens 16384 -ngl 999 --port 8080
```

- vLLM: the package also sends `enable_thinking: false` with every V2 request, unless a gateway in between has rejected that field; the server flag keeps thinking off when a gateway drops or rejects it.
- vLLM, faster: each model has an optional EAGLE-3 draft head for speculative decoding, [typhoon-ocr2-2b-eagle3](https://huggingface.co/typhoon-ai/typhoon-ocr2-2b-eagle3) and [typhoon-ocr2-0.8b-eagle3](https://huggingface.co/typhoon-ai/typhoon-ocr2-0.8b-eagle3). Add `--speculative-config '{"method": "eagle3", "model": "typhoon-ai/typhoon-ocr2-2b-eagle3", "num_speculative_tokens": 3}'` to `vllm serve` (tested with vLLM 0.33.0). Generation was 1.9 to 2.5 times faster for the 2B model and 1.5 to 1.7 times for the 0.8B model in our tests. Nothing changes on the package side: `base_url` and `model` stay the same.
- Ollama runs these models on its MLX engine. On Linux with an NVIDIA GPU this needs Ollama's MLX runner, `ollama-linux-amd64-mlx` from the [Ollama releases](https://github.com/ollama/ollama/releases); the standard install script does not include it. Only Linux with an NVIDIA GPU was tested.
- llama.cpp: current versions of `llama-server` listen on port 9931 by default; `--port 8080` keeps the address below. They also answer 4 requests at once that share the `-c` context, and the package sends 4 pages at once: pages with long answers can then run out of context together, and the server fails every request in progress. As the model card says, give each request its room with `-np 4 -c 163840` (`-np N`, and `-c` times N), or serve one page at a time with `-np 1` and `max_workers=1` (`--max-workers 1`).

| Server | `base_url` | `model` |
|---|---|---|
| vLLM | `"http://localhost:8000/v1"` | `"typhoon-ai/typhoon-ocr2-2b"` |
| Ollama | `"http://localhost:11434"` | `"SuraponOCR/typhoon-ocr2-2b"` |
| llama.cpp | `"http://localhost:8080/v1"` | `"typhoon-ocr2"` (llama-server serves one model; any name works) |

```python
ocr = TyphoonOCR(
    base_url="http://localhost:8000/v1",
    model="typhoon-ai/typhoon-ocr2-2b",
    version="v2",
    upscale=True,
)
```

For the smaller page parser, use `typhoon-ai/typhoon-ocr2-0.8b` (Ollama: `SuraponOCR/typhoon-ocr2-0.8b`). On vLLM older than 0.30.0, add `--enforce-eager` to `vllm serve`: in the default compiled mode of vLLM 0.27.1, accuracy dropped by about 9 points on MDPBench. Other OpenAI-compatible servers work the same way.

## Inputs

`parse()`, `ask()` and `extract()` take a file path, the file's bytes or a PIL image:

- **PDFs**: each page is rendered with a long side of 2,048 px (`pdf_long_side`), filled-in form fields included.
- **Images** (PNG, JPEG, …): photos are turned upright by their EXIF orientation. 16-bit greyscale scans whose ink is above 8-bit values are scaled to 8 bits. A transparent area is sent with the colour stored under it, often black: put an image with black text on a transparent background on white first (for example `Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))`).
- **Multi-page TIFFs** (scanner and fax output): each frame is a page; thumbnails stored in the file are skipped. If the file's list of pages is damaged partway, the pages before the damage are read, with a warning.
- **PIL images** are sent as they are. Only an image straight from `Image.open()` is also turned upright; if you rotated or changed it yourself, it is sent as you made it (use `ImageOps.exif_transpose` to turn it upright).

`pages` picks PDF pages or TIFF frames, counting from 1: `2`, `"1-3,5"`, `[1, 4]` or `range(1, 4)`; the default, `None`, is every page. A page listed twice is parsed once, and a selection with no page in it, such as `""` or `[]`, raises `ValueError` for a PDF or multi-page TIFF (an image, which has one page, is parsed as with `None`). Limitation: very long pages, such as receipt rolls, also get a 2,048 px long side and may come out too narrow to read; set a larger `pdf_long_side` in `TyphoonOCR(...)` or crop the page into parts.

## What you get back

`ocr.parse()` returns one `PageResult` per page:

| Field | Meaning |
|---|---|
| `page.page` | page number (1-based) |
| `page.markdown` | the page as Markdown; by default without headers, footers and page footnotes (see below) |
| `page.elements` | V2 only: **every** element the model returned, as `{"bbox": [x1, y1, x2, y2], "category": "...", "text": "..."}`, in reading order |
| `page.image_size` | `(width, height)` of the page image the boxes refer to |
| `page.truncated` | True if the output stopped at the token limit (usually a repetition loop); `elements` keeps the complete ones |
| `page.raw` | the model's raw output |
| `page.parse_error` | V2 only: why the output could not be read as layout elements (then `elements` is empty and `markdown` is `""`); `None` when it could |

`PageResult` is a dataclass. `parse_error` is its seventh field, added after 0.1.0, so `dataclasses.asdict(page)` also has a `"parse_error"` key and `dataclasses.astuple(page)` has seven items.

### Headers, footers and other categories

The model always returns header, footer and page_footnote elements, and they are always in `page.elements`. Only `page.markdown` leaves them out by default, because that is how the model was evaluated (and running page numbers and headers usually get in the way of reading). You choose what the Markdown keeps:

```python
pages = ocr.parse("report.pdf")                                   # default: no header, footer, page_footnote
pages = ocr.parse("report.pdf", keep_header_footer=True)          # keep every element
pages = ocr.parse("report.pdf", exclude=["footer"])               # keep headers, drop footers only
pages = ocr.parse("report.pdf", exclude=["header", "footer", "page_footnote", "figure_caption"])  # your own list

footers = [e["text"] for e in pages[0].elements if e["category"] == "footer"]   # the elements themselves
```

On the command line: `--keep-header-footer`, or `--exclude footer` (several names allowed; `--exclude` with no names keeps everything).

The 13 categories are header, title, text, figure, table, formula, figure_caption, table_caption, formula_caption, figure_footnote, table_footnote, page_footnote and footer. Boxes use a 0–1000 grid relative to the page, so they do not depend on the image size:

```python
from typhoon_ocr2 import draw_layout, load_pages, to_pixels

image = load_pages("report.pdf", pages=[1])[0]
first = pages[0].elements[0]
print(first["category"], to_pixels(first["bbox"], *image.size))   # box in pixels
draw_layout(image, pages[0].elements).save("page1_layout.png")     # boxes numbered in reading order
```

### Errors and partial results

A request that gets HTTP 429, 500, 502, 503 or 504, or whose connection fails, is retried up to `max_retries` times (default 3), waiting as long as the server asks (its `Retry-After`, or the time in the Typhoon API's rate-limit message; up to 60 s). If a long PDF still runs into the Typhoon API's rate limit (20 requests a minute), use `max_workers=1` or a larger `max_retries`. If a page still fails, `parse()` sends no further pages, lets the pages already sent finish, and raises `PartialParseError` (a `TyphoonOCRError`) that keeps the parsed pages:

```python
from typhoon_ocr2 import PartialParseError

try:
    pages = ocr.parse("report.pdf")
except PartialParseError as e:
    pages = e.results            # the pages that were parsed, in the order requested
    print(e.errors)              # {page number: the exception}
    print(e.skipped)             # pages that were never sent
```

A page whose output is not layout JSON (for example, the model answered in prose) does not raise: `page.parse_error` says why and `page.raw` keeps the answer. Wrong arguments (`pages`, and `exclude`, also with `version="v1.5"`, which does not use it), a missing file, a file that cannot be opened (for example, a PDF with a password) and a first page that cannot be read raise `ValueError` or `OSError` before anything is sent; an image of more than twice Pillow's limit (`PIL.Image.MAX_IMAGE_PIXELS`, so about 179 megapixels) raises Pillow's `DecompressionBombError`, and between the limit and twice it Pillow only warns (`DecompressionBombWarning`) and the page is sent. A later page that cannot be read (a PDF or TIFF damaged in the middle) is reported like a failed page: `PartialParseError`, with the `ValueError` or `OSError` in `e.errors`, its number in `e.unreadable`, and the pages before it in `e.results`.

## Questions and key information (Typhoon OCR V2 2B)

```python
answer = ocr.ask("receipt.jpg", "What is the total amount?")      # plain text
fields = ocr.extract("receipt.jpg", ["Date", "Total", "VAT 7%"])   # dict with these keys
everything = ocr.extract("receipt.jpg")                            # every field; the model picks the keys
```

The 0.8B model is a page parser only; it is much weaker at questions and extraction. Check extracted values before you use them: key-information extraction is the 2B model's weakest task.

## Command line

```bash
export TYPHOON_API_KEY=YOUR_KEY
API=https://api.opentyphoon.ai/v1

typhoon-ocr2 report.pdf --base-url $API                                  # Markdown of every page (V2)
typhoon-ocr2 report.pdf --base-url $API --pages 2-4 --format json        # layout elements
typhoon-ocr2 receipt.jpg --base-url $API --ask "What is the total amount?"
typhoon-ocr2 receipt.jpg --base-url $API --extract Date Total
typhoon-ocr2 page.jpg --base-url $API --model typhoon-ocr                # Typhoon OCR 1.5
typhoon-ocr2 page.jpg --base-url $API --model typhoon-ocr --figure-language English   # figures described in English
typhoon-ocr2 report.pdf --base-url $API --max-workers 1 --timeout 1200   # one page at a time, longer wait
typhoon-ocr2 page.jpg --base-url http://localhost:11434 --model SuraponOCR/typhoon-ocr2-2b   # Ollama
```

To avoid repeating `--base-url`, set `TYPHOON_OCR_BASE_URL` (and `TYPHOON_OCR_MODEL` for `--model`). Add `--no-upscale` to turn upscaling off.

`--pages` picks PDF pages or TIFF frames; `--ask` and `--extract` read the first one. `--exclude` takes the 13 category names listed above. `--max-workers N` sets how many pages are sent at once (default: 1 on Ollama, 4 elsewhere), `--timeout SECONDS` how long to wait for each answer (default: 600), and `--figure-language` the language of Typhoon OCR 1.5 figure descriptions (default: Thai). `typhoon-ocr2 --help` lists every option.

Standard output that the console's or file's encoding cannot hold is written as UTF-8 instead, for example Thai text redirected to a file on Windows with an English locale (cp1252); other output keeps that encoding. Messages on stderr always keep it, with `\uXXXX` escapes for characters it cannot show. If a page still fails after the retries, the pages that were parsed are written out anyway, stderr names the failed and unsent pages with the `--pages` value that parses them again (leaving out pages that cannot be read from the file), and the exit code is 1. A page whose output is not layout JSON only gets a warning on stderr; with `--format json` it also gets a `"raw"` key holding the model's answer.

## Settings

All settings of `TyphoonOCR(...)`. The defaults are the settings the models were evaluated with.

| Setting | Default | What it does |
|---|---|---|
| `base_url` | `$TYPHOON_OCR_BASE_URL`, else `"http://localhost:8000/v1"` | server address |
| `model` | `$TYPHOON_OCR_MODEL`; else on the Typhoon API, chosen from `version`; on Ollama, `"SuraponOCR/typhoon-ocr2-2b"`; on other servers, the Typhoon OCR model they serve (when they serve several, the first of `version`, V2 by default) | model name on the server; needed only for your own server |
| `api_key` | `$TYPHOON_API_KEY`, else `$TYPHOON_OCR_API_KEY`, else `$OPENAI_API_KEY` | key for the Typhoon API or another server that needs one (spaces and line breaks at its end are dropped). It is sent to `base_url`, whatever server that is; a warning says when it would go over plain http to another machine |
| `version` | `"v2"` (or follows the model name, also one found on the server) | `"v2"` or `"v1.5"`: the prompt and output format |
| `upscale` | `True` | enlarge images whose long side is under 1,600 px. On ThaiOCRBench this raised Typhoon OCR V2 2B from 59.9 to 62.9 and 0.8B from 37.3 to 41.3; large pages are not changed. `False` turns it off; a number sets another size |
| `max_tokens` | `16384` | output limit per page |
| `pdf_long_side` | `2048` | PDF pages are rendered with this long side, in pixels |
| `max_workers` | `4` (`1` on Ollama) | pages sent in parallel. Ollama answers one request at a time unless `OLLAMA_NUM_PARALLEL` is set |
| `max_retries` | `3` | retries after HTTP 429, 500, 502, 503 or 504 or a failed or dropped connection. Waits as the server asks (`Retry-After`, or the time in the Typhoon API's 429 message; up to 60 s), else 0–1 s, 0–2 s, 0–4 s, … (at most 30 s). Timeouts are not retried: the server may still be working on the page, or could not be reached within `timeout` |
| `timeout` | `600` | seconds to wait for the answer to one HTTP request, including the time it waits in the server's queue |
| `backend` | `"auto"` | `"openai"` or `"ollama"`; auto picks Ollama for `http://localhost:11434` |
| `figure_language` | `"Thai"` | v1.5 only: language of figure descriptions |

The prompts are the ones the models were trained with (`typhoon_ocr2.PARSE_PROMPT` for V2), and V2 decoding is greedy with thinking turned off (Typhoon OCR 1.5 uses temperature 0.1, top_p 0.6 and repetition_penalty 1.1, as in the original package).

Several threads can parse PDFs at the same time, for example with one `TyphoonOCR` each: PDFium, the PDF renderer, is not thread-safe, so the package renders under a lock.

## Coming from `typhoon-ocr`

| `typhoon-ocr` | `typhoon-ocr2` |
|---|---|
| `from typhoon_ocr import ocr_document` | `from typhoon_ocr2 import ocr_document` (same idea), or `TyphoonOCR` for more |
| `ocr_document(path, page_num=1, base_url=..., model=...)` | `ocr_document(path, pages=1, base_url=..., model=...)` |
| one page per call: `page_num` defaults to 1 | **every page** by default, sent in parallel: pass `pages=1` for the old behaviour, or e.g. `pages="1-3,5"` |
| Typhoon OCR 1.5 by default: `task_type="v1.5"`, model `typhoon-ocr` | **Typhoon OCR V2** by default. For 1.5 on the Typhoon API, pass `model="typhoon-ocr"` as before (CLI: `--model typhoon-ocr`): a 1.5 model name selects `version="v1.5"`, which is what `task_type="v1.5"` becomes |
| server: `$TYPHOON_BASE_URL`, else the Typhoon API | `$TYPHOON_OCR_BASE_URL`, else `http://localhost:8000/v1`; `$TYPHOON_BASE_URL` is not read. For the Typhoon API, pass `base_url="https://api.opentyphoon.ai/v1"` |
| API key: `$TYPHOON_OCR_API_KEY`, then `$TYPHOON_API_KEY`, then `$OPENAI_API_KEY` | `$TYPHOON_API_KEY`, then `$TYPHOON_OCR_API_KEY`, then `$OPENAI_API_KEY`: if two hold different keys, pass `api_key=` |
| a base64 string instead of a file path | not accepted: pass the bytes, `base64.b64decode(s)` |
| `pdf_or_image_path=`, `target_image_dim=`, `target_text_length=` | removed: pass the file as the first argument; the package sizes the pages for the model |
| returns a Markdown string | `TyphoonOCR.parse()` also returns layout elements with boxes |
| needs Poppler for PDFs | no system dependency |

## License

Apache-2.0. See [LICENSE](https://github.com/srn-source/typhoon-ocr2/blob/master/LICENSE) and [NOTICE](https://github.com/srn-source/typhoon-ocr2/blob/master/NOTICE). Use of the Typhoon models is subject to the [OpenTyphoon Terms and Conditions](https://opentyphoon.ai/tac).
