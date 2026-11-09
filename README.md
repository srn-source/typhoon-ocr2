# typhoon-ocr2

Read Thai and English documents (PDFs and images) with **Typhoon OCR**. For every page you get:

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

pages = ocr.parse("report.pdf", pages="1-3")     # a PDF or an image; pages: "1-3", 2, [1, 4] or None for all
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
)
print(ocr_v15.parse("page.jpg")[0].markdown)
```

| | `version="v2"` | `version="v1.5"` |
|---|---|---|
| model used on the Typhoon API | `typhoon-ocr-v2` | `typhoon-ocr-v1.5` |
| `page.markdown` | yes | yes |
| `page.elements` (category, box, text) | yes | no (empty list) |
| `ask()` / `extract()` | yes | not tested |

**`version` or `model`?** `version` tells the package which prompt to send and how to read the answer. `model` is the model's name on the server. On the Typhoon API you only need `version`: the package picks the matching model for you. You need `model` when you run the model yourself (next section), because there the name depends on how you started the server.

## Run the model yourself

Start a server, then point `base_url` and `model` at it:

```bash
# vLLM 0.30.0 or newer (GPU)
vllm serve typhoon-ai/typhoon-ocr2-2b --max-model-len 40960 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 2048, "max_pixels": 16777216}'

# Ollama 0.40.15 or newer
ollama pull SuraponOCR/typhoon-ocr2-2b

# llama.cpp (GGUF)
llama-server -hf typhoon-ai/typhoon-ocr2-2b-GGUF:Q8_0 --jinja --reasoning off -c 40960 --image-max-tokens 16384
```

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
typhoon-ocr2 page.jpg --base-url $API --version v1.5                     # Typhoon OCR 1.5
typhoon-ocr2 page.jpg --base-url http://localhost:11434 --model SuraponOCR/typhoon-ocr2-2b   # Ollama
```

To avoid repeating `--base-url`, set `TYPHOON_OCR_BASE_URL`. Add `--no-upscale` to turn upscaling off.

## Settings

All settings of `TyphoonOCR(...)`. The defaults are the settings the models were evaluated with.

| Setting | Default | What it does |
|---|---|---|
| `base_url` | `$TYPHOON_OCR_BASE_URL`, else `"http://localhost:8000/v1"` | server address |
| `model` | on the Typhoon API, chosen from `version`; on Ollama, `"SuraponOCR/typhoon-ocr2-2b"`; on other servers, the first model they serve | model name on the server; needed only for your own server |
| `api_key` | `$TYPHOON_API_KEY` | key for the Typhoon API |
| `version` | `"v2"` (or follows `model`, if you set one) | `"v2"` or `"v1.5"`: the prompt and output format |
| `upscale` | `True` | enlarge images whose long side is under 1,600 px. On ThaiOCRBench this raised Typhoon OCR V2 2B from 59.9 to 62.9 and 0.8B from 37.3 to 41.3; large pages are not changed. `False` turns it off; a number sets another size |
| `max_tokens` | `16384` | output limit per page |
| `pdf_long_side` | `2048` | PDF pages are rendered with this long side, in pixels |
| `max_workers` | `4` | PDF pages sent in parallel |
| `timeout` | `600` | seconds to wait for one page |
| `backend` | `"auto"` | `"openai"` or `"ollama"`; auto picks Ollama for `http://localhost:11434` |
| `figure_language` | `"Thai"` | v1.5 only: language of figure descriptions |

The prompts are the ones the models were trained with (`typhoon_ocr2.PARSE_PROMPT` for V2), and decoding is greedy with thinking turned off.

## Coming from `typhoon-ocr`

| `typhoon-ocr` | `typhoon-ocr2` |
|---|---|
| `from typhoon_ocr import ocr_document` | `from typhoon_ocr2 import ocr_document` (same idea), or `TyphoonOCR` for more |
| `ocr_document(path, page_num=1, base_url=..., model=...)` | `ocr_document(path, pages=1, base_url=..., model=...)` |
| returns a Markdown string | `TyphoonOCR.parse()` also returns layout elements with boxes |
| one page per call | `pages="1-3,5"` or all pages, sent in parallel |
| needs Poppler for PDFs | no system dependency |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Use of the Typhoon models is subject to the [OpenTyphoon Terms and Conditions](https://opentyphoon.ai/tac).
