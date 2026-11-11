"""Typhoon OCR V2: Thai and English document parsing with layout, from images and PDFs.

    >>> from typhoon_ocr2 import TyphoonOCR
    >>> ocr = TyphoonOCR(base_url="http://localhost:8000/v1", model="typhoon-ai/typhoon-ocr2-2b")   # vLLM
    >>> # Ollama: TyphoonOCR(base_url="http://localhost:11434"); llama.cpp: base_url="http://localhost:8080/v1"
    >>> page = ocr.parse("report.pdf", pages=1)[0]
    >>> page.elements[0]
    {'bbox': [62, 48, 938, 96], 'category': 'title', 'text': 'Annual Report 2025'}
    >>> print(page.markdown)
"""
from .client import PageResult, PartialParseError, TyphoonOCR, TyphoonOCRError, ocr_document
from .images import iter_pages, load_pages, render_pdf_pages, upscale_min_long
from .output import CATEGORIES, draw_layout, parse_elements, parse_json, to_markdown, to_pixels, validate_exclude
from .prompts import EXTRACT_ALL_PROMPT, PARSE_PROMPT, V15_PROMPT

__version__ = "0.2.0"

__all__ = [
    "TyphoonOCR", "PageResult", "TyphoonOCRError", "PartialParseError", "ocr_document",
    "parse_elements", "to_markdown", "to_pixels", "draw_layout", "parse_json", "validate_exclude", "CATEGORIES",
    "iter_pages", "load_pages", "render_pdf_pages", "upscale_min_long",
    "PARSE_PROMPT", "EXTRACT_ALL_PROMPT", "V15_PROMPT",
]
