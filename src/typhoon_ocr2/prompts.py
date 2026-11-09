"""Prompts the Typhoon OCR models were trained with. Use them exactly as written."""

# Typhoon OCR V2: full-page parsing into layout elements (the doc2json prompt from Infinity-Parser2).
PARSE_PROMPT = """- Extract layout information from the provided PDF image.
- For each layout element, output its bbox, category, and the text content within the bbox.
- Bbox format: [x1, y1, x2, y2].
- Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula', 'figure_caption', 'table_caption', 'formula_caption', 'figure_footnote', 'table_footnote', 'page_footnote', 'footer'].
- Text extraction and formatting:
  1) For 'figure', the text field must be an empty string.
  2) For 'formula', format text as LaTeX.
  3) For 'table', format text as HTML.
  4) For all other categories (e.g., text, title), format text as Markdown.
- The output text must be exactly the original text from the image, with no translation or rewriting.
- Sort all layout elements in human reading order.
- Final output must be a single JSON object."""

# Typhoon OCR V2 2B: extract every field as JSON without naming the fields.
EXTRACT_ALL_PROMPT = 'Perform open-schema key information extraction (WYSIWYG) on this image: one JSON object, printed labels as keys, values copied exactly, "" for blank fields, arrays for repeated rows. Return only JSON.'

# Typhoon OCR 1.5: full-page Markdown. `figure_language` is "Thai" or "English".
V15_PROMPT = """Extract all text from the image.


Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.


Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:


<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in {figure_language}.
</figure>


- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes.
    """


def v15_prompt(figure_language: str = "Thai") -> str:
    if figure_language not in ("Thai", "English"):
        raise ValueError("figure_language must be 'Thai' or 'English'")
    return V15_PROMPT.format(figure_language=figure_language)


def extract_fields_prompt(fields) -> str:
    """Prompt for extracting named fields as one JSON object (Typhoon OCR V2 2B)."""
    keys = "\n".join(str(f) for f in fields)
    return (
        "Extract the following fields from the image and return one JSON object with exactly these keys. "
        "Copy values exactly as they appear; use \"\" for fields that are missing or blank. Return only JSON.\n" + keys
    )
