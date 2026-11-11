"""Load pages from images and PDFs, and prepare them for the model."""
from __future__ import annotations

import base64
import contextlib
import errno
import io
import itertools
import logging
import operator
import os
import stat
import struct
import threading
import warnings
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageOps

ImageInput = Union[str, os.PathLike, bytes, Image.Image]

# Images whose long side is below this are enlarged before they are sent. On ThaiOCRBench this raised
# Typhoon OCR V2 2B from 59.9 to 62.9 and 0.8B from 37.3 to 41.3; on large pages it changes nothing.
UPSCALE_MIN_LONG = 1600
# PDF pages are rendered with this long side, as in our olmOCR-bench evaluation.
PDF_LONG_SIDE = 2048

# PDFium is not thread-safe, not even with different documents, so every call into it holds this lock. It is
# reentrant because the garbage collector can close an abandoned iter_pages() document in the middle of a render,
# in the thread that holds the lock.
_PDFIUM_LOCK = threading.RLock()
# Documents of abandoned iter_pages() generators that could not be closed without waiting for the lock: the next
# holder of the lock closes them (see _close()).
_UNCLOSED: List[object] = []
# Looked up once, here: getLogger() takes logging's module lock, which must never be waited for under _PDFIUM_LOCK.
_PDFIUM_DOC_LOG = logging.getLogger("pypdfium2._helpers.document")
_GREY16 =("I;16", "I;16L", "I;16B", "I;16N", "I")      # 16-bit greyscale (Pillow 9 opens 16-bit PNGs as "I")


def upscale_min_long(image: Image.Image, min_long: int = UPSCALE_MIN_LONG) -> Image.Image:
    """Enlarge the image so that its long side is at least `min_long` pixels, keeping the aspect
    ratio. Images that are already large enough, or empty, are returned unchanged."""
    if min(image.size) == 0:
        return image
    scale = min_long / max(image.size)
    if scale <= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def is_pdf(source: ImageInput) -> bool:
    """True for PDF bytes, and for a path that ends in .pdf or names a file starting with %PDF-."""
    if isinstance(source, (str, os.PathLike)):
        if os.fsdecode(source).lower().endswith(".pdf"):     # str() of an os.DirEntry is "<DirEntry 'x.pdf'>"
            return True
        try:
            if not os.path.isfile(source):
                return False
            with open(source, "rb") as f:
                return f.read(5) == b"%PDF-"
        except (OSError, ValueError):
            return False
    if isinstance(source, bytes):
        return source[:5] == b"%PDF-"
    return False


def pdf_page_count(source: Union[str, os.PathLike, bytes]) -> int:
    with _pdfium_locked():
        doc = _open_pdf(source)
        try:
            return len(doc)
        finally:
            doc.close()


def iter_pages(source: ImageInput, pages: Optional[Iterable[int]] = None,
               pdf_long_side: int = PDF_LONG_SIDE) -> Iterator[Tuple[int, Image.Image]]:
    """Yield (page number, RGB PIL image) for each page, one page at a time.

    PDF pages are rendered only when they are reached (long side `pdf_long_side` pixels), each frame of a
    multi-page TIFF is a page (thumbnails stored in it are not), and any other image is page 1. A PIL image is
    used as it is, turned upright only when it comes straight from Image.open(). `pages` (1-based) selects PDF
    pages or TIFF frames; every number is checked against the page count before the first page is yielded. The
    file is closed when the iterator is used up or closed."""
    numbered = _page_source(source, pages, pdf_long_side)
    next(numbered)                      # the checked page numbers
    yield from numbered


def render_pdf_pages(source: Union[str, os.PathLike, bytes], pages: Optional[Iterable[int]] = None,
                     long_side: int = PDF_LONG_SIDE) -> List[Image.Image]:
    """Render PDF pages (1-based numbers; all pages when `pages` is None) to RGB images whose long side
    is `long_side` pixels. Raises ValueError for a page out of range (before anything is rendered) and for a
    PDF that cannot be opened, e.g. a password-protected one."""
    numbered = _pdf_pages(source, pages, long_side)
    next(numbered)
    return [image for _, image in numbered]


def load_pages(source: ImageInput, pages: Optional[Iterable[int]] = None,
               pdf_long_side: int = PDF_LONG_SIDE) -> List[Image.Image]:
    """Return the pages of a PDF or multi-page TIFF, or a single image, as RGB PIL images, all at once
    (iter_pages gives them one at a time).

    `source` can be a file path, raw bytes, or a PIL image. `pages` (1-based) applies to PDFs and multi-page
    TIFFs only. Photos are turned upright by their EXIF orientation; a transparent background whose stored colour
    hides what is on it (black text on transparent black) is composited over white, while light content (white
    text) and content that shows on the stored colour keep it; and 16-bit greyscale that would come out blank is
    scaled to 8 bits."""
    return [image for _, image in iter_pages(source, pages, pdf_long_side)]


def to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def to_data_url(image: Image.Image) -> str:
    return "data:image/png;base64," + to_base64_png(image)


def parse_pages(spec: Optional[Union[str, int, Iterable[int]]]) -> Optional[List[int]]:
    """'1-3,5' -> [1, 2, 3, 5]; an int -> [int]; a list, tuple, range or other iterable of ints (or of whole
    numbers such as 2.0) -> a list; None -> None (all pages). Repeated pages are dropped, keeping the first. Raises
    ValueError for page numbers below 1, reversed or open ranges ('3-1', '-2', '1-'), and a spec that selects no
    page."""
    selected = _selection(spec)
    return None if selected is None else list(selected)


def _selection(spec, empty: bool = False) -> Optional[Iterator[int]]:
    """parse_pages(spec), checked at once but expanded page by page as it is read: a typo such as '1-1000000000'
    then gives 'page N is out of range' as soon as the page count is known, instead of a list of a billion pages.
    empty=True lets a spec that selects no page through: parse() refuses it for a PDF or multi-page TIFF only."""
    if spec is None:
        return None
    if isinstance(spec, (bytes, bytearray)):            # would be read as the codes of its characters
        raise ValueError(f"pages must be a str, an int or an iterable of ints, not {type(spec).__name__}")
    runs: List[Sequence[int]] = []
    if isinstance(spec, str):
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            first, dash, last = part.partition("-")
            if not dash:
                runs.append([_page_number(part)])
            elif not first.strip() or not last.strip():
                raise ValueError(f"page range {part!r} is open: give the first and the last page, e.g. '1-3'")
            else:
                a, b = _page_number(first), _page_number(last)
                if a > b:
                    raise ValueError(f"page range {part!r} is reversed: write it as '{b}-{a}'")
                runs.append(range(a, b + 1))
    elif isinstance(spec, range) and spec.step > 0:
        if spec:
            _page_number(spec[0])                       # the smallest number in it
        runs.append(spec)
    else:
        try:
            items = iter(spec)
        except TypeError:                   # an int, or a numpy integer
            items = iter([spec])
        runs.append([_page_number(p) for p in items])
    if not any(runs) and not empty:
        raise ValueError(f"no page selected: pages={spec!r}")
    return _once_each(runs)


def _once_each(runs: List[Sequence[int]]) -> Iterator[int]:
    if len(runs) == 1 and isinstance(runs[0], range):  # nothing repeats: no set of every page number
        yield from runs[0]
        return
    seen = set()
    for run in runs:
        for n in run:
            if n not in seen:
                seen.add(n)
                yield n


def _page_number(p) -> int:
    try:
        if isinstance(p, str):
            n = int(p)
        else:
            try:
                n = operator.index(p)
            except TypeError:               # a whole number of another type: 2.0, Decimal("2"), numpy.float32(2)
                n = int(p)
                if n != p:
                    raise ValueError from None
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{p!r} is not a page number") from None
    if n < 1:
        raise ValueError(f"page numbers start at 1, not {n}")
    return n


def _page_source(source: ImageInput, pages: Optional[Iterable[int]], long_side: int, any_page: bool = False):
    """Generator behind iter_pages. It yields the list of selected page numbers first (each checked against
    the page count), then (number, RGB image) for each page, made on demand. any_page=True lets a single
    image stand for any page number (ask() and extract())."""
    if not isinstance(source, Image.Image) and is_pdf(source):
        return _pdf_pages(source, pages, long_side)
    return _image_pages(source, pages, any_page)


def _one_page(source: ImageInput, page: int, long_side: int) -> Image.Image:
    """Page `page` of a PDF or multi-page TIFF (a whole number of another type, such as 2.0 or "2", too, as parse()
    reads it); any other image whatever `page` is."""
    try:
        page = _page_number(page)
    except ValueError:                  # an image does not use it; for a PDF or TIFF, _checked() says what is wrong
        pass
    numbered = _page_source(source, [page], long_side, any_page=True)
    try:
        next(numbered)
        return next(numbered)[1]
    finally:
        numbered.close()


def _pdf_pages(source: Union[str, os.PathLike, bytes], pages: Optional[Iterable[int]], long_side: int):
    pdfium = _pdfium()
    with _pdfium_locked():
        doc = _open_pdf(source, forms=True)
    try:
        with _pdfium_locked():
            count = len(doc)
        numbers = _checked(pages, count, "PDF")

        def render(n: int) -> Image.Image:
            with _pdfium_locked():              # released before the page is handed out
                try:
                    return _render_page(doc, n, long_side)
                except pdfium.PdfiumError as e:
                    raise ValueError(f"cannot render page {n} of the PDF{_name(source)}: {_reason(e)}") from e
        yield from _first_page_first(numbers, render)
    finally:
        _close(doc)


@contextlib.contextmanager
def _pdfium_locked() -> Iterator[None]:
    """Hold _PDFIUM_LOCK; before it is released, close the documents that _close() left to the next holder."""
    with _PDFIUM_LOCK:
        try:
            yield
        finally:
            while _UNCLOSED:
                _UNCLOSED.pop().close()


def _close(doc) -> None:
    """Close a document now if the lock is free (or this thread holds it), else leave it to the next holder of the
    lock. This runs in a generator's finally, which can run in the garbage collector, in a thread that holds a lock
    that another thread waits for while it holds _PDFIUM_LOCK (logging's module lock, taken inside pypdfium2):
    waiting for _PDFIUM_LOCK there would deadlock both threads."""
    _UNCLOSED.append(doc)
    if _PDFIUM_LOCK.acquire(blocking=False):
        try:
            while _UNCLOSED:
                _UNCLOSED.pop().close()
        finally:
            _PDFIUM_LOCK.release()


def _first_page_first(numbers: List[int], make) -> Iterator:
    """Yield `numbers`, then (n, make(n)) for each n. The first page is made before `numbers` is yielded, so that a
    first page that cannot be read raises ValueError or OSError before anything is sent, as a damaged image does."""
    page = make(numbers[0]) if numbers else None
    yield numbers
    for n in numbers:
        if page is None:
            page = make(n)
        yield n, page
        page = None


def _open_pdf(source: Union[str, os.PathLike, bytes], forms: bool = False):
    """Open a PDF with PDFium; the caller holds _PDFIUM_LOCK."""
    pdfium = _pdfium()
    try:                                        # pypdfium2 takes a str or pathlib.Path, not other os.PathLike objects
        doc = pdfium.PdfDocument(os.fsdecode(source) if isinstance(source, os.PathLike) else source)
    except pdfium.PdfiumError as e:
        raise ValueError(f"cannot open the PDF{_name(source)}: {_reason(e)}; "
                         "is it password-protected or damaged?") from e
    except FileNotFoundError:                   # pypdfium2's message is the path alone
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.fspath(source)) from None
    if forms and hasattr(doc, "init_forms"):    # pypdfium2 >= 4.2; 4.0 and 4.1 set up forms by themselves
        _PDFIUM_DOC_LOG.addFilter(_no_xfa_notice)
        try:
            doc.init_forms()                    # so that filled-in form fields are drawn
        finally:
            _PDFIUM_DOC_LOG.removeFilter(_no_xfa_notice)
    return doc


def _no_xfa_notice(record: logging.LogRecord) -> bool:
    """A logging filter for init_forms(): pypdfium2 5 logs, for every XFA form (Adobe LiveCycle and AEM forms), that its
    build cannot render XFA and how to build one that can. Its PyPI builds cannot; the page and its AcroForm fields are
    drawn as without XFA, and 0.1.0, which did not set up forms, printed nothing."""
    return "XFA" not in str(record.msg)


def _render_page(doc, n: int, long_side: int) -> Image.Image:
    """Render page n (1-based) to RGB; the caller holds _PDFIUM_LOCK."""
    page = doc[n - 1]
    try:
        w, h = page.get_size()                      # in points (1/72 inch)
        bitmap = page.render(scale=long_side / max(w, h))
        try:
            return bitmap.to_pil().convert("RGB")
        finally:
            bitmap.close()                          # freed here, inside the lock, not later by the garbage collector
    finally:
        page.close()


def _image_pages(source: ImageInput, pages: Optional[Iterable[int]], any_page: bool):
    # A single image, and the first page of a multi-page TIFF, is decoded before the page numbers are yielded, so
    # that a damaged file raises OSError before anything is sent, as before. Later TIFF frames are decoded one at a
    # time.
    if isinstance(source, Image.Image):             # the page itself: as before, `pages` only gives its number
        numbers = [1] if pages is None else list(itertools.islice(pages, 1)) or [1]
        # Only an image straight from Image.open() is turned upright: a rotated, converted or otherwise changed
        # copy (format None) may still carry the orientation tag of the photo it was made from.
        page = _to_rgb(source, upright=source.format is not None)
        yield numbers
        yield numbers[0], page
        return
    data = source if isinstance(source, bytes) else _read_once(source)

    def open_image() -> Image.Image:
        if data is None:
            return Image.open(source)
        try:
            return Image.open(io.BytesIO(data))
        except Image.UnidentifiedImageError:        # a named pipe: say its name, as Image.open(path) does
            if not isinstance(source, (str, os.PathLike)):
                raise
            raise Image.UnidentifiedImageError(f"cannot identify image file {os.fspath(source)!r}") from None
    # The file is opened once at a time, never twice at once: a path such as /dev/stdin or /dev/fd/N (macOS) gives
    # every file opened on it one shared read position, so two files open together would read each other's data.
    with open_image() as probe:
        tiff = probe.format == "TIFF"
    # Every other image, animated or not, is the frame it opens at (a PSD's is 1).
    frames, damaged = _tiff_frames(open_image, source) if tiff else ([0], False)
    with open_image() as image:
        if len(frames) == 1 and not damaged:
            _check_single(pages, any_page)
            if tiff:
                _seek(image, frames[0])
            page = _to_rgb(image)
            yield [1]
            yield 1, page
            return

        def frame(n: int) -> Image.Image:
            _seek(image, frames[n - 1])
            return _to_rgb(image)
        yield from _first_page_first(_checked(pages, len(frames), "TIFF"), frame)


def _read_once(source) -> Optional[bytes]:
    """The contents of a source that can be read only once, which Image.open() reads into memory itself: a stream that
    cannot seek (a pipe, an HTTP response), and a path to a named pipe or a socket (also <(...) in a shell, and
    /dev/stdin from a pipe or a socket). None for every other source, which can be opened again, as a TIFF is to look
    through its frames."""
    if isinstance(source, (str, os.PathLike)):
        try:
            mode = os.stat(source).st_mode
            if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)):
                return None
        except (OSError, ValueError):               # e.g. a missing file: Image.open() says what is wrong
            return None
        with open(source, "rb") as f:
            return f.read()
    try:
        source.seek(0)                              # as Image.open() does
    except (AttributeError, io.UnsupportedOperation):
        return source.read()
    return None


def _tiff_frames(open_image: Callable[[], Image.Image], source=None) -> Tuple[List[int], bool]:
    """The frames of a TIFF that are pages: each frame (scanner and fax output) except reduced-resolution copies
    (thumbnails, pyramid levels) and transparency masks. Also whether its frame list is damaged: then the pages
    before the damage are the pages, with a UserWarning. The frames are looked through in a file of their own
    (`open_image()`), so that the pages are read from one that nothing has moved through, page 1 exactly as before."""
    frames, i = [], 0
    with open_image() as image:
        try:
            while True:                             # not n_frames, which fails as a whole when one frame is damaged
                try:
                    image.seek(i)                   # EOFError after the last frame
                except Image.DecompressionBombError:    # Pillow 9 checks each frame's size here: it is still a page,
                    pass                                # and reading it raises the error, as in later versions
                if not image.tag_v2.get(254, 0) & 5:    # NewSubfileType: 1 = reduced resolution, 4 = transparency mask
                    frames.append(i)
                i += 1
        except EOFError:
            damaged = False
        except Exception:                           # frame i cannot be read: keep the pages before it
            damaged = True
    frames = frames or [0]
    if damaged:
        warnings.warn(f"the TIFF{_name(source)} is damaged after page {len(frames)}: the pages after it cannot be "
                      "read", stacklevel=2)
    return frames, damaged


def _seek(image: Image.Image, frame: int) -> None:
    """image.seek(frame) for a TIFF, without what Pillow keeps from the frames it went through: the palette of a
    palette frame (Pillow 12 and older then decode a frame without one with it: a black page, or "unrecognized image
    mode") and an ICC profile (every version, when the frame has none; the PNG sent would carry it)."""
    if image.tell() == frame:
        return
    image.seek(frame)
    if image.mode not in ("P", "PA"):
        image.palette = None                        # as Pillow 13 does
    if 34675 not in image.tag_v2:                   # ICCProfile
        image.info.pop("icc_profile", None)


def _checked(pages: Optional[Iterable[int]], count: int, kind: str) -> List[int]:
    if pages is None:
        return list(range(1, count + 1))
    numbers = []
    for p in pages:                                 # one at a time: a selection is not expanded beyond the count
        try:
            p = operator.index(p)
        except TypeError:
            raise ValueError(f"{p!r} is not a page number") from None
        if not 1 <= p <= count:
            raise ValueError(f"page {p} is out of range: the {kind} has {count} page(s)")
        numbers.append(p)
    return numbers


def _check_single(pages: Optional[Iterable[int]], any_page: bool) -> None:
    if not any_page and pages is not None and list(itertools.islice(pages, 2)) not in ([], [1]):
        raise ValueError("`pages` can only be used with PDFs and multi-page TIFFs; an image has one page")


def _to_rgb(image: Image.Image, upright: bool = True) -> Image.Image:
    """The image in RGB: upright by its EXIF orientation (when `upright`), and 16-bit greyscale whose ink is above 255
    scaled to 8 bits. Other images, transparent ones included, get the plain convert("RGB") of 0.1.0: a transparent
    area shows the colour stored under it, often black. A damaged file raises OSError, also where Pillow raises
    SyntaxError (a PNG whose chunks are broken)."""
    try:
        return _rgb(_upright(image) if upright else image)
    except SyntaxError as e:
        raise OSError(str(e)) from e


def _rgb(image: Image.Image) -> Image.Image:
    if image.mode in _GREY16:
        grey = image.convert("I")                   # getextrema() does not work on every 16-bit mode
        extrema = grey.getextrema()                 # None for an empty image
        if extrema and extrema[1] > 255 and _clipping_whitens_the_ink(grey):
            top = (1 << int(extrema[1]).bit_length()) - 1     # the data's own depth: 511 (9 bits) ... 65535 (16 bits)
            return grey.point(lambda v: v * (255 / top)).convert("L").convert("RGB")
    return image.convert("RGB")


def _clipping_whitens_the_ink(grey: Image.Image) -> bool:
    """True when most of the dark half of a 16-bit greyscale image (mode "I") is above 255, which convert("RGB") turns
    white: a real 16-bit scan, which 0.1.0 sent as a blank page. 8-bit values stored in 16 bits, also with some above
    255 (a sharpened or brightened scan), and ink below 256 (paper above it turns white) keep that conversion."""
    dark = sum(grey.histogram()[:128])              # 256 bins from the image's darkest value to its lightest
    kept = sum(grey.convert("L").histogram()[:255])     # values that stay below white
    return 2 * kept < dark


def _upright(image: Image.Image) -> Image.Image:
    """Turn a photo upright by its EXIF orientation (phones store portrait photos sideways). TIFFs are left alone:
    Pillow applies their orientation itself when it loads them (Pillow 9 keeps the tag, so it would apply twice).
    Pixels that cannot be read raise here, as in convert(): only errors in the EXIF data itself are ignored (Pillow
    keeps what it decoded of a damaged file, and a second load() would return it without an error)."""
    if image.format == "TIFF":
        return image
    if "exif" not in image.info and image.format == "PNG":
        image.load()                                # a PNG's eXIf chunk may come after the pixels
    try:
        if _exif_orientation(image) not in (2, 3, 4, 5, 6, 7, 8):
            return image
    except Exception:                               # unreadable EXIF: leave the image as it is
        return image
    image.load()
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def _exif_orientation(image: Image.Image) -> int:
    """The Orientation tag in the first directory of the image's EXIF block, 1 when there is none. Read here, not with
    image.getexif(), which also takes an orientation from XMP metadata when the EXIF block has none (photo viewers do
    not; Pillow does it for some formats and versions only) and warns about every malformed tag, also of photos that
    need no turning. Raises struct.error for a block cut short."""
    data = image.info.get("exif")
    if not isinstance(data, bytes):
        return 1
    if data.startswith(b"Exif\x00\x00"):            # JPEG and PNG; WebP may leave it out
        data = data[6:]
    order = {b"II*\x00": "<", b"MM\x00*": ">"}.get(data[:4])
    if order is None:
        return 1
    ifd = struct.unpack_from(order + "I", data, 4)[0]
    for k in range(struct.unpack_from(order + "H", data, ifd)[0]):
        tag, kind, count = struct.unpack_from(order + "HHI", data, ifd + 2 + 12 * k)
        if tag == 0x0112:                           # a SHORT, as the standard says, or a LONG
            if kind not in (3, 4) or count != 1:
                return 1
            return struct.unpack_from(order + ("H" if kind == 3 else "I"), data, ifd + 10 + 12 * k)[0]
    return 1


def _name(source) -> str:
    return f" {os.fspath(source)!r}" if isinstance(source, (str, os.PathLike)) else ""


def _reason(error: Exception) -> str:
    return str(error).rstrip(".")


def _pdfium():
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover - it is a dependency, but give a clear message anyway
        raise ImportError("PDF support needs pypdfium2: pip install pypdfium2") from e
    return pdfium
