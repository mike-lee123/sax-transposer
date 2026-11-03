"""Render a transposed score (transpose.Result) as a printable A4 jianpu PDF."""
from __future__ import annotations

import re

import pymupdf

import transpose as tp

FONT = "cjk"  # name of the embedded font on each page
PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN = 50
SIZE = 17  # note digits
ROW_H = 58  # chord line + digits + octave dots below
LYRIC_SIZE = 12.5
LYRIC_H = 19  # extra height per verse of lyrics
BLACK, GREY, CHORD = (0, 0, 0), (0.4, 0.4, 0.4), (0.69, 0.49, 0.17)
FLAG_COLORS = {"out": (0.9, 0.28, 0.3), "moved": (0.85, 0.47, 0.02)}

_font = pymupdf.Font("china-t")  # PyMuPDF's bundled Droid Sans Fallback: digits, ♭ ♯ and Chinese


def _w(text: str, size: float = SIZE) -> float:
    return _font.text_length(text, fontsize=size)


SPACE = SIZE * 0.6  # wider than the font's space so notes are easy to read
PAD = 1.6  # each side of a note or rest; underlines run through it so a beat reads as one beam
UL_Y, UL_GAP = 3.2, 2.8  # first underline below the baseline, distance to the next


def _text_w(text: str) -> float:
    return sum(SPACE if c == " " else _w(c) for c in text)


def _tokens(res: tp.Result) -> list[list[tuple]]:
    """Split the score into source lines of (kind, value) tokens; text is cut after every bar line.

    Lyrics lines that belong to a melody line are dropped: their syllables are drawn under the notes.
    """
    lines: list[list[tuple]] = [[]]
    for s in res.parsed.segs:
        if s.kind == "text":
            for i, part in enumerate(s.text.split("\n")):
                if i:
                    lines.append([])
                for piece in re.split(r"(?<=\|)|([{}])", part):
                    if piece in ("{", "}"):
                        lines[-1].append(("tup", piece))
                    elif piece:
                        lines[-1].append(("text", piece))
        elif s.kind == "key":
            lines[-1].append(("text", f"1={res.tgt.jianpu}"))
        elif s.kind == "chord":
            lines[-1].append(("chord", tp.transpose_chord(s.text, res.src, res.tgt, res.inst)))
        elif s.kind == "rest":
            lines[-1].append(("rest", s))
        elif s.kind == "slur":
            lines[-1].append(("slur", s.text))
        else:
            lines[-1].append(("note", s.note))
    lines = [ln for i, ln in enumerate(lines, 1) if i not in res.lyric_lines]
    if not any(s.kind == "key" for s in res.parsed.segs):
        lines.insert(0, [("text", f"1={res.tgt.jianpu}")])
    return lines


def _width(tok: tuple) -> float:
    kind, v = tok
    if kind in ("chord", "tup", "slur"):
        return 0
    if kind == "text":
        return _text_w(v)
    if kind == "rest":
        return 2 * PAD + _w("0") + v.dot * _w("·")
    return 2 * PAD + _w(str(v.deg)) + v.dot * _w("·") + (_w(tp.acc_sym(v.acc), SIZE * 0.6) if v.acc else 0)


def _underline(page: pymupdf.Page, x0: float, x1: float, y: float, n: int, color) -> None:
    for i in range(n):
        yy = y + UL_Y + i * UL_GAP
        page.draw_line((x0, yy), (x1, yy), color=color, width=0.9)


def _measures(line: list[tuple]) -> list[list[tuple]]:
    """Group tokens into chunks that end at a bar line, so rows wrap between measures."""
    out: list[list[tuple]] = [[]]
    for tok in line:
        out[-1].append(tok)
        if tok[0] == "text" and tok[1].rstrip().endswith("|"):
            out.append([])
    return [m for m in out if m]


def _rows(line: list[tuple], width: float) -> list[list[tuple]]:
    rows: list[list[tuple]] = [[]]
    used = 0.0
    for m in _measures(line):
        mw = sum(_width(t) for t in m)
        if rows[-1] and used + mw > width:
            rows.append([])
            used = 0.0
            if m[0][0] == "text":  # don't start a row with the spaces after a bar line
                m = [("text", m[0][1].lstrip())] + m[1:]
                mw = sum(_width(t) for t in m)
        if mw <= width:
            rows[-1].extend(m)
            used += mw
            continue
        for t in m:  # a single measure wider than the page: wrap token by token
            if rows[-1] and used + _width(t) > width:
                rows.append([])
                used = 0.0
            rows[-1].append(t)
            used += _width(t)
    return rows


def _draw_row(page: pymupdf.Page, row: list[tuple], y: float, res: tp.Result, open_slurs: int = 0) -> int:
    """y is the digit baseline. open_slurs: slurs carried over from the row above.

    Returns the slurs still open at the end of this row.
    """
    x = MARGIN
    chord_end = 0.0
    lyric_end: dict[int, float] = {}  # per verse, so long syllables push the next one right
    tuplets: list[list] = []  # [start x, does it pass over high octave dots]
    slurs = [[x, False] for _ in range(open_slurs)]  # same shape
    for kind, v in row:
        if kind == "slur":
            if v == "(":
                slurs.append([x, False])
            elif slurs:
                _slur(page, *slurs.pop(), x, y)
        elif kind == "chord":
            cx = max(x, chord_end + 3)
            page.insert_text((cx, y - SIZE - 10), v, fontname=FONT, fontsize=SIZE * 0.62, color=CHORD)
            chord_end = cx + _w(v, SIZE * 0.62)
        elif kind == "tup":
            if v == "{":
                tuplets.append([x, False])
            elif tuplets:  # triplet: an arc over the group with a 3 set into its top
                x_start, high = tuplets.pop()
                top = _slur(page, x_start, high, x, y)
                mid, tw = (x_start + x) / 2, _w("3", 8)
                page.draw_rect((mid - tw / 2 - 1, top - 3, mid + tw / 2 + 1, top + 3), color=None, fill=(1, 1, 1))
                page.insert_text((mid - tw / 2, top + 2.8), "3", fontname=FONT, fontsize=8)
        elif kind == "rest":
            x0 = x
            x += PAD
            page.insert_text((x, y), "0" + "·" * v.dot, fontname=FONT, fontsize=SIZE)
            x += _w("0") + v.dot * _w("·") + PAD
            _underline(page, x0, x, y, v.ul, BLACK)
        elif kind == "text":
            color = GREY if v.strip().startswith("詞") else BLACK
            for c in v:
                if c != " ":
                    page.insert_text((x, y), c, fontname=FONT, fontsize=SIZE, color=color)
                x += SPACE if c == " " else _w(c)
        else:
            color = FLAG_COLORS.get(v.flag, BLACK)
            x0 = x
            x += PAD
            if v.acc:
                acc = tp.acc_sym(v.acc)
                page.insert_text((x, y - SIZE * 0.4), acc, fontname=FONT, fontsize=SIZE * 0.6, color=color)
                x += _w(acc, SIZE * 0.6)
            digit = str(v.deg)
            dw = _w(digit)
            page.insert_text((x, y), digit, fontname=FONT, fontsize=SIZE, color=color)
            o = res.written_octave(v)
            if o > 0:
                for s in slurs + tuplets:
                    s[1] = True
            cx = x + dw / 2
            for i in range(abs(o)):
                below = y + 4.5 + v.ul * UL_GAP + (1 if v.ul else 0)  # low dots go under the underlines
                cy = y - SIZE * 0.8 - 2.5 - i * 4.5 if o > 0 else below + i * 4.5
                page.draw_circle((cx, cy), 1.6, color=color, fill=color)
            if v.dot:
                page.insert_text((x + dw, y), "·" * v.dot, fontname=FONT, fontsize=SIZE, color=color)
            for i, syl in enumerate(v.lyrics):
                if syl:
                    sw = _w(syl, LYRIC_SIZE)
                    lx = max(cx - sw / 2, lyric_end.get(i, 0) + 3)
                    page.insert_text((lx, y + 24 + i * LYRIC_H), syl, fontname=FONT, fontsize=LYRIC_SIZE)
                    lyric_end[i] = lx + sw
            x += dw + v.dot * _w("·") + PAD
            _underline(page, x0, x, y, v.ul, color)
    for s in slurs:  # continues on the next row
        _slur(page, *s, x, y)
    return len(slurs)


def _slur(page: pymupdf.Page, x0: float, high: bool, x1: float, y: float) -> float:
    """A slur, tie or triplet arc over the notes between x0 and x1, raised above high octave dots.

    Returns the y of the top of the arc.
    """
    ye = y - SIZE * 0.85 - (5 if high else 0)
    h = min(7.0, 2.5 + (x1 - x0) * 0.05)
    d = (x1 - x0) * 0.25
    page.draw_bezier((x0 + 2, ye), (x0 + d, ye - h), (x1 - d, ye - h), (x1 - 2, ye), color=BLACK, width=0.8)
    return ye - 0.75 * h  # a cubic Bézier with level control points peaks at 3/4 of their height


def _new_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_font(fontname=FONT, fontbuffer=_font.buffer)  # embedded, so widths match _w()
    return page


def build_pdf(res: tp.Result, title: str = "") -> bytes:
    doc = pymupdf.open()
    width = PAGE_W - 2 * MARGIN
    rows = [r for line in _tokens(res) for r in (_rows(line, width) or [[]])]
    page = _new_page(doc)
    y = MARGIN
    heading = f"{title}　{res.inst.name}譜" if title else f"{res.inst.name}譜"
    page.insert_text((MARGIN, y + 16), heading, fontname=FONT, fontsize=20)
    info = (f"原曲 1={res.src.jianpu}（{res.src.label}）→ 記譜 1={res.tgt.jianpu}（{tp.sig_long(res.tgt.sig)}）"
            f"　實際聲音{tp.octave_words(res.octave)}")
    page.insert_text((MARGIN, y + 38), info, fontname=FONT, fontsize=9.5, color=GREY)
    y += 62

    open_slurs = 0
    for row in rows:
        if not row:
            y += ROW_H * 0.4
            continue
        verses = max((len(v.lyrics) for k, v in row if k == "note"), default=0)
        h = ROW_H + verses * LYRIC_H
        if y + h > PAGE_H - MARGIN:
            page = _new_page(doc)
            y = MARGIN
        open_slurs = _draw_row(page, row, y + ROW_H - 22, res, open_slurs)
        y += h

    legend = "紅色＝超出音域　橘色＝已移八度" if res.out_count or res.moved_count else ""
    for i, p in enumerate(doc):
        footer = f"{i + 1} / {doc.page_count}"
        p.insert_text((PAGE_W - MARGIN - _w(footer, 9), PAGE_H - 28), footer, fontname=FONT, fontsize=9, color=GREY)
        if legend:
            p.insert_text((MARGIN, PAGE_H - 28), legend, fontname=FONT, fontsize=9, color=GREY)

    doc.subset_fonts()
    return doc.tobytes(garbage=3, deflate=True)
