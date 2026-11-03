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


def _text_w(text: str) -> float:
    return sum(SPACE if c == " " else _w(c) for c in text)


def _tokens(res: tp.Result) -> list[list[tuple]]:
    """Split the score into source lines of (kind, value) tokens; text is cut after every bar line."""
    lines: list[list[tuple]] = [[]]
    for s in res.parsed.segs:
        if s.kind == "text":
            for i, part in enumerate(s.text.split("\n")):
                if i:
                    lines.append([])
                for piece in part.replace("|", "|\0").split("\0"):
                    if piece:
                        lines[-1].append(("text", piece))
        elif s.kind == "key":
            lines[-1].append(("text", f"1={res.tgt.jianpu}"))
        elif s.kind == "chord":
            lines[-1].append(("chord", tp.transpose_chord(s.text, res.src, res.tgt, res.inst)))
        else:
            lines[-1].append(("note", s.note))
    if not any(s.kind == "key" for s in res.parsed.segs):
        lines.insert(0, [("text", f"1={res.tgt.jianpu}")])
    return lines


LYRIC_PREFIX = re.compile(r"^\s*詞\s*\d*\s*[:：]?\s*")
# One syllable per note: "_" holds the previous syllable over an extra note; Latin words stay whole;
# trailing punctuation sticks to its syllable.
SYLLABLE = re.compile(r"_|[A-Za-z'’]+[,.!?;:，。、！？；：」』）)]*|[^\s_][,.!?;:，。、！？；：」』）)]*")


def _is_lyrics(line: list[tuple]) -> bool:
    return bool(line) and all(k == "text" for k, _ in line) and "".join(v for _, v in line).strip().startswith("詞")


def _syllables(line: list[tuple]) -> list[str]:
    text = LYRIC_PREFIX.sub("", "".join(v for _, v in line))
    return ["" if s == "_" else s for s in SYLLABLE.findall(text)]


def _attach_lyrics(lines: list[list[tuple]]) -> tuple[list[list[tuple]], dict[int, list[str]]]:
    """Fold each run of 「詞：」 lines into the melody line above it.

    Returns the remaining lines and {id(note): [syllable per verse]}.
    A lyrics line with no melody line above it stays as plain text.
    """
    kept: list[list[tuple]] = []
    lyrics: dict[int, list[str]] = {}
    melody: list | None = None  # notes of the last melody line
    verse = 0
    for line in lines:
        if _is_lyrics(line) and melody:
            for note, syl in zip(melody, _syllables(line)):
                lyrics.setdefault(id(note), [])
                lyrics[id(note)] += [""] * (verse - len(lyrics[id(note)])) + [syl]
            verse += 1
            continue
        kept.append(line)
        notes = [v for k, v in line if k == "note"]
        if notes:
            melody, verse = notes, 0
        elif "".join(v for k, v in line if k == "text").strip():
            melody = None
    return kept, lyrics


def _width(tok: tuple) -> float:
    kind, v = tok
    if kind == "chord":
        return 0
    if kind == "text":
        return _text_w(v)
    return _w(str(v.deg)) + (_w(tp.acc_sym(v.acc), SIZE * 0.6) if v.acc else 0)


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


def _draw_row(page: pymupdf.Page, row: list[tuple], y: float, res: tp.Result,
              lyrics: dict[int, list[str]]) -> None:
    """y is the digit baseline."""
    x = MARGIN
    chord_end = 0.0
    lyric_end: dict[int, float] = {}  # per verse, so long syllables push the next one right
    for kind, v in row:
        if kind == "chord":
            cx = max(x, chord_end + 3)
            page.insert_text((cx, y - SIZE - 4), v, fontname=FONT, fontsize=SIZE * 0.62, color=CHORD)
            chord_end = cx + _w(v, SIZE * 0.62)
        elif kind == "text":
            color = GREY if v.strip().startswith("詞") else BLACK
            for c in v:
                if c != " ":
                    page.insert_text((x, y), c, fontname=FONT, fontsize=SIZE, color=color)
                x += SPACE if c == " " else _w(c)
        else:
            color = FLAG_COLORS.get(v.flag, BLACK)
            if v.acc:
                acc = tp.acc_sym(v.acc)
                page.insert_text((x, y - SIZE * 0.4), acc, fontname=FONT, fontsize=SIZE * 0.6, color=color)
                x += _w(acc, SIZE * 0.6)
            digit = str(v.deg)
            dw = _w(digit)
            page.insert_text((x, y), digit, fontname=FONT, fontsize=SIZE, color=color)
            o = res.written_octave(v)
            cx = x + dw / 2
            for i in range(abs(o)):
                below = y + (7 if v.flag == "moved" else 4.5)  # keep clear of the underline
                cy = y - SIZE * 0.8 - 2.5 - i * 4.5 if o > 0 else below + i * 4.5
                page.draw_circle((cx, cy), 1.6, color=color, fill=color)
            if v.flag == "moved":
                page.draw_line((x, y + 2.5), (x + dw, y + 2.5), color=color, width=0.8)
            for i, syl in enumerate(lyrics.get(id(v), [])):
                if syl:
                    sw = _w(syl, LYRIC_SIZE)
                    lx = max(cx - sw / 2, lyric_end.get(i, 0) + 3)
                    page.insert_text((lx, y + 24 + i * LYRIC_H), syl, fontname=FONT, fontsize=LYRIC_SIZE)
                    lyric_end[i] = lx + sw
            x += dw


def _new_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_font(fontname=FONT, fontbuffer=_font.buffer)  # embedded, so widths match _w()
    return page


def build_pdf(res: tp.Result, title: str = "") -> bytes:
    doc = pymupdf.open()
    width = PAGE_W - 2 * MARGIN
    lines, lyrics = _attach_lyrics(_tokens(res))
    rows = [r for line in lines for r in (_rows(line, width) or [[]])]
    page = _new_page(doc)
    y = MARGIN
    heading = f"{title}　{res.inst.name}譜" if title else f"{res.inst.name}譜"
    page.insert_text((MARGIN, y + 16), heading, fontname=FONT, fontsize=20)
    info = (f"原曲 1={res.src.jianpu}（{res.src.label}）→ 記譜 1={res.tgt.jianpu}（{tp.sig_long(res.tgt.sig)}）"
            f"　實際聲音{tp.octave_words(res.octave)}")
    page.insert_text((MARGIN, y + 38), info, fontname=FONT, fontsize=9.5, color=GREY)
    y += 62

    for row in rows:
        if not row:
            y += ROW_H * 0.4
            continue
        verses = max((len(lyrics.get(id(v), [])) for k, v in row if k == "note"), default=0)
        h = ROW_H + verses * LYRIC_H
        if y + h > PAGE_H - MARGIN:
            page = _new_page(doc)
            y = MARGIN
        _draw_row(page, row, y + ROW_H - 22, res, lyrics)
        y += h

    legend = "紅色＝超出音域　橘色底線＝已單獨移八度" if res.out_count or res.moved_count else ""
    for i, p in enumerate(doc):
        footer = f"{i + 1} / {doc.page_count}"
        p.insert_text((PAGE_W - MARGIN - _w(footer, 9), PAGE_H - 28), footer, fontname=FONT, fontsize=9, color=GREY)
        if legend:
            p.insert_text((MARGIN, PAGE_H - 28), legend, fontname=FONT, fontsize=9, color=GREY)

    doc.subset_fonts()
    return doc.tobytes(garbage=3, deflate=True)
