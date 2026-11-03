"""Transpose concert-pitch scores (jianpu or note names) for saxophones.

Every note is kept as a scale degree (1-7 plus accidental) relative to the key,
together with its absolute MIDI pitch. Transposing keeps the degree, moves the
key, and re-derives octave dots and note spellings from the written pitch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
NAT = [0, 2, 4, 5, 7, 9, 11]  # also the major scale in semitones

# (letter index, accidental) -> sharps (+) / flats (-) in the key signature
SIG = {
    (0, 0): 0, (4, 0): 1, (1, 0): 2, (5, 0): 3, (2, 0): 4, (6, 0): 5, (3, 1): 6, (0, 1): 7,
    (3, 0): -1, (6, -1): -2, (2, -1): -3, (5, -1): -4, (1, -1): -5, (4, -1): -6, (0, -1): -7,
}

LOWEST = 58  # written B♭3, the lowest note on every saxophone
HIGHEST = 89  # written F6
HIGHEST_FS = 90  # written F♯6 on instruments with a high F♯ key


@dataclass(frozen=True)
class Key:
    l: int  # letter index into LETTERS
    a: int  # accidental: -1 flat, 0 natural, +1 sharp

    @property
    def pc(self) -> int:
        return (NAT[self.l] + self.a) % 12

    @property
    def sig(self) -> int | None:
        return SIG.get((self.l, self.a))

    @property
    def name(self) -> str:
        return LETTERS[self.l] + acc_sym(self.a)

    @property
    def jianpu(self) -> str:
        return acc_sym(self.a) + LETTERS[self.l]

    @property
    def jianpu_ascii(self) -> str:
        return acc_ascii(self.a) + LETTERS[self.l]

    @property
    def relative_minor(self) -> "Key":
        l = (self.l + 5) % 7
        return Key(l, wrap((self.pc + 9) % 12 - NAT[l]))

    @property
    def label(self) -> str:
        return f"{self.name} 大調 ／ {self.relative_minor.name} 小調"


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    en: str
    tune: str
    semis: int  # written = concert + semis
    letters: int  # letter shift (mod 7) that matches `semis`
    interval: str


INSTRUMENTS = [
    Instrument("soprano", "高音薩克斯風", "Soprano", "B♭", 2, 1, "上移大二度"),
    Instrument("alto", "中音薩克斯風", "Alto", "E♭", 9, 5, "上移大六度"),
    Instrument("tenor", "次中音薩克斯風", "Tenor", "B♭", 14, 1, "上移八度＋大二度"),
    Instrument("bari", "上低音薩克斯風", "Baritone", "E♭", 21, 5, "上移八度＋大六度"),
]
INSTRUMENT_BY_ID = {i.id: i for i in INSTRUMENTS}

SOURCE_KEYS = [Key(l, a) for l, a in [
    (0, 0), (1, -1), (1, 0), (2, -1), (2, 0), (3, 0), (3, 1),
    (4, -1), (4, 0), (5, -1), (5, 0), (6, -1), (6, 0),
]]


def wrap(d: int) -> int:
    d %= 12
    return d - 12 if d > 6 else d


def acc_sym(a: int) -> str:
    return {-2: "𝄫", -1: "♭", 0: "", 1: "♯", 2: "𝄪"}.get(a, "")


def acc_ascii(a: int) -> str:
    return "#" * a if a > 0 else "b" * -a


def respell(k: Key) -> Key:
    """Prefer a key signature with at most six sharps or flats."""
    if k.sig is not None and abs(k.sig) <= 6:
        return k
    for (l, a), s in SIG.items():
        cand = Key(l, a)
        if cand.pc == k.pc and abs(s) <= 6:
            return cand
    return k


def target_key(src: Key, inst: Instrument) -> Key:
    l = (src.l + inst.letters) % 7
    pc = (src.pc + inst.semis) % 12
    return respell(Key(l, wrap(pc - NAT[l])))


def spell(midi: int, letter: int, ascii_acc: bool = False) -> str:
    a = wrap(midi % 12 - NAT[letter])
    octave = (midi - a) // 12 - 1
    return LETTERS[letter] + (acc_ascii(a) if ascii_acc else acc_sym(a)) + str(octave)


def spell_pc(letter: int, pc: int) -> str:
    a = wrap(pc - NAT[letter])
    if abs(a) > 1:
        for d in (1, -1):
            l2 = (letter + d) % 7
            a2 = wrap(pc - NAT[l2])
            if abs(a2) <= 1:
                return LETTERS[l2] + acc_sym(a2)
    return LETTERS[letter] + acc_sym(a)


def sig_short(s: int) -> str:
    return "無升降" if s == 0 else f"{s}♯" if s > 0 else f"{-s}♭"


def sig_long(s: int) -> str:
    if s == 0:
        return "無升降記號"
    if s > 0:
        return f"{s} 個升記號（" + " ".join(c + "♯" for c in "FCGDAEB"[:s]) + "）"
    return f"{-s} 個降記號（" + " ".join(c + "♭" for c in "BEADGCF"[:-s]) + "）"


def octave_words(k: int) -> str:
    if k == 0:
        return "與原曲同音高"
    n = ["", "一", "兩", "三"][abs(k)]
    return f"比原曲{'低' if k < 0 else '高'}{n}個八度"


# ---------- parsing ----------

R_KEY = re.compile(r"1\s*=\s*([#♯b♭]?)([A-G])([#♯b♭]?)")
R_TIME = re.compile(r"\d+\s*/\s*\d+")
R_CHORD = re.compile(r"\[([^\]\n]{1,16})\]")
# Rhythm suffixes: "·" augmentation dot, "_" one underline (eighth), "__" two (sixteenth).
# Notes of one beat are written without spaces ("5·_6__") so their underlines join.
R_JIANPU = re.compile(r"(\.*)([#♯b♭]?)(\.*)([1-7])([̣̤̇̈]*)([',]*)(·*)(_{0,3})")
R_NAME = re.compile(r"([A-G])(##|bb|#|♯|b|♭)?(-?\d)(·*)(_{0,3})")
R_REST = re.compile(r"0(·*)(_{0,3})")
R_SLUR_JIANPU = re.compile(r"\(\s*[{.#♯b♭0-7]")
R_SLUR_NAMES = re.compile(r"\(\s*[{0A-G]")
R_CHORD_PARTS = re.compile(r"^([A-G])([#♯b♭]?)(.*?)(?:/([A-G])([#♯b♭]?))?$")
R_LYRIC_LINE = re.compile(r"[ \t]*詞[^\n]*")
# Also drops a verse number copied from the page: 「詞：1.雖然」.
R_LYRIC_PREFIX = re.compile(r"^\s*詞\s*\d*\s*[:：]?\s*(?:\d+\s*[.、．]\s*)?")
# One syllable per note: "_" holds the previous syllable over an extra note; Latin words stay whole;
# trailing punctuation (including 「……」 and "...") sticks to its syllable and never takes a note.
_PUNCT = r"[,.!?;:，。、！？；：」』）)…‥·．\-~～]*"
R_SYLLABLE = re.compile(r"_|[A-Za-z'’]+" + _PUNCT + r"|[^\s_,.!?;:，。、！？；：」』）)…‥·．\-~～]" + _PUNCT)


@dataclass
class Note:
    line: int
    idx: int
    deg: int = 0
    acc: int = 0
    oct: int = 0
    letter: int = 0
    letter_acc: int = 0
    midi: int = 0
    written: int = 0
    flag: str = ""  # "", "out", "moved"
    bar: int = 0  # bars are moved an octave as a whole to fit the range
    dot: int = 0  # augmentation dots "·"
    ul: int = 0  # underlines: 1 eighth, 2 sixteenth
    lyrics: list[str] = field(default_factory=list)  # one syllable per verse ("" = none)


@dataclass
class Seg:
    kind: str  # text | key | chord | note | rest | slur ("(" or ")" in text)
    text: str = ""
    key: Key | None = None
    note: Note | None = None
    dot: int = 0  # rest only
    ul: int = 0  # rest only


@dataclass
class Parsed:
    segs: list[Seg]
    header: Key | None

    @property
    def notes(self) -> list[Note]:
        return [s.note for s in self.segs if s.kind == "note"]


def _acc_char(c: str) -> int:
    if c in ("#", "♯"):
        return 1
    if c in ("b", "♭"):
        return -1
    return 0


def parse(text: str, mode: str) -> Parsed:
    segs: list[Seg] = []
    buf: list[str] = []
    header = None
    line, idx, bar = 1, 0, 0
    slurs = 0  # open "(" slurs

    def flush():
        if buf:
            segs.append(Seg("text", "".join(buf)))
            buf.clear()

    i = 0
    while i < len(text):
        # A lyrics line is plain text: verse numbers like 「1.雖然」 must not be read as notes.
        if (i == 0 or text[i - 1] == "\n") and (m := R_LYRIC_LINE.match(text, i)):
            buf.append(m[0])
            i = m.end()
            continue
        if m := R_KEY.match(text, i):
            flush()
            k = respell(Key(LETTERS.index(m[2]), _acc_char(m[1]) or _acc_char(m[3])))
            segs.append(Seg("key", key=k))
            header = header or k
            i = m.end()
            continue
        if m := R_TIME.match(text, i):
            buf.append(m[0])
            i = m.end()
            continue
        if m := R_CHORD.match(text, i):
            flush()
            segs.append(Seg("chord", m[1]))
            i = m.end()
            continue
        if mode == "jianpu" and (m := R_JIANPU.match(text, i)):
            flush()
            o = -(len(m[1]) + len(m[3]))
            for c in m[5]:
                o += {"̇": 1, "̈": 2, "̣": -1, "̤": -2}[c]
            for c in m[6]:
                o += 1 if c == "'" else -1
            idx += 1
            segs.append(Seg("note", note=Note(line, idx, deg=int(m[4]), acc=_acc_char(m[2]), oct=o, bar=bar,
                                              dot=len(m[7]), ul=len(m[8]))))
            i = m.end()
            continue
        if mode == "names" and (m := R_NAME.match(text, i)):
            flush()
            l = LETTERS.index(m[1])
            a = 0
            if m[2]:
                a = 2 * _acc_char(m[2][0]) if len(m[2]) == 2 else _acc_char(m[2])
            idx += 1
            segs.append(Seg("note", note=Note(line, idx, letter=l, letter_acc=a, bar=bar,
                                              midi=(int(m[3]) + 1) * 12 + NAT[l] + a,
                                              dot=len(m[4]), ul=len(m[5]))))
            i = m.end()
            continue
        if m := R_REST.match(text, i):
            flush()
            segs.append(Seg("rest", dot=len(m[1]), ul=len(m[2])))
            i = m.end()
            continue
        # Slur or tie: "(" right before a note, closed by the next ")". Other brackets stay text.
        if (text[i] == "(" and (R_SLUR_JIANPU if mode == "jianpu" else R_SLUR_NAMES).match(text, i)) or (
                text[i] == ")" and slurs):
            flush()
            slurs += 1 if text[i] == "(" else -1
            segs.append(Seg("slur", text[i]))
            i += 1
            continue
        if text[i] in "|\n":
            bar += 1
        if text[i] == "\n":
            line += 1
            idx = 0
        buf.append(text[i])
        i += 1
    flush()
    return Parsed(segs, header)


def attach_lyrics(text: str, notes: list[Note]) -> set[int]:
    """Give each note its syllables from the 「詞：」 lines under its melody line.

    Returns the line numbers of the lyrics lines that were used (a lyrics line with
    no melody line above it is left alone and shown as plain text).
    """
    by_line: dict[int, list[Note]] = {}
    for n in notes:
        n.lyrics = []
        by_line.setdefault(n.line, []).append(n)
    used: set[int] = set()
    melody: list[Note] | None = None
    verse = 0
    for i, raw in enumerate(text.split("\n"), 1):
        if i in by_line:
            melody, verse = by_line[i], 0
        elif raw.strip().startswith("詞") and melody:
            syllables = ["" if s == "_" else s for s in R_SYLLABLE.findall(R_LYRIC_PREFIX.sub("", raw))]
            for n, syl in zip(melody, syllables):
                n.lyrics += [""] * (verse - len(n.lyrics)) + [syl]
            verse += 1
            used.add(i)
        elif raw.strip():
            melody = None
    return used


def assign_pitches(notes: list[Note], src: Key, ref_octave: int, mode: str) -> None:
    ref = 60 + src.pc + 12 * ref_octave  # pitch of the middle "1"
    for n in notes:
        if mode == "jianpu":
            n.midi = ref + NAT[n.deg - 1] + n.acc + 12 * n.oct
        else:
            n.deg = (n.letter - src.l) % 7 + 1
            n.acc = wrap(n.midi % 12 - (src.pc + NAT[n.deg - 1]) % 12)


MOVE_COST = 1.0  # per note moved an octave away from the rest of the tune


def note_cost(w: int) -> float:
    """How awkward an in-range written pitch is: 0 from D4 up to C♯6 (octave key, no palm keys)."""
    if w < 60:  # B♭3, B3: hard to play softly and in tune
        return 1
    if w < 62:  # C4, C♯4
        return 0.5
    if w > 85:  # D6–F♯6: palm keys
        return 0.5
    return 0


def fit_range(notes: list[Note], inst: Instrument, k: int, high: int) -> tuple[list[int], list[str]]:
    """Written pitches at octave shift k, with out-of-range bars moved an octave as a whole.

    A bar that can't fit as a whole has only its out-of-range notes moved. Returns
    (written pitches, flags) with flag "moved" for moved notes and "out" for notes that still don't fit.
    """
    ws = [n.midi + inst.semis + 12 * k for n in notes]
    flags = [""] * len(notes)
    bars: dict[int, list[int]] = {}
    for i, n in enumerate(notes):
        bars.setdefault(n.bar, []).append(i)
    for idxs in bars.values():
        if all(LOWEST <= ws[i] <= high for i in idxs):
            continue
        fits = [s for s in (12, -12, 24, -24) if all(LOWEST <= ws[i] + s <= high for i in idxs)]
        if fits:
            s = min(fits, key=lambda s: (sum(note_cost(ws[i] + s) for i in idxs), abs(s)))
            for i in idxs:
                ws[i] += s
                flags[i] = "moved"
            continue
        for i in idxs:
            w = ws[i]
            while w < LOWEST:
                w += 12
            while w > high:
                w -= 12
            if w != ws[i]:
                ws[i], flags[i] = w, "moved" if LOWEST <= w <= high else "out"
    return ws, flags


def score_octave(notes: list[Note], inst: Instrument, k: int, high: int) -> float:
    ws, flags = fit_range(notes, inst, k, high)
    return (sum(note_cost(w) for w in ws) + MOVE_COST * sum(f == "moved" for f in flags)
            + 1000 * flags.count("out") + 0.75 * abs(k))  # stay at the original pitch unless clearly better


def best_octave(notes: list[Note], inst: Instrument, high: int) -> int:
    """The octave shift that needs the fewest moved notes and keeps the tune in the comfortable range."""
    if not notes:
        return 0
    return min([0, -1, 1, -2, 2, -3], key=lambda k: score_octave(notes, inst, k, high))


# ---------- transposition result ----------

@dataclass
class Result:
    parsed: Parsed
    src: Key
    tgt: Key
    inst: Instrument
    octave: int
    auto_octave: int
    high: int
    lyric_lines: set[int] = field(default_factory=set)
    jianpu: str = ""
    jianpu_ascii: str = ""
    names: str = ""
    html: str = ""
    rows: list[dict] = field(default_factory=list)

    @property
    def notes(self) -> list[Note]:
        return self.parsed.notes

    @property
    def out_count(self) -> int:
        return sum(n.flag == "out" for n in self.notes)

    @property
    def moved_count(self) -> int:
        return sum(n.flag == "moved" for n in self.notes)

    def concert_name(self, n: Note) -> str:
        return spell(n.midi, (self.src.l + n.deg - 1) % 7)

    def written_name(self, n: Note) -> str:
        return spell(n.written, (self.tgt.l + n.deg - 1) % 7)

    def written_octave(self, n: Note) -> int:
        return round((n.written - (60 + self.tgt.pc) - NAT[n.deg - 1] - n.acc) / 12)


def transpose_chord(raw: str, src: Key, tgt: Key, inst: Instrument, ascii_acc: bool = False) -> str:
    m = R_CHORD_PARTS.match(raw)
    if not m:
        return raw
    shift = tgt.l - src.l

    def one(letter: str, acc: str) -> str:
        l = LETTERS.index(letter)
        return spell_pc((l + shift) % 7, (NAT[l] + _acc_char(acc) + inst.semis) % 12)

    s = one(m[1], m[2]) + m[3] + ("/" + one(m[4], m[5]) if m[4] else "")
    return s.replace("♯", "#").replace("♭", "b") if ascii_acc else s


def _dots(o: int) -> str:
    if o == 0:
        return ""
    if o > 0:
        return "̈" if o == 2 else "̇" * o
    return "̤" if o == -2 else "̣" * -o


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def transpose(text: str, mode: str, src: Key, inst: Instrument, ref_octave: int = 0,
              octave: int | None = None, fix_single: bool = False, high_fs: bool = False) -> Result:
    parsed = parse(text, mode)
    notes = parsed.notes
    assign_pitches(notes, src, ref_octave, mode)
    high = HIGHEST_FS if high_fs else HIGHEST
    auto = best_octave(notes, inst, high)
    k = auto if octave is None else octave
    tgt = target_key(src, inst)

    if fix_single:
        ws, flags = fit_range(notes, inst, k, high)
    else:
        ws = [n.midi + inst.semis + 12 * k for n in notes]
        flags = ["out" if w < LOWEST or w > high else "" for w in ws]
    for n, w, f in zip(notes, ws, flags):
        n.written, n.flag = w, f

    res = Result(parsed, src, tgt, inst, k, auto, high, attach_lyrics(text, notes))
    uni, asc, names, html = [], [], [], []
    line = 1

    tuplets = 0  # open "{" triplet groups in the HTML
    slurs = 0  # open "(" slurs in the HTML

    def html_text(t: str) -> None:
        """Add plain text to the HTML view, leaving out lyrics lines (they are drawn under the notes)
        and turning "{…}" into a triplet group."""
        nonlocal line, tuplets
        for j, part in enumerate(t.split("\n")):
            if j:
                if line not in res.lyric_lines:
                    # a slur can't be one box across a line break: end it here, start it again below
                    html.append("</span>" * slurs + "\n" + '<span class="sl">' * slurs)
                line += 1
            if line in res.lyric_lines:
                continue
            for piece in re.split(r"([{}])", part):
                if piece == "{":
                    html.append('<span class="tup">')
                    tuplets += 1
                elif piece == "}":
                    if tuplets:
                        html.append("</span>")
                        tuplets -= 1
                elif piece:
                    html.append(_esc(piece))

    has_key = any(s.kind == "key" for s in parsed.segs)
    if not has_key:
        uni.append(f"1={tgt.jianpu}\n")
        asc.append(f"1={tgt.jianpu_ascii}\n")
        names.append(f"記譜：{tgt.name} 大調\n")
        html.append(f'<span class="kh">1={_esc(tgt.jianpu)}</span>\n')

    for s in parsed.segs:
        if s.kind == "text":
            uni.append(s.text); asc.append(s.text); names.append(s.text); html_text(s.text)
        elif s.kind == "key":
            uni.append(f"1={tgt.jianpu}"); asc.append(f"1={tgt.jianpu_ascii}")
            names.append(f"記譜：{tgt.name} 大調")
            html.append(f'<span class="kh">1={_esc(tgt.jianpu)}</span>')
        elif s.kind == "chord":
            c = transpose_chord(s.text, src, tgt, inst)
            uni.append(f"[{c}]"); names.append(f"[{c}]")
            asc.append(f"[{transpose_chord(s.text, src, tgt, inst, ascii_acc=True)}]")
            html.append(f'<span class="ch">{_esc(c)}</span>')
        elif s.kind == "slur":
            uni.append(s.text); asc.append(s.text); names.append(s.text)
            if s.text == "(":
                html.append('<span class="sl">')
                slurs += 1
            else:
                html.append("</span>")
                slurs -= 1
        elif s.kind == "rest":
            r = "0" + "·" * s.dot + "_" * s.ul
            uni.append(r); asc.append(r); names.append(r)
            html.append(f'<span class="n r u{s.ul}"><span class="d">0</span>{"·" * s.dot}</span>')
        else:
            n = s.note
            o = res.written_octave(n)
            rhythm = "·" * n.dot + "_" * n.ul
            uni.append(acc_sym(n.acc) + str(n.deg) + _dots(o) + rhythm)
            asc.append(acc_ascii(n.acc) + "." * max(0, -o) + str(n.deg) + "'" * max(0, o) + rhythm)
            names.append(spell(n.written, (tgt.l + n.deg - 1) % 7, ascii_acc=True) + rhythm)
            tip = f"記譜 {res.written_name(n)} · 實音 {res.concert_name(n)}"
            up = f'<span class="up">{"<i></i>" * o}</span>' if o > 0 else ""
            dn = f'<span class="dn">{"<i></i>" * -o}</span>' if o < 0 else ""
            ac = f'<span class="acc">{acc_sym(n.acc)}</span>' if n.acc else ""
            head = f'{ac}<span class="d">{n.deg}{up}{dn}</span>{"·" * n.dot}'
            if any(n.lyrics):  # stack the syllables under the note; the column is as wide as the longest one
                ly = "".join(f"<span>{_esc(t) or '&nbsp;'}</span>" for t in n.lyrics)
                head = f'<span class="nh">{head}</span><span class="ly">{ly}</span>'
            cls = f'n {n.flag} u{n.ul}{" lyr" if any(n.lyrics) else ""}'
            html.append(f'<span class="{cls}" title="{_esc(tip)}">{head}</span>')
            res.rows.append({
                "行": n.line, "第幾個音": n.idx,
                "原譜": acc_sym(n.acc) + str(n.deg) + _dots(n.oct) if mode == "jianpu" else res.concert_name(n),
                "實音": res.concert_name(n),
                "薩克斯風簡譜": acc_sym(n.acc) + str(n.deg) + _dots(o),
                "記譜音名": res.written_name(n),
                "狀態": {"out": "超出音域", "moved": "已移八度"}.get(n.flag, "可吹"),
            })

    html.append("</span>" * (tuplets + slurs))  # an unclosed "{" or "(" must not swallow the rest of the page
    res.jianpu, res.jianpu_ascii, res.names, res.html = "".join(uni), "".join(asc), "".join(names), "".join(html)
    return res


def compare_instruments(text: str, mode: str, src: Key, ref_octave: int = 0, high_fs: bool = False) -> list[dict]:
    rows = []
    for inst in INSTRUMENTS:
        r = transpose(text, mode, src, inst, ref_octave, fix_single=True, high_fs=high_fs)
        ws = [n.written for n in r.notes]
        span = "—"
        if ws:
            lo = min(r.notes, key=lambda n: n.written)
            hi = max(r.notes, key=lambda n: n.written)
            span = f"{r.written_name(lo)} – {r.written_name(hi)}"
        rows.append({
            "樂器": f"{inst.name}（{inst.tune}）",
            "記譜調": f"1={r.tgt.jianpu}（{sig_short(r.tgt.sig)}）",
            "移調規則": inst.interval,
            "實際聲音": octave_words(r.octave),
            "記譜音域": span,
            "需移八度的音": r.moved_count,
        })
    return rows


def key_from_text(name: str) -> Key | None:
    """Parse 'Eb', 'E♭', '♭E', 'F#', 'C' into a Key from SOURCE_KEYS (enharmonics resolved)."""
    m = re.fullmatch(r"\s*([#♯b♭]?)([A-Ga-g])([#♯b♭]?)\s*", name or "")
    if not m:
        return None
    k = Key(LETTERS.index(m[2].upper()), _acc_char(m[1]) or _acc_char(m[3]))
    for s in SOURCE_KEYS:
        if s == k:
            return s
    return next((s for s in SOURCE_KEYS if s.pc == k.pc), None)
