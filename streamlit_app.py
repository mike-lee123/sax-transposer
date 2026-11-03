import hmac
import os

import altair as alt
import pandas as pd
import pymupdf
import streamlit as st

import transpose as tp
from score_pdf import build_pdf
from score_reader import ScoreReadError, list_gemini_models, read_score

st.set_page_config(page_title="薩克斯風移調器", page_icon=":material/music_note:", layout="wide")


def require_password() -> None:
    """If APP_PASSWORD is set in secrets, lock the app so a public URL can't spend the stored API keys."""
    try:
        password = st.secrets.get("APP_PASSWORD")
    except Exception:
        password = None
    if not password or st.session_state.get("unlocked"):
        return
    st.title("薩克斯風移調器")
    entered = st.text_input("請輸入密碼", type="password")
    if entered and hmac.compare_digest(entered, str(password)):
        st.session_state.unlocked = True
        st.rerun()
    if entered:
        st.error("密碼不正確。", icon=":material/lock:")
    st.stop()


require_password()

EXAMPLES = {
    "C 小調練習句（簡譜，含低音 5̣ 6̣）": (
        "jianpu",
        "1=♭E 4/4（C 小調）\n"
        "[Cm]6̣ 1 3 2 | 1 - 7̣ 6̣ | [Fm]6̣ 1 2 4 | [G7]3 - - - |\n"
        "[Cm]5̣ 6̣ 1 3 | [A♭]6 - 5 3 | [G7]2 7̣ ♯5̣ 7̣ | [Cm]6̣ - - 0 |",
    ),
    "小星星（簡譜，1=C，含歌詞）": (
        "jianpu",
        "1=C 4/4\n"
        "[C]1 1 5 5 | [F]6 6 [C]5 - | [F]4 4 [C]3 3 | [G7]2 2 [C]1 - |\n"
        "詞：一 閃 一 閃 亮 晶 晶 滿 天 都 是 小 星 星\n"
        "[C]5 5 [F]4 4 | [C]3 3 [G7]2 - | [C]5 5 [F]4 4 | [C]3 3 [G7]2 - |\n"
        "詞：掛 在 天 上 放 光 明 好 像 許 多 小 眼 睛",
    ),
    "C 小調練習句（音名）": (
        "names",
        "1=♭E 4/4\n"
        "[Cm] C4 Eb4 G4 F4 | Eb4 - D4 C4 | [Fm] C4 Eb4 F4 Ab4 | [G7] G4 - - - |\n"
        "[Cm] Bb3 C4 Eb4 G4 | [Ab] C5 - Bb4 G4 | [G7] F4 D4 B3 D4 | [Cm] C4 - - - |",
    ),
}
MODES = {"jianpu": "簡譜", "names": "音名（五線譜）"}
OCTAVES = {None: "自動（選最好吹的八度）", 0: "與原曲同音高", -1: "比原曲低一個八度",
           1: "比原曲高一個八度", -2: "比原曲低兩個八度"}

# ---------- state ----------
if "score_text" not in st.session_state:
    mode, text = next(iter(EXAMPLES.values()))
    st.session_state.score_text = text
    st.session_state.mode = mode
    st.session_state.reading = None
    st.session_state.read_error = None


PROVIDERS = {"claude": "Claude", "gemini": "Gemini"}
KEY_NAMES = {"claude": ["ANTHROPIC_API_KEY"], "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]}


def get_api_key(provider: str) -> str | None:
    for name in KEY_NAMES[provider]:
        try:
            key = st.secrets.get(name)
        except Exception:
            key = None
        if key or os.environ.get(name):
            return key or os.environ.get(name)
    return st.session_state.get(f"api_key_{provider}") or None


@st.cache_data(max_entries=30, ttl="1d", show_spinner=False)
def cached_read(files: tuple[tuple[str, bytes], ...], include_lyrics: bool, provider: str,
                model: str | None, _api_key: str | None):
    return read_score(list(files), provider=provider, api_key=_api_key, model=model, include_lyrics=include_lyrics)


@st.cache_data(max_entries=50, show_spinner=False)
def cached_key_options(text: str, mode: str, src: tp.Key, inst_id: str, ref: int, high_fs: bool) -> list[dict]:
    return tp.key_options(text, mode, src, tp.INSTRUMENT_BY_ID[inst_id], ref, high_fs)


@st.cache_data(ttl="1h", max_entries=5, show_spinner=False)
def gemini_models(_api_key: str | None, key_tag: str) -> tuple[list[str], str | None]:
    return list_gemini_models(_api_key)


@st.cache_data(max_entries=30, show_spinner=False)
def pdf_previews(data: bytes, max_pages: int = 4) -> list[bytes]:
    doc = pymupdf.open(stream=data, filetype="pdf")
    return [doc[i].get_pixmap(dpi=80).tobytes("png") for i in range(min(max_pages, doc.page_count))]


def load_example():
    name = st.session_state.example
    if name:
        st.session_state.mode, st.session_state.score_text = EXAMPLES[name]
        st.session_state.reading = None
    st.session_state.example = None


def run_reader():
    uploads = st.session_state.get("uploads") or []
    st.session_state.read_error = None
    provider = st.session_state.get("provider", "claude")
    model = st.session_state.get("gemini_model") if provider == "gemini" else None
    try:
        reading = cached_read(tuple((f.name, f.getvalue()) for f in uploads),
                              st.session_state.get("lyrics", False), provider, model, get_api_key(provider))
    except ScoreReadError as e:
        st.session_state.read_error = str(e)
        return
    st.session_state.reading = reading
    st.session_state.mode = "jianpu" if reading.notation == "jianpu" else "names"
    text = reading.text
    if not tp.R_KEY.search(text) and tp.key_from_text(reading.key):
        text = f"1={tp.key_from_text(reading.key).jianpu}\n" + text
    st.session_state.score_text = text


# ---------- sidebar: settings ----------
with st.sidebar:
    st.header("設定", divider="orange")
    inst_id = st.segmented_control(
        "樂器", [i.id for i in tp.INSTRUMENTS], default="soprano", required=True, key="inst",
        format_func=lambda i: f"{tp.INSTRUMENT_BY_ID[i].name.replace('薩克斯風', '')} {tp.INSTRUMENT_BY_ID[i].tune}",
        width="stretch",
    )
    inst = tp.INSTRUMENT_BY_ID[inst_id]
    key_choice = st.selectbox(
        "原曲調性（實音）", ["auto"] + list(range(len(tp.SOURCE_KEYS))), key="src_key",
        format_func=lambda i: "自動（依譜中的 1=）" if i == "auto"
        else f"{tp.SOURCE_KEYS[i].label}　1={tp.SOURCE_KEYS[i].jianpu}",
    )
    play_choice = st.selectbox(
        "演奏調性（移到別的調）", ["same", "auto"] + list(range(len(tp.SOURCE_KEYS))), key="play_key",
        format_func=lambda i: {"same": "與原曲相同", "auto": "自動推薦（這把樂器最好吹的調）"}.get(i)
        or f"{tp.SOURCE_KEYS[i].label}　1={tp.SOURCE_KEYS[i].jianpu}",
        help="整首歌移到另一個調演奏（例如配合歌手的音域，或讓旋律落在比較好吹的音區），"
             "會往最近的方向移，最多六個半音。和弦一起移調。"
             "「自動推薦」會挑出全部音都在 D4–A5 好吹音區、調號最簡單的調。",
    )
    ref_octave = st.selectbox(
        "簡譜中音 1 的實際音高", [0, -1, 1], key="ref_oct",
        format_func=lambda r: {0: "中央 C 到上方 B 之間（預設）", -1: "低一個八度", 1: "高一個八度"}[r],
        help="簡譜沒有絕對音高。預設把中音 1 放在中央 C（C4）到 B4 之間，適合一般人聲旋律。",
    )
    octave = st.selectbox("實際聲音的八度", list(OCTAVES), format_func=OCTAVES.get, key="octave")
    fix_single = st.toggle("把超出音域的單音自動移八度", value=True, key="fix",
                           help="開啟時保證每個音都在樂器音域內，被移動的音以橘色標示。"
                                "常見原因是讀譜時八度點讀錯，可以對照原譜修正曲譜文字。")
    high_fs = st.toggle("我的樂器有高音 F♯ 鍵", key="high_fs")

    st.header("讀譜", divider="orange")
    provider = st.segmented_control("讀譜 AI", list(PROVIDERS), format_func=PROVIDERS.get, default="claude",
                                    required=True, key="provider", width="stretch")
    env_hint = " 或 ".join(KEY_NAMES[provider])
    if get_api_key(provider) is None or st.session_state.get(f"api_key_{provider}"):
        st.text_input(
            "Anthropic API key" if provider == "claude" else "Gemini API key", type="password",
            key=f"api_key_{provider}",
            help=f"上傳 JPG／PDF 時需要。也可以設定環境變數 {env_hint}，或寫在 .streamlit/secrets.toml。",
        )
    else:
        st.caption(f":material/check_circle: 已從環境變數或 secrets 讀到 {PROVIDERS[provider]} API key")
    if provider == "gemini":
        key = get_api_key("gemini")
        models, list_error = gemini_models(key, str(hash(key)))
        if list_error:
            gemini_models.clear()  # don't keep the fallback list for an hour; retry on the next rerun
            if key:
                st.caption(f":orange[無法取得模型清單，暫用預設模型。原因：{list_error}]")
        st.selectbox("Gemini 模型", models, key="gemini_model",
                     help="清單來自你的 Gemini 帳戶。flash 模型速度快、有免費額度；pro 模型通常讀得較仔細。")
        st.caption("免費方案有每分鐘、每天的次數限制，且依 Google 條款，資料可能被用來改進其產品。")
    else:
        st.caption("使用 Claude Opus 5.5，依用量向你的 Anthropic 帳戶收費。")

# ---------- main ----------
st.title("薩克斯風移調器")
st.caption("上傳 JPG／PDF 曲譜或貼上簡譜，轉成高音、中音、次中音、上低音薩克斯風可以直接吹奏的記譜，並檢查音域。")

left, right = st.columns([5, 7], gap="large")

with left:
    with st.container(border=True):
        st.subheader(":material/upload_file: 上傳曲譜", anchor=False)
        uploads = st.file_uploader(
            "JPG、PNG 或 PDF（可一次上傳多頁，依檔名順序讀取）",
            type=["jpg", "jpeg", "png", "webp", "pdf"], accept_multiple_files=True, key="uploads",
        )
        if uploads:
            with st.container(horizontal=True):
                for f in uploads[:6]:
                    if f.name.lower().endswith(".pdf"):
                        for i, png in enumerate(pdf_previews(f.getvalue())):
                            st.image(png, width=110, caption=f"{f.name} 第 {i + 1} 頁", alt=f"{f.name} 第 {i + 1} 頁預覽")
                    else:
                        st.image(f, width=110, caption=f.name, alt=f"{f.name} 預覽")
        has_key = get_api_key(provider) is not None
        st.toggle("一起抄寫歌詞（對齊在音符下方）", value=True, key="lyrics")
        st.button(f"用 {PROVIDERS[provider]} 讀譜", icon=":material/auto_awesome:", type="primary",
                  on_click=run_reader, disabled=not uploads or not has_key, width="stretch")
        if uploads and not has_key:
            st.caption(f"請先在左側欄填入 {PROVIDERS[provider]} API key。")
        if st.session_state.read_error:
            st.error(st.session_state.read_error, icon=":material/error:")
        reading = st.session_state.reading
        if reading:
            st.success(f"已讀取{('《' + reading.title + '》') if reading.title else '曲譜'}："
                       f"{'簡譜' if reading.notation == 'jianpu' else '五線譜'}，{reading.key} 大調，"
                       f"{reading.time_signature}。請核對下方文字後再使用。", icon=":material/task_alt:")
            if st.session_state.get("lyrics") and "詞" not in reading.text:
                st.warning("這次沒有讀到歌詞。如果曲譜上有歌詞，請再按一次讀譜，或在下方「曲譜文字」自行加上「詞：」行。",
                           icon=":material/lyrics:")
            if reading.notes:
                with st.expander(f"AI 標註的不確定之處（{len(reading.notes)}）", icon=":material/help:"):
                    for n in reading.notes:
                        st.markdown(f"- {n}")

    with st.container(border=True):
        st.subheader(":material/edit_note: 曲譜文字（實音）", anchor=False)
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.segmented_control("輸入格式", list(MODES), format_func=MODES.get, required=True, key="mode")
            st.selectbox("載入範例", [None] + list(EXAMPLES), key="example", on_change=load_example,
                         format_func=lambda x: "選擇範例…" if x is None else x)
        st.text_area("曲譜文字", key="score_text", height=220, label_visibility="collapsed")
        with st.expander("輸入規則", icon=":material/info:"):
            st.markdown(
                "- 調號寫在開頭：`1=C`、`1=♭E`、`1=bE`、`1=#F`\n"
                "- 低音：`.6` 或 `6̣`；高音：`6'` 或 `6̇`；升降：`#4`、`b7`\n"
                "- 音名格式：`C4`（中央 C）、`Eb4`、`F#5`\n"
                "- 和弦放在方括號：`[Cm]`、`[G7]`、`[F/A]`，會一起移調\n"
                "- 節奏：八分音符 `5_`，十六分音符 `5__`，附點 `5·`（附點八分 `5·_`），休止符同樣 `0_`。"
                "同一拍的音連著寫、不留空格，底線就會連成一條：`0_1_2_3_`、`5·_3__`。三連音用大括號：`{3'_2'_3'_}`\n"
                "- 連音線（圓滑線、legato）與連結線：用小括號把弧線下的音包起來，`(5 6)`、`(5 - | 5 -)`。三連音的弧線會自動畫出，不必加括號\n"
                "- 歌詞：寫在旋律行的下一行，行首加 `詞：`，一個音一個字、用空格分開；"
                "一個字唱多個音時，多出的音寫 `_`。第二段歌詞再寫一行 `詞：`。PDF 會把字對齊在音符下方\n"
                "- `0`、`-`、`|`、附點 `·` 等其他文字原樣保留。附點請用 `·`，`5.6` 會被讀成「5 和低音 6」"
            )

mode = st.session_state.mode
text = st.session_state.score_text
header = tp.parse(text, mode).header
if key_choice == "auto":
    src = header and tp.key_from_text(header.name.replace("♯", "#").replace("♭", "b")) or tp.SOURCE_KEYS[0]
else:
    src = tp.SOURCE_KEYS[key_choice]
ref = ref_octave if mode == "jianpu" else 0
options = cached_key_options(text, mode, src, inst_id, ref, high_fs)
if play_choice == "auto" and options:
    play = options[0]["key"]
    if octave is None:
        octave = options[0]["octave"]
else:
    play = None if play_choice in ("same", "auto") else tp.SOURCE_KEYS[play_choice]
res = tp.transpose(text, mode, src, inst, ref_octave if mode == "jianpu" else 0, octave, fix_single, high_fs, play)

with right:
    notes = res.notes
    if not notes:
        st.info("還沒有讀到任何音符。請上傳曲譜，或在左邊貼上簡譜／音名。", icon=":material/music_off:")
    else:
        if res.out_count:
            status = f"{res.out_count} 個音超出"
        elif res.moved_count:
            status = f"可吹（{res.moved_count} 個移八度）"
        else:
            status = "全部可吹"
        with st.container(horizontal=True):
            st.metric("記譜調", f"1={res.tgt.jianpu}", border=True)
            st.metric("調號", tp.sig_short(res.tgt.sig), border=True)
            st.metric("實際聲音", tp.octave_words(res.octave), border=True)
            st.metric("音域檢查", status, border=True)

        auto_note = "（自動選擇）" if octave is None else ""
        st.markdown(
            f"**原曲** 1={src.jianpu}（{src.label}）"
            + (f"→ **移到** 1={res.src.jianpu}（{'升' if res.shift > 0 else '降'} {abs(res.shift)} 個半音）"
               if res.orig else "")
            + f"→ **{inst.name}記譜** 1={res.tgt.jianpu}"
            f"（{res.tgt.label}，{tp.sig_long(res.tgt.sig)}）  \n"
            f"每個音{inst.interval}；實際聲音{tp.octave_words(res.octave)}{auto_note}。"
            + ("" if key_choice != "auto" or header else "  \n:orange[譜中沒有寫 1=，暫時當作 C 調。請在左側欄選擇原曲調性。]")
        )
        if options:
            best = options[0]
            if play_choice == "auto":
                st.success(f"自動推薦：{best['演奏調性']}，{inst.name}記譜 {best['記譜調']}，音域 {best['記譜音域']}。",
                           icon=":material/recommend:")
            with st.expander(f"{inst.name}吹這首歌，各調好不好吹", icon=":material/tune:",
                             expanded=play_choice == "auto"):
                st.dataframe(
                    pd.DataFrame(options).drop(columns=["key", "octave", "score"]), hide_index=True,
                    column_config={
                        "太低或太高的音": st.column_config.NumberColumn(
                            help="低於 D4（要用左手小指的低音鍵）或高於 A5 的音", format="%d"),
                        "需移八度的音": st.column_config.NumberColumn(format="%d"),
                    },
                    alt="這首歌移到各調時的記譜調、音域與難吹的音數，最好吹的排在最前面",
                )
                st.caption("由上到下越好吹：先看要不要移八度，再看太低、太高的音，再看調號的升降記號數，"
                           "最後看離原調多遠。想用哪一個，就在左側欄「演奏調性」選它。")

        with st.container(border=True):
            st.subheader(f":material/music_note: {inst.name}譜", anchor=False)
            st.html(
                "<style>"
                ".jp{font-size:22px;line-height:2.7;white-space:pre-wrap;word-break:break-word;font-variant-numeric:tabular-nums}"
                ".jp .n{display:inline-block;position:relative;padding:0 .1em}.jp .acc{font-size:.6em;vertical-align:.55em;margin-right:1px}"
                ".jp .d{position:relative;display:inline-block}"
                ".jp .up,.jp .dn{position:absolute;left:50%;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;gap:2px}"
                ".jp .up{bottom:calc(100% - .55em)}.jp .dn{top:calc(100% - .5em)}"
                ".jp i{display:block;width:4px;height:4px;border-radius:50%;background:currentColor}"
                ".jp .out .d{color:#e5484d;background:rgba(229,72,77,.15);border-radius:4px;padding:0 2px}"
                ".jp .moved .d{color:#d97706}"
                # rhythm: underlines of neighbouring notes touch, so a beat written without spaces is one beam
                ".jp .u1::after,.jp .u2::after,.jp .u3::after,.jp .u2::before,.jp .u3::before{content:'';position:absolute;"
                "left:0;right:0;border-top:.08em solid currentColor}"
                ".jp .n::after{top:1.86em}.jp .n::before{top:2.04em}"
                ".jp .u1 .dn{top:calc(100% - .3em)}.jp .u2 .dn,.jp .u3 .dn{top:calc(100% - .12em)}"
                # slur, tie and triplet: the top edge of a flattened ellipse over the group
                ".jp .sl,.jp .tup{position:relative;display:inline-block}"
                ".jp .sl:has(.lyr),.jp .tup:has(.lyr){vertical-align:top}"
                ".jp .sl::before,.jp .tup::after{content:'';position:absolute;left:.2em;right:.2em;top:.02em;"
                "height:.45em;border-top:.07em solid currentColor;border-radius:50%/100% 100% 0 0}"
                ".jp .tup::before{content:'3';position:absolute;left:50%;top:-.36em;transform:translateX(-50%);"
                "font-size:.5em;line-height:1;font-style:italic;font-weight:700}"
                ".jp .ch{font-size:.58em;font-weight:700;color:#b07d2b;vertical-align:.95em;margin-right:2px}"
                ".jp .kh{font-size:.8em;font-weight:700}"
                ".jp .lyr{display:inline-flex;flex-direction:column;align-items:center;vertical-align:top}"
                ".jp .ly{display:flex;flex-direction:column;align-items:center;margin:-.1em .15em .3em;"
                "font-size:.62em;line-height:1.5;font-weight:400;white-space:nowrap}"
                "</style>"
                f'<div class="jp">{res.html}</div>'
            )
            st.caption("簡譜八度以記譜 C4–B4 之間的 1 為中音。滑鼠移到音符上可看記譜音名和實音。"
                       "紅色＝超出音域，橘色＝已移八度。底線＝八分音符（兩條＝十六分），上方的 3＝三連音。")
            t1, t2, t3 = st.tabs(["簡譜（點號）", "簡譜（純文字）", "記譜音名"])
            t1.code(res.jianpu, language=None, wrap_lines=True)
            t2.code(res.jianpu_ascii, language=None, wrap_lines=True)
            t3.code(res.names, language=None, wrap_lines=True)
            title = st.session_state.reading.title if st.session_state.reading else ""
            with st.container(horizontal=True):
                st.download_button(
                    "下載 PDF 曲譜", icon=":material/picture_as_pdf:", type="primary",
                    data=build_pdf(res, title), file_name=f"sax-{inst.id}-1={res.tgt.jianpu_ascii}.pdf",
                    mime="application/pdf",
                )
                st.download_button(
                    "下載文字檔", icon=":material/download:",
                    data=f"{inst.name} 記譜 1={res.tgt.jianpu}\n\n{res.jianpu}\n\n--- 純文字 ---\n{res.jianpu_ascii}\n\n--- 記譜音名 ---\n{res.names}\n",
                    file_name=f"sax-{inst.id}-1={res.tgt.jianpu_ascii}.txt", mime="text/plain",
                )

        with st.container(border=True):
            st.subheader(":material/straighten: 記譜音域檢查", anchor=False)
            lo_n = min(notes, key=lambda n: n.written)
            hi_n = max(notes, key=lambda n: n.written)
            bars = pd.DataFrame([
                {"項目": "樂器可吹範圍", "low": tp.LOWEST, "high": res.high,
                 "說明": f"B♭3 – {'F♯6' if res.high == tp.HIGHEST_FS else 'F6'}"},
                {"項目": "這首曲子", "low": lo_n.written, "high": hi_n.written,
                 "說明": f"{res.written_name(lo_n)} – {res.written_name(hi_n)}"},
            ])
            labels = {58: "B♭3", 60: "C4", 72: "C5", 84: "C6", 89: "F6"}
            expr = "{" + ",".join(f"{k}:'{v}'" for k, v in labels.items()) + "}[datum.value]"
            lo_axis = min(54, lo_n.written - 2)
            hi_axis = max(93, hi_n.written + 2)
            chart = alt.Chart(bars).mark_bar(cornerRadius=4, height=22).encode(
                x=alt.X("low:Q", scale=alt.Scale(domain=[lo_axis, hi_axis]), title=None,
                        axis=alt.Axis(values=list(labels), labelExpr=expr, grid=True)),
                x2="high:Q",
                y=alt.Y("項目:N", title=None, sort=None),
                color=alt.Color("項目:N", legend=None,
                                scale=alt.Scale(range=["#3a9d6a", "#b07d2b"])),
                tooltip=["項目", "說明"],
            ).properties(height=110)
            st.altair_chart(chart, alt="樂器記譜音域與這首曲子用到的音域比較")
            only_bad = st.toggle("只顯示有問題的音", value=res.out_count > 0, key="only_bad")
            rows = [r for r in res.rows if r["狀態"] != "可吹"] if only_bad else res.rows
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(400, 38 + 35 * max(1, len(rows))),
                         alt="每個音的原譜、實音、薩克斯風記譜與音域狀態")
            if res.out_count:
                st.warning("處理方式：在左側欄改變「實際聲音的八度」、把那一整句移八度、開啟「自動移八度」，或換一種薩克斯風。",
                           icon=":material/lightbulb:")

        with st.container(border=True):
            st.subheader(":material/compare_arrows: 哪一種薩克斯風最適合？", anchor=False)
            cmp = pd.DataFrame(tp.compare_instruments(text, mode, src, ref_octave if mode == "jianpu" else 0, high_fs, play))
            st.dataframe(
                cmp, hide_index=True,
                column_config={"需移八度的音": st.column_config.NumberColumn("需移八度的音（個）", format="%d")},
                alt="四種薩克斯風吹這首曲子的記譜調、八度與音域比較",
            )
            st.caption("每種樂器都已自動選擇最好吹的八度。")

with st.expander("調號速查表（實音 → 記譜）", icon=":material/table_chart:"):
    ref_rows = []
    for k in tp.SOURCE_KEYS:
        if k == tp.Key(4, -1):  # G♭ duplicates F♯
            continue
        row = {"原曲（實音）": f"{k.name} 大調 / {k.relative_minor.name} 小調"}
        for i in tp.INSTRUMENTS:
            t = tp.target_key(k, i)
            row[f"{i.name.replace('薩克斯風', '')} {i.tune}"] = f"{t.name}（{tp.sig_short(t.sig)}）"
        ref_rows.append(row)
    st.dataframe(pd.DataFrame(ref_rows), hide_index=True, alt="各調在四種薩克斯風上的記譜調")
    st.caption("所有薩克斯風的記譜音域都是 B♭3–F6，指法相同，差別只在實際發出的音高。")
