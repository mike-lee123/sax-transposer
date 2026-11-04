"""Read a score from JPG/PNG/PDF with Claude and return it as transposer input text."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

import anthropic
from PIL import Image

CLAUDE_MODEL = "claude-opus-5-5"
MAX_IMAGE_SIDE = 2400  # long-side pixels; keeps each image well under the API size limits

PROMPT = """這是一份曲譜（{n} 個檔案，依上傳順序排列）。請把旋律完整、精確地抄寫成文字，供移調程式使用。

先判斷記譜方式：
- 簡譜（數字譜）→ notation = "jianpu"
- 五線譜 → notation = "staff"

key：原曲的大調主音（實音），只寫音名，例如 "C"、"Eb"、"F#"。
- 簡譜：照譜上的「1=」寫，例如 1=♭E 就寫 "Eb"。譜上寫「D調」「D 調」「調號 D」也一樣是 "D"，
  不要用和弦或旋律去猜調性。完全找不到調號時才自行判斷，並在 notes 裡說明。
- 五線譜：依調號寫對應的大調，小調曲也寫關係大調（C 小調三個降記號寫 "Eb"）。

text 的格式規則：
- 每一行對應原譜的一行，第一行寫調號與拍號，例如「1=bE 4/4」。
- 簡譜：
  - 數字 1–7，休止符 0。
  - 低八度在數字前加英文句點，一個點一個「.」：低音 6 寫 .6，倍低音 5 寫 ..5。
  - 高八度在數字後加撇號：高音 1 寫 1'，倍高音 3 寫 3''。
  - 升降記號寫在數字前：#4、b7。
  - 延長線寫 -，小節線寫 |，附點寫 ·（不要用英文句點表示附點），緊接在數字（和八度記號）後面：5·。
- 節奏（一定要抄，不可省略）：
  - 數字下有一條底線（八分音符）在後面加一個 _，兩條底線（十六分音符）加兩個 _：5_、6__。
  - 附點加底線時，附點在前：5·_ 6__。休止符也一樣：0_、0__。
  - 同一拍、底線連在一起的音要寫在一起、中間不留空格，拍與拍之間才用空格：0_1_2_3_ 5 5_6_、5·_3__ 5 -。
  - 三連音（上方標 3）用大括號包起來：{{3'_2'_3'_}}。
  - 連音線的位置要和原譜完全一致：「(」放在弧線起點正下方那個音的前面，「)」放在弧線終點那個音的後面，
    不可以自己補、延長或合併弧線。絕對不要用括號把整句、整行或整個樂段包起來；
    連音線一般只跨 2–4 個音，不會從休止符開始，也不會跨過休止符。下列不是連音線，不要加括號：三連音上方的弧線（已用大括號表示）、
    延長記號 𝄐（弧線中間有一點）、裝飾音、反覆記號。看不清楚的弧線寧可不加，並寫進 notes。
  - 連音線（圓滑線、legato）和連結線（音符上方的弧線）用小括號把弧線下的音包起來，不可省略：(5 6)、(5_6_ 5_3_)、跨小節的 (5 - | 5 -)；
    「(」後面緊接第一個音，「)」緊接最後一個音。三連音上方的弧線和 3 都屬於三連音記號，只用大括號，不要再加小括號。
  - 沒有底線的音（四分音符以上）照常用空格分開。
- 五線譜：
  - 每個音寫成音名＋升降＋八度數字，中央 C 是 C4：C4、Eb4、F#5、Bb3。
  - 把調號套用到每個音上（例如 E♭ 大調裡的 B 要寫成 Bb）。
  - 小節線寫 |，休止符寫 0。
  - 節奏用同樣的寫法：四分音符不加記號，二分音符後面加 -（C4 -），八分音符加 _，十六分音符加 __，附點加 ·，
    同一拍的音寫在一起不留空格：C4_D4_ E4 G4·_A4__，三連音用大括號：{{C5_D5_E5_}}，
    連音線（圓滑線、legato）和連結線用小括號：(C4 D4)。
- 和弦代號放在方括號，放在它開始的那個音前面：[Cm]、[G7]、[F/A]。
- 有反覆記號、D.S.、Coda 時，照譜面順序抄寫，不要展開，並在 notes 裡說明。
- {lyrics}
- 只抄主旋律（最上面那一聲部）。

notes：列出你看不清楚、不確定、或做了判斷的地方（例如「第 3 行第 2 小節第一個音的低音點不清楚」）。沒有就給空陣列。"""

_PROPS = {
    "notation": {"type": "string", "enum": ["jianpu", "staff"]},
    "key": {"type": "string"},
    "time_signature": {"type": "string"},
    "title": {"type": "string"},
    "text": {"type": "string"},
    "notes": {"type": "array", "items": {"type": "string"}},
}
SCHEMA = {"type": "object", "properties": _PROPS, "required": list(_PROPS), "additionalProperties": False}


@dataclass
class ScoreReading:
    notation: str
    key: str
    time_signature: str
    title: str
    text: str
    notes: list[str]
    model: str


class ScoreReadError(Exception):
    pass


def _jpeg(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > MAX_IMAGE_SIDE:
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _is_pdf(name: str) -> bool:
    return name.lower().endswith(".pdf")


def _prompt(n: int, include_lyrics: bool) -> str:
    lyrics = ("把歌詞寫在對應旋律行的下一行，行首加「詞：」。歌詞要和音符一一對應：每個音一個字，字與字之間用空格分開；"
              "同一個字唱兩個以上的音（圓滑線、拖腔）時，多出來的每個音寫一個 _；休止符和延長線 - 不對應歌詞；"
              "英文以音節為單位。有第二段、第三段歌詞時，依序再寫一行「詞：」。"
              "歌詞前面的段落編號（1. 2. 3.）不要抄；「啊……」這種拖長的字只寫一次，後面的音用 _ 佔位，不要寫刪節號。"
              "歌詞裡不要出現數字。" if include_lyrics
              else "不要抄寫歌詞。")
    return PROMPT.format(n=n, lyrics=lyrics)


def _parse(text: str, model: str) -> ScoreReading:
    try:
        data = json.loads(text)
        return ScoreReading(model=model, **{k: data[k] for k in _PROPS})
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ScoreReadError("讀譜結果格式不正確，請再試一次。") from e


def read_score(files: list[tuple[str, bytes]], api_key: str | None = None,
               include_lyrics: bool = False) -> ScoreReading:
    """files: (filename, bytes) pairs in page order."""
    content = []
    for name, data in files:
        if _is_pdf(name):
            content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                            "data": base64.standard_b64encode(data).decode()}})
        else:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.standard_b64encode(_jpeg(data)).decode()}})
    content.append({"type": "text", "text": _prompt(len(files), include_lyrics)})

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    try:
        with client.beta.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=32000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise ScoreReadError("Anthropic API key 無效，請確認金鑰。") from e
    except anthropic.PermissionDeniedError as e:
        raise ScoreReadError("這個 API key 沒有使用此模型的權限。") from e
    except anthropic.RateLimitError as e:
        raise ScoreReadError("API 使用量已達上限，或帳戶額度不足，請稍後再試。") from e
    except anthropic.BadRequestError as e:
        raise ScoreReadError(f"檔案無法處理：{e.message}") from e
    except anthropic.APIStatusError as e:
        raise ScoreReadError(f"Claude API 發生錯誤（{e.status_code}），請稍後再試。") from e
    except anthropic.APIConnectionError as e:
        raise ScoreReadError("無法連線到 Claude API，請檢查網路。") from e

    if message.stop_reason == "refusal":
        raise ScoreReadError("Claude 無法處理這份檔案。")
    if message.stop_reason == "max_tokens":
        raise ScoreReadError("曲譜太長，一次讀不完。請分成較少頁數上傳。")
    return _parse(next((b.text for b in message.content if b.type == "text"), ""), message.model)
