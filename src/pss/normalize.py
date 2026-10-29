"""Persian text normalization.

Every step is an independent flag so that its effect on retrieval quality can be
measured in isolation (see ``PRESETS`` and the ablation in ``bench.py``).

Steps (applied in this order):
  chars        NFKC + unify Arabic/Persian letter variants (ي->ی, ك->ک, ة->ه, ۀ->ه, أإٱ->ا)
  diacritics   strip harakat (fatha, kasra, ...) and tatweel (kashida)
  digits       map Persian (۰-۹) and Arabic-Indic (٠-٩) digits to one script
  punct        unify Arabic punctuation (؟ ، ؛ « ») to ASCII
  zwnj_mode    "fix"   -> canonical half-space usage (می شود -> می‌شود, کتاب ها -> کتاب‌ها,
                          stray/duplicate ZWNJ removed)
               "space" -> replace every ZWNJ with a plain space
               "off"   -> leave untouched
  whitespace   collapse runs of whitespace, strip
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

ZWNJ = "\u200c"
_AR = r"\u0600-\u06FF"  # Arabic script block (covers Persian letters)

# --- character maps ---------------------------------------------------------
_CHAR_MAP = str.maketrans(
    {
        "\u064A": "\u06CC",  # ي  Arabic yeh        -> ی
        "\u0649": "\u06CC",  # ى  alef maksura      -> ی
        "\u0643": "\u06A9",  # ك  Arabic kaf        -> ک
        "\u06AA": "\u06A9",  # ڪ  swash kaf         -> ک
        "\u0629": "\u0647",  # ة  teh marbuta       -> ه
        "\u06C0": "\u0647",  # ۀ  heh with yeh above-> ه
        "\u06BE": "\u0647",  # ھ  heh doachashmee   -> ه
        "\u0623": "\u0627",  # أ  -> ا
        "\u0625": "\u0627",  # إ  -> ا
        "\u0671": "\u0627",  # ٱ  -> ا
    }
)
_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u0640]")  # harakat + superscript alef + tatweel

_ASCII = "0123456789"
_FA = "۰۱۲۳۴۵۶۷۸۹"
_AR_INDIC = "٠١٢٣٤٥٦٧٨٩"
_SEPARATORS = {"\u066B": ".", "\u066C": ","}  # Arabic decimal / thousands separators
_DIGITS_TO_ASCII = str.maketrans({**dict(zip(_FA, _ASCII)), **dict(zip(_AR_INDIC, _ASCII)), **_SEPARATORS})
_DIGITS_TO_FA = str.maketrans({**dict(zip(_ASCII, _FA)), **dict(zip(_AR_INDIC, _FA)), **_SEPARATORS})

_PUNCT = str.maketrans({"؟": "?", "،": ",", "؛": ";", "«": '"', "»": '"', "٪": "%"})

# --- ZWNJ handling ----------------------------------------------------------
_INVISIBLE = re.compile("[\u200b\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_ZWNJ_RUN = re.compile(f"{ZWNJ}{{2,}}")
# keep a ZWNJ only when it sits between two Arabic-script characters
_ZWNJ_STRAY = re.compile(f"(?<![{_AR}]){ZWNJ}+|{ZWNJ}+(?![{_AR}])")
# verb prefix  می / نمی  +  space  +  word  ->  می‌word
_MI_PREFIX = re.compile(f"(?<![{_AR}])(ن?می) (?=[{_AR}])")
# plural / comparative suffix separated by a space -> half-space
_SUFFIX = re.compile(f"(?<=[{_AR}]) (هایی|های|ها|ترین|تری|تر)(?![{_AR}])")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class NormConfig:
    chars: bool = True
    diacritics: bool = True
    digits: str | None = "ascii"  # "ascii" | "fa" | None
    punct: bool = True
    zwnj_mode: str = "fix"  # "fix" | "space" | "off"
    whitespace: bool = True

    def __post_init__(self) -> None:
        if self.digits not in ("ascii", "fa", None):
            raise ValueError(f"digits must be 'ascii', 'fa' or None, got {self.digits!r}")
        if self.zwnj_mode not in ("fix", "space", "off"):
            raise ValueError(f"zwnj_mode must be 'fix', 'space' or 'off', got {self.zwnj_mode!r}")

    @classmethod
    def none(cls) -> NormConfig:
        return cls(chars=False, diacritics=False, digits=None, punct=False, zwnj_mode="off", whitespace=False)


_OFF = NormConfig.none()
PRESETS: dict[str, NormConfig] = {
    "none": _OFF,
    "chars": replace(_OFF, chars=True),
    "diacritics": replace(_OFF, diacritics=True),
    "digits": replace(_OFF, digits="ascii"),
    "punct": replace(_OFF, punct=True),
    "zwnj_fix": replace(_OFF, zwnj_mode="fix"),
    "zwnj_space": replace(_OFF, zwnj_mode="space"),
    "whitespace": replace(_OFF, whitespace=True),
    "full": NormConfig(),
}


def preset(name: str) -> NormConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown normalization preset {name!r}; choose from {sorted(PRESETS)}") from None


def _fix_zwnj(text: str) -> str:
    text = _ZWNJ_RUN.sub(ZWNJ, text)
    text = _ZWNJ_STRAY.sub("", text)
    text = _MI_PREFIX.sub(lambda m: m.group(1) + ZWNJ, text)
    text = _SUFFIX.sub(lambda m: ZWNJ + m.group(1), text)
    return text


def normalize(text: str, cfg: NormConfig | None = None) -> str:
    """Normalize Persian text according to ``cfg`` (default: all steps on)."""
    if not text:
        return ""
    cfg = cfg or NormConfig()
    if cfg.chars:
        text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    if cfg.diacritics:
        text = _DIACRITICS.sub("", text)
    if cfg.digits == "ascii":
        text = text.translate(_DIGITS_TO_ASCII)
    elif cfg.digits == "fa":
        text = text.translate(_DIGITS_TO_FA)
    if cfg.punct:
        text = text.translate(_PUNCT)
    if cfg.zwnj_mode != "off":
        text = _INVISIBLE.sub("", text)
        text = text.replace(ZWNJ, " ") if cfg.zwnj_mode == "space" else _fix_zwnj(text)
    if cfg.whitespace:
        text = _WS.sub(" ", text).strip()
    return text
