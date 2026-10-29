import pytest

from pss.normalize import ZWNJ, NormConfig, normalize, preset


def test_arabic_letters_unified():
    assert normalize("علي كتاب") == "علی کتاب"
    assert normalize("مدرسة") == "مدرسه"


def test_digits_to_ascii_and_fa():
    assert normalize("سال ۱۴۰۳ و ٢٠٢٤") == "سال 1403 و 2024"
    assert normalize("سال 2024", NormConfig(digits="fa")) == "سال ۲۰۲۴"
    assert normalize("٣٫١٤") == "3.14"


def test_diacritics_and_tatweel_removed():
    assert normalize("مُحَمَّد") == "محمد"
    assert normalize("سلاـــم") == "سلام"


def test_zwnj_fix_verb_prefix_and_plural():
    assert normalize("می شود") == f"می{ZWNJ}شود"
    assert normalize("نمی دانم") == f"نمی{ZWNJ}دانم"
    assert normalize("كتاب ها") == f"کتاب{ZWNJ}ها"
    assert normalize("بزرگ تر") == f"بزرگ{ZWNJ}تر"


def test_zwnj_fix_leaves_normal_words_alone():
    assert normalize("تنها") == "تنها"
    assert normalize("ما می رویم") == f"ما می{ZWNJ}رویم"
    assert normalize("این هاله") == "این هاله"  # 'ها' is only a suffix when followed by a non-letter


def test_zwnj_cleanup():
    assert normalize(f"می{ZWNJ}{ZWNJ}شود") == f"می{ZWNJ}شود"
    assert normalize(f"{ZWNJ}سلام{ZWNJ}") == "سلام"
    assert normalize(f"سلام {ZWNJ}دنیا") == "سلام دنیا"
    assert normalize(f"12{ZWNJ}34") == "1234"


def test_zwnj_space_mode():
    assert normalize(f"می{ZWNJ}شود", NormConfig(zwnj_mode="space")) == "می شود"


def test_punct_and_whitespace():
    assert normalize("چطوری؟  «خوبم»،   ممنون") == 'چطوری? "خوبم", ممنون'
    assert normalize("a\u00a0 b\n\n c") == "a b c"


def test_none_preset_is_identity():
    s = f"كتاب ها ۱۲۳ می{ZWNJ}شود  ؟"
    assert normalize(s, preset("none")) == s


def test_idempotent():
    s = "كتاب ها می شود ۱۲۳ مُحَمَّد؟  "
    once = normalize(s)
    assert normalize(once) == once


def test_empty():
    assert normalize("") == ""


def test_bad_config():
    with pytest.raises(ValueError):
        NormConfig(zwnj_mode="bogus")
    with pytest.raises(ValueError):
        preset("nope")
