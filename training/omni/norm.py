"""Exact port of the competition normalizer (score.py) + our CTC target mapping."""
import re

ELL = '\u2026'  # single-symbol stand-in for word-final '...'


def _lowercase_sentence_initial(match):
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text):
    BRACKETED = re.compile(r"\[[^\]]+\]")
    UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
    WORD_PAREN = re.compile(r"\(([^()]*)\)")
    PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
    COMMA = re.compile(",+")
    SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
    SENTENCE_END = re.compile("[!?]+")
    MULTISPACE = re.compile("  +")
    text = text.replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(MULTISPACE, " ", text)
    return text


VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyzáéíóúñü'") + [ELL]
_KEEP = set(VOCAB_CHARS) | {' '}
_STRIP_EXTRA = re.compile(r"[\[\]{}|#`*<>=+$%&@_\\/~^]")


def to_target(raw):
    """raw transcript -> lowercase CTC target string using the official normalizer."""
    t = normalize_text(raw)
    t = t.replace('\u2019', "'").replace('`', "'")
    t = _STRIP_EXTRA.sub(' ', t)
    t = t.lower()
    t = re.sub(r'\.\.\.(?=\s|$)', ELL, t)            # word-final ellipsis -> one symbol
    t = t.replace('.', ' ').replace('-', ' ')          # anything else with periods/hyphens
    t = ''.join(c if c in _KEEP else ' ' for c in t)
    t = re.sub(r"\s+", ' ', t).strip()
    t = re.sub(r"(^|\s)'+(\s|$)", ' ', t)              # lone apostrophes
    t = re.sub(r"(^|\s)" + ELL + r"+(\s|$)", ' ', t)   # lone ellipses
    return re.sub(r'\s+', ' ', t).strip()


def from_model(hyp):
    """model output (lowercase, ELL symbol) -> submission text."""
    return hyp.replace(ELL, '...').strip()


def score_tokens(s):
    return normalize_text(s).split()


if __name__ == '__main__':
    for s in ["Tejua nikelnamik porque nikchiutoya... [laugh] se pio (tiyouej) ¿Ken tonijtok? Nik... ya.",
              "amantsin inka amo nikpia tomin ... uelis"]:
        print(repr(s), '->', repr(to_target(s)), '->', repr(from_model(to_target(s))), normalize_text(from_model(to_target(s))).split())
