import re
from .normalizer import TAG_RE
from .config import MAX_CHUNK_CHARS

SENTENCE_SPLIT_RE = re.compile(
    r'(?<!\b[A-Za-z]\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bSt\.)(?<!\bNo\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?।])\s+|\n+'
)

PUNCT_SPLIT_RE = re.compile(r'(?<=[,;:\-—])\s+')
CONJ_SPLIT_RE = re.compile(
    r'\s+(?=(?:and|but|or|because|while|after|before|மற்றும்|ஆனால்|எனவே)\b)',
    re.IGNORECASE,
)


def split_long_sentence(sentence: str, max_chars: int = MAX_CHUNK_CHARS):
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return [sentence]

    protected_spans = [m.span() for m in TAG_RE.finditer(sentence)]

    def _inside_tag(pos):
        return any(start <= pos < end for start, end in protected_spans)

    pieces, last_end = [], 0
    for m in PUNCT_SPLIT_RE.finditer(sentence):
        if _inside_tag(m.start()):
            continue
        p = sentence[last_end:m.start()].strip().rstrip(',;:-— ')
        if p:
            pieces.append(p)
        last_end = m.end()
    final_p = sentence[last_end:].strip().rstrip(',;:-— ')
    if final_p:
        pieces.append(final_p)

    if len(pieces) <= 1:
        pieces, last_end = [], 0
        for m in CONJ_SPLIT_RE.finditer(sentence):
            if _inside_tag(m.start()):
                continue
            p = sentence[last_end:m.start()].strip()
            if p:
                pieces.append(p)
            last_end = m.start()
        final_p = sentence[last_end:].strip()
        if final_p:
            pieces.append(final_p)

    if len(pieces) <= 1:
        words = sentence.split(" ")
        pieces, cur = [], ""
        for w in words:
            candidate = (cur + " " + w).strip() if cur else w
            if cur and len(candidate) > max_chars:
                pieces.append(cur)
                cur = w
            else:
                cur = candidate
        if cur:
            pieces.append(cur)

    grouped, current = [], ""
    for piece in pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if current and len(candidate) > max_chars:
            grouped.append(current)
            current = piece
        else:
            current = candidate
    if current:
        grouped.append(current)
    return grouped


MIN_MERGE_CHARS = 25


def split_into_sentences(text: str):
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    pieces = []
    for s in sentences:
        pieces.extend(split_long_sentence(s))

    merged = []
    current = ""
    for piece in pieces:
        candidate = (current + " " + piece) if current else piece
        if not current:
            current = piece
        elif len(piece) < MIN_MERGE_CHARS and len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
        else:
            merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged
