import re
from indic_numtowords import num2words as indic_num2words

TAMIL_RE = re.compile(r'[஀-௿]')

EN_DIGIT_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

EN_LETTER_NAMES = {c: c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}

PRONUNCIATION_OVERRIDES = {
    "mins": "minutes",
    "min": "minutes",
    "sec": "seconds",
    "secs": "seconds",
    "hrs": "hours",
    "hr": "hour",
    "km": "kilometers",
    "kms": "kilometers",
    "kg": "kilograms",
    "kgs": "kilograms",
    "gm": "grams",
    "gms": "grams",
    "ltr": "liters",
    "ltrs": "liters",
    "amt": "amount",
    "approx": "approximately",
    "govt": "government",
    "dept": "department",
    "est": "estimated",
    "avg": "average",
    "info": "information",
    "nos": "numbers",
    "appt": "appointment",
    "addr": "address",
    "msg": "message",
    "txn": "transaction",
    "acct": "account",
    "bal": "balance",
    "qty": "quantity",
    "ref": "reference",
    "ID": "I D",
    "Journey": "ஜர்னி",
}

PRONUNCIATION_OVERRIDE_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in PRONUNCIATION_OVERRIDES) + r')\b',
    re.IGNORECASE,
)

TAG_RE = re.compile(r'\{\{(OTP|PHONE|AMOUNT|TIME|CODE|NUM):([^}]+)\}\}', re.IGNORECASE)

STATE_CODES = "AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UA|UK|UP|WB"
INDIAN_CAR_RE = re.compile(
    rf'\b({STATE_CODES})[ -]?([0-9]{{1,2}})[ -]?([A-Za-z]{{1,3}})?[ -]?([0-9]{{1,4}})\b',
    re.IGNORECASE,
)
BH_CAR_RE = re.compile(r'\b([0-9]{2})[ -]?(BH)[ -]?([0-9]{4})[ -]?([A-Za-z]{1,2})\b', re.IGNORECASE)
ALNUM_CODE_RE = re.compile(
    r'\b(?=[A-Za-z0-9]{5,12}\b)(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{5,12}\b'
)
TIME_RE = re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)\b')
LONG_DIGIT_RE = re.compile(r'\b\d{4,}\b')
SHORT_NUM_RE = re.compile(r'\b\d{1,3}\b')


def contains_tamil(text: str) -> bool:
    return bool(TAMIL_RE.search(text))


def spell_digits(digit_str: str) -> str:
    return " ".join(EN_DIGIT_WORDS[d] for d in digit_str if d.isdigit())


def spell_alnum_code(code: str) -> str:
    parts = []
    prev = None
    for ch in code:
        if ch.isdigit():
            parts.append(EN_DIGIT_WORDS[ch])
            prev = ch
        elif ch.isalpha():
            letter = EN_LETTER_NAMES.get(ch.upper(), ch.upper())
            if prev and prev.upper() == ch.upper():
                parts[-1] = f"double {letter}"
            else:
                parts.append(letter)
            prev = ch
    return " ".join(parts)


def spell_time(hh: str, mm: str) -> str:
    h = int(hh) % 24
    m = int(mm)
    period = "A M" if h < 12 else "P M"
    h12 = h % 12
    h12 = 12 if h12 == 0 else h12
    hour_word = indic_num2words(h12, lang="en")
    if m == 0:
        return f"{hour_word} o'clock {period}"
    elif m < 10:
        return f"{hour_word} oh {indic_num2words(m, lang='en')} {period}"
    else:
        return f"{hour_word} {indic_num2words(m, lang='en')} {period}"


def spell_amount(num_str: str, lang: str = "en") -> str:
    cleaned = re.sub(r'[^\d.]', '', str(num_str)).strip()
    if not cleaned:
        return num_str
    parts = cleaned.split('.')
    whole = parts[0] or '0'
    result = indic_num2words(int(whole), lang="en")
    if len(parts) > 1 and parts[1]:
        decimal = parts[1]
        result += " point " + " ".join(
            indic_num2words(int(d), lang="en") for d in decimal
        )
    return result


def apply_pronunciation_overrides(text: str) -> str:
    def _replace(m):
        matched = m.group(0)
        for key, val in PRONUNCIATION_OVERRIDES.items():
            if key.lower() == matched.lower():
                return val
        return matched
    return PRONUNCIATION_OVERRIDE_RE.sub(_replace, text)


def expand_tag(match: re.Match, lang: str = "en") -> str:
    kind, value = match.group(1).upper(), match.group(2).strip()
    if kind in ("OTP", "PHONE"):
        return spell_digits(value)
    elif kind in ("AMOUNT", "NUM"):
        return spell_amount(value, lang="en")
    elif kind == "TIME":
        hh, mm = value.split(":")
        return spell_time(hh, mm)
    elif kind == "CODE":
        clean_code = re.sub(r'[\s-]+', '', value)
        return spell_alnum_code(clean_code)
    return value


def normalize_tagged_text(text: str) -> str:
    def _sub(m):
        expansion = expand_tag(m, lang="en")
        if m.group(1).upper() in ("OTP", "PHONE", "CODE"):
            preceding = text[:m.start()].rstrip()
            if preceding and preceding[-1] not in ",:.!?":
                return ", " + expansion
        return expansion
    return TAG_RE.sub(_sub, text)


def _replace_car_match(m: re.Match) -> str:
    clean_car = re.sub(r'[\s-]+', '', m.group(0)).upper()
    return spell_alnum_code(clean_car)


def heuristic_normalize(text: str) -> str:
    text = INDIAN_CAR_RE.sub(_replace_car_match, text)
    text = BH_CAR_RE.sub(_replace_car_match, text)

    def _comma_before(pattern_sub_fn):
        def _sub(m):
            expansion = pattern_sub_fn(m.group(0))
            preceding = text[:m.start()].rstrip()
            if preceding and preceding[-1] not in ",:.!?":
                return ", " + expansion
            return expansion
        return _sub

    text = ALNUM_CODE_RE.sub(_comma_before(spell_alnum_code), text)
    text = LONG_DIGIT_RE.sub(_comma_before(spell_digits), text)
    text = TIME_RE.sub(lambda m: spell_time(m.group(1), m.group(2)), text)
    text = SHORT_NUM_RE.sub(lambda m: spell_amount(m.group(0), lang="en"), text)
    return text


def normalize_text(text: str) -> str:
    text = apply_pronunciation_overrides(text)
    text = normalize_tagged_text(text)
    text = heuristic_normalize(text)
    return text
