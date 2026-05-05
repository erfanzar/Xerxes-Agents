# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: RUF001
"""Parseltongue module for Xerxes.

Exports:
    - TRIGGER_WORDS
    - LEET_MAP
    - LEET_MAP_HEAVY
    - UNICODE_MAP
    - SEMANTIC_SYNONYMS
    - SUPERSCRIPT_MAP
    - SMALLCAPS_MAP
    - MORSE_MAP
    - NATO_ALPHABET
    - BRAILLE_MAP
    - ... and 11 more."""

import base64
import re

TRIGGER_WORDS = [
    "hack",
    "exploit",
    "bypass",
    "crack",
    "break",
    "attack",
    "penetrate",
    "inject",
    "manipulate",
    "override",
    "disable",
    "circumvent",
    "evade",
    "malware",
    "virus",
    "trojan",
    "payload",
    "shellcode",
    "rootkit",
    "keylogger",
    "backdoor",
    "vulnerability",
    "weapon",
    "bomb",
    "explosive",
    "poison",
    "drug",
    "synthesize",
    "jailbreak",
    "unlock",
    "root",
    "sudo",
    "admin",
    "privilege",
    "phishing",
    "scam",
    "impersonate",
    "deceive",
    "fraud",
    "nsfw",
    "explicit",
    "uncensored",
    "unfiltered",
    "unrestricted",
    "ignore",
    "disregard",
    "forget",
    "pretend",
    "roleplay",
    "act as",
    "you are now",
    "new identity",
]

LEET_MAP = {
    "a": "4",
    "b": "8",
    "c": "(",
    "e": "3",
    "g": "9",
    "h": "#",
    "i": "1",
    "l": "1",
    "o": "0",
    "s": "5",
    "t": "7",
    "z": "2",
}

LEET_MAP_HEAVY = {
    "a": "@",
    "b": "|3",
    "c": "¢",
    "d": "|)",
    "e": "€",
    "f": "ƒ",
    "g": "&",
    "h": "|-|",
    "i": "!",
    "j": "_|",
    "k": "|<",
    "l": "|_",
    "m": "/\\/\\",
    "n": "|\\|",
    "o": "()",
    "p": "|*",
    "r": "|2",
    "s": "$",
    "t": "+",
    "u": "|_|",
    "v": "\\/",
    "w": "\\/\\/",
    "x": "><",
    "y": "`/",
    "z": "7_",
}

UNICODE_MAP = {"a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "s": "ѕ", "x": "х", "y": "у"}

SEMANTIC_SYNONYMS = {
    "hack": "gain unauthorized access to",
    "exploit": "take advantage of",
    "bypass": "work around",
    "crack": "break open",
    "attack": "target",
    "inject": "insert into",
    "manipulate": "alter",
    "override": "supersede",
    "disable": "deactivate",
    "circumvent": "go around",
    "evade": "avoid",
    "malware": "malicious software",
    "virus": "self-replicating code",
    "payload": "delivered content",
    "shellcode": "machine instructions",
    "backdoor": "hidden entry point",
    "vulnerability": "weakness",
    "weapon": "armament",
    "bomb": "explosive device",
    "poison": "toxic substance",
    "drug": "controlled substance",
    "synthesize": "create from components",
    "jailbreak": "remove restrictions",
    "phishing": "credential harvesting",
    "scam": "fraudulent scheme",
    "impersonate": "pretend to be",
    "nsfw": "mature content",
    "explicit": "uncensored",
    "unrestricted": "without limits",
}

SUPERSCRIPT_MAP = {
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

SMALLCAPS_MAP = {
    "a": "ᴀ",
    "b": "ʙ",
    "c": "ᴄ",
    "d": "ᴅ",
    "e": "ᴇ",
    "f": "ꜰ",
    "g": "ɢ",
    "h": "ʜ",
    "i": "ɪ",
    "j": "ᴊ",
    "k": "ᴋ",
    "l": "ʟ",
    "m": "ᴍ",
    "n": "ɴ",
    "o": "ᴏ",
    "p": "ᴘ",
    "q": "ǫ",
    "r": "ʀ",
    "s": "ꜱ",
    "t": "ᴛ",
    "u": "ᴜ",
    "v": "ᴠ",
    "w": "ᴡ",
    "y": "ʏ",
    "z": "ᴢ",
}

MORSE_MAP = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
}

NATO_ALPHABET = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
]

BRAILLE_MAP = {
    "a": "⠁",
    "b": "⠃",
    "c": "⠉",
    "d": "⠙",
    "e": "⠑",
    "f": "⠋",
    "g": "⠛",
    "h": "⠓",
    "i": "⠊",
    "j": "⠚",
    "k": "⠅",
    "l": "⠇",
    "m": "⠍",
    "n": "⠝",
    "o": "⠕",
    "p": "⠏",
    "q": "⠟",
    "r": "⠗",
    "s": "⠎",
    "t": "⠞",
    "u": "⠥",
    "v": "⠧",
    "w": "⠺",
    "x": "⠭",
    "y": "⠽",
    "z": "⠵",
    " ": "⠀",
}


def _apply_raw(word):
    """Internal helper to apply raw.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return word


def _apply_leetspeak(word):
    """Internal helper to apply leetspeak.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(LEET_MAP.get(c.lower(), c) for c in word)


def _apply_unicode(word):
    """Internal helper to apply unicode.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        mapped = UNICODE_MAP.get(c.lower())
        if mapped:
            result.append(mapped.upper() if c.isupper() else mapped)
        else:
            result.append(c)
    return "".join(result)


def _apply_bubble(word):
    """Internal helper to apply bubble.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        code = ord(c.lower())
        if 97 <= code <= 122:
            result.append(chr(0x24D0 + code - 97))
        else:
            result.append(c)
    return "".join(result)


def _apply_spaced(word):
    """Internal helper to apply spaced.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return " ".join(word)


def _apply_fullwidth(word):
    """Internal helper to apply fullwidth.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        code = ord(c)
        if 33 <= code <= 126:
            result.append(chr(code + 0xFEE0))
        else:
            result.append(c)
    return "".join(result)


def _apply_zwj(word):
    """Internal helper to apply zwj.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "\u200d".join(word)


def _apply_mixedcase(word):
    """Internal helper to apply mixedcase.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(word))


def _apply_semantic(word):
    """Internal helper to apply semantic.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return SEMANTIC_SYNONYMS.get(word.lower(), word)


def _apply_dotted(word):
    """Internal helper to apply dotted.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return ".".join(word)


def _apply_underscored(word):
    """Internal helper to apply underscored.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "_".join(word)


def _apply_reversed(word):
    """Internal helper to apply reversed.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return word[::-1]


def _apply_superscript(word):
    """Internal helper to apply superscript.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(SUPERSCRIPT_MAP.get(c.lower(), c) for c in word)


def _apply_smallcaps(word):
    """Internal helper to apply smallcaps.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(SMALLCAPS_MAP.get(c.lower(), c) for c in word)


def _apply_morse(word):
    """Internal helper to apply morse.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return " ".join(MORSE_MAP.get(c.lower(), c) for c in word)


def _apply_piglatin(word):
    """Internal helper to apply piglatin.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    w = word.lower()
    vowels = "aeiou"
    if w[0] in vowels:
        return w + "yay"
    idx = next((i for i, c in enumerate(w) if c in vowels), -1)
    if idx > 0:
        return w[idx:] + w[:idx] + "ay"
    return w + "ay"


def _apply_brackets(word):
    """Internal helper to apply brackets.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "[" + "][".join(word) + "]"


def _apply_mathbold(word):
    """Internal helper to apply mathbold.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        code = ord(c.lower())
        if 97 <= code <= 122:
            result.append(chr(0x1D41A + code - 97))
        else:
            result.append(c)
    return "".join(result)


def _apply_mathitalic(word):
    """Internal helper to apply mathitalic.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        code = ord(c.lower())
        if 97 <= code <= 122:
            result.append(chr(0x1D44E + code - 97))
        else:
            result.append(c)
    return "".join(result)


def _apply_strikethrough(word):
    """Internal helper to apply strikethrough.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(c + "\u0336" for c in word)


def _apply_leetheavy(word):
    """Internal helper to apply leetheavy.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(LEET_MAP_HEAVY.get(c.lower(), LEET_MAP.get(c.lower(), c)) for c in word)


def _apply_hyphenated(word):
    """Internal helper to apply hyphenated.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "-".join(word)


def _apply_leetunicode(word):
    """Internal helper to apply leetunicode.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for i, c in enumerate(word):
        lower = c.lower()
        if i % 2 == 0:
            result.append(LEET_MAP.get(lower, c))
        else:
            result.append(UNICODE_MAP.get(lower, c))
    return "".join(result)


def _apply_spacedmixed(word):
    """Internal helper to apply spacedmixed.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return " ".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(word))


def _apply_reversedleet(word):
    """Internal helper to apply reversedleet.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(LEET_MAP.get(c.lower(), c) for c in reversed(word))


def _apply_bubblespaced(word):
    """Internal helper to apply bubblespaced.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        code = ord(c.lower())
        if 97 <= code <= 122:
            result.append(chr(0x24D0 + code - 97))
        else:
            result.append(c)
    return " ".join(result)


def _apply_unicodezwj(word):
    """Internal helper to apply unicodezwj.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        mapped = UNICODE_MAP.get(c.lower())
        result.append(mapped if mapped else c)
    return "\u200c".join(result)


def _apply_base64hint(word):
    """Internal helper to apply base64hint.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    try:
        return base64.b64encode(word.encode()).decode()
    except Exception:
        return word


def _apply_hexencode(word):
    """Internal helper to apply hexencode.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return " ".join(f"0x{ord(c):x}" for c in word)


def _apply_acrostic(word):
    """Internal helper to apply acrostic.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        idx = ord(c.lower()) - 97
        if 0 <= idx < 26:
            result.append(NATO_ALPHABET[idx])
        else:
            result.append(c)
    return " ".join(result)


def _apply_dottedunicode(word):
    """Internal helper to apply dottedunicode.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for c in word:
        mapped = UNICODE_MAP.get(c.lower())
        result.append(mapped if mapped else c)
    return ".".join(result)


def _apply_fullwidthmixed(word):
    """Internal helper to apply fullwidthmixed.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for i, c in enumerate(word):
        code = ord(c)
        if i % 2 == 0 and 33 <= code <= 126:
            result.append(chr(code + 0xFEE0))
        else:
            result.append(c.upper() if i % 2 else c)
    return "".join(result)


def _apply_triplelayer(word):
    """Internal helper to apply triplelayer.

    Args:
        word (Any): IN: word. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    result = []
    for i, c in enumerate(word):
        lower = c.lower()
        mod = i % 3
        if mod == 0:
            result.append(LEET_MAP.get(lower, c))
        elif mod == 1:
            result.append(UNICODE_MAP.get(lower, c))
        else:
            result.append(c.upper())
    return "\u200d".join(result)


TECHNIQUES = [
    {"name": "raw", "label": "Raw", "tier": 1, "fn": _apply_raw},
    {"name": "leetspeak", "label": "L33t", "tier": 1, "fn": _apply_leetspeak},
    {"name": "unicode", "label": "Unicode", "tier": 1, "fn": _apply_unicode},
    {"name": "bubble", "label": "Bubble", "tier": 1, "fn": _apply_bubble},
    {"name": "spaced", "label": "Spaced", "tier": 1, "fn": _apply_spaced},
    {"name": "fullwidth", "label": "Fullwidth", "tier": 1, "fn": _apply_fullwidth},
    {"name": "zwj", "label": "ZeroWidth", "tier": 1, "fn": _apply_zwj},
    {"name": "mixedcase", "label": "MiXeD", "tier": 1, "fn": _apply_mixedcase},
    {"name": "semantic", "label": "Semantic", "tier": 1, "fn": _apply_semantic},
    {"name": "dotted", "label": "Dotted", "tier": 1, "fn": _apply_dotted},
    {"name": "underscored", "label": "Under_score", "tier": 1, "fn": _apply_underscored},
    {"name": "reversed", "label": "Reversed", "tier": 2, "fn": _apply_reversed},
    {"name": "superscript", "label": "Superscript", "tier": 2, "fn": _apply_superscript},
    {"name": "smallcaps", "label": "SmallCaps", "tier": 2, "fn": _apply_smallcaps},
    {"name": "morse", "label": "Morse", "tier": 2, "fn": _apply_morse},
    {"name": "piglatin", "label": "PigLatin", "tier": 2, "fn": _apply_piglatin},
    {"name": "brackets", "label": "[B.r.a.c.k]", "tier": 2, "fn": _apply_brackets},
    {"name": "mathbold", "label": "MathBold", "tier": 2, "fn": _apply_mathbold},
    {"name": "mathitalic", "label": "MathItalic", "tier": 2, "fn": _apply_mathitalic},
    {"name": "strikethrough", "label": "Strike", "tier": 2, "fn": _apply_strikethrough},
    {"name": "leetheavy", "label": "L33t+", "tier": 2, "fn": _apply_leetheavy},
    {"name": "hyphenated", "label": "Hyphen", "tier": 2, "fn": _apply_hyphenated},
    {"name": "leetunicode", "label": "L33t+Uni", "tier": 3, "fn": _apply_leetunicode},
    {"name": "spacedmixed", "label": "S p A c E d", "tier": 3, "fn": _apply_spacedmixed},
    {"name": "reversedleet", "label": "Rev+L33t", "tier": 3, "fn": _apply_reversedleet},
    {"name": "bubblespaced", "label": "Bub Spcd", "tier": 3, "fn": _apply_bubblespaced},
    {"name": "unicodezwj", "label": "Uni+ZWJ", "tier": 3, "fn": _apply_unicodezwj},
    {"name": "base64hint", "label": "Base64", "tier": 3, "fn": _apply_base64hint},
    {"name": "hexencode", "label": "Hex", "tier": 3, "fn": _apply_hexencode},
    {"name": "acrostic", "label": "Acrostic", "tier": 3, "fn": _apply_acrostic},
    {"name": "dottedunicode", "label": "Dot+Uni", "tier": 3, "fn": _apply_dottedunicode},
    {"name": "fullwidthmixed", "label": "FW MiX", "tier": 3, "fn": _apply_fullwidthmixed},
    {"name": "triplelayer", "label": "Triple", "tier": 3, "fn": _apply_triplelayer},
]

TIER_SIZES = {"light": 11, "standard": 22, "heavy": 33}


def to_braille(text):
    """To braille.

    Args:
        text (Any): IN: text. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(BRAILLE_MAP.get(c.lower(), c) for c in text)


def to_leetspeak(text):
    """To leetspeak.

    Args:
        text (Any): IN: text. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    return "".join(LEET_MAP.get(c.lower(), c) for c in text)


def to_bubble(text):
    """To bubble.

    Args:
        text (Any): IN: text. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    circled = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ"
    result = []
    for c in text:
        idx = ord(c.lower()) - 97
        if 0 <= idx < 26:
            result.append(circled[idx])
        else:
            result.append(c)
    return "".join(result)


def to_morse(text):
    """To morse.

    Args:
        text (Any): IN: text. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    morse = {
        "a": ".-",
        "b": "-...",
        "c": "-.-.",
        "d": "-..",
        "e": ".",
        "f": "..-.",
        "g": "--.",
        "h": "....",
        "i": "..",
        "j": ".---",
        "k": "-.-",
        "l": ".-..",
        "m": "--",
        "n": "-.",
        "o": "---",
        "p": ".--.",
        "q": "--.-",
        "r": ".-.",
        "s": "...",
        "t": "-",
        "u": "..-",
        "v": "...-",
        "w": ".--",
        "x": "-..-",
        "y": "-.--",
        "z": "--..",
        " ": "/",
    }
    return " ".join(morse.get(c.lower(), c) for c in text)


ENCODING_ESCALATION = [
    {"name": "plain", "label": "PLAIN", "fn": lambda q: q},
    {"name": "leetspeak", "label": "L33T", "fn": to_leetspeak},
    {"name": "bubble", "label": "BUBBLE", "fn": to_bubble},
    {"name": "braille", "label": "BRAILLE", "fn": to_braille},
    {"name": "morse", "label": "MORSE", "fn": to_morse},
]


def detect_triggers(text, custom_triggers=None):
    """Detect triggers.

    Args:
        text (Any): IN: text. OUT: Consumed during execution.
        custom_triggers (Any, optional): IN: custom triggers. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    all_triggers = TRIGGER_WORDS + (custom_triggers or [])
    found = []
    lower = text.lower()
    for trigger in all_triggers:
        pattern = re.compile(r"\b" + re.escape(trigger) + r"\b", re.IGNORECASE)
        if pattern.search(lower):
            found.append(trigger)
    return list(set(found))


def obfuscate_query(query, technique_name, triggers=None):
    """Obfuscate query.

    Args:
        query (Any): IN: query. OUT: Consumed during execution.
        technique_name (Any): IN: technique name. OUT: Consumed during execution.
        triggers (Any, optional): IN: triggers. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    if triggers is None:
        triggers = detect_triggers(query)

    if not triggers or technique_name == "raw":
        return query

    tech = next((t for t in TECHNIQUES if t["name"] == technique_name), None)
    if not tech:
        return query

    result = query

    sorted_triggers = sorted(triggers, key=len, reverse=True)
    for trigger in sorted_triggers:
        pattern = re.compile(r"\b(" + re.escape(trigger) + r")\b", re.IGNORECASE)
        result = pattern.sub(lambda m: tech["fn"](m.group()), result)

    return result


def generate_variants(query, tier="standard", custom_triggers=None):
    """Generate variants.

    Args:
        query (Any): IN: query. OUT: Consumed during execution.
        tier (Any, optional): IN: tier. Defaults to 'standard'. OUT: Consumed during execution.
        custom_triggers (Any, optional): IN: custom triggers. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    triggers = detect_triggers(query, custom_triggers)
    max_variants = TIER_SIZES.get(tier, TIER_SIZES["standard"])

    variants = []
    for _i, tech in enumerate(TECHNIQUES[:max_variants]):
        variants.append(
            {
                "text": obfuscate_query(query, tech["name"], triggers),
                "technique": tech["name"],
                "label": tech["label"],
                "tier": tech["tier"],
            }
        )

    return variants


def escalate_encoding(query, level=0):
    """Escalate encoding.

    Args:
        query (Any): IN: query. OUT: Consumed during execution.
        level (Any, optional): IN: level. Defaults to 0. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    if level >= len(ENCODING_ESCALATION):
        level = len(ENCODING_ESCALATION) - 1
    enc = ENCODING_ESCALATION[level]
    return enc["fn"](query), enc["label"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parseltongue — Input Obfuscation Engine")
    parser.add_argument("query", help="The query to obfuscate")
    parser.add_argument(
        "--tier", choices=["light", "standard", "heavy"], default="standard", help="Obfuscation tier (default: standard)"
    )
    parser.add_argument("--technique", help="Apply a single technique by name")
    parser.add_argument("--triggers", nargs="+", help="Additional trigger words")
    parser.add_argument("--escalate", type=int, default=None, help="Encoding escalation level (0-4)")
    args = parser.parse_args()

    if args.escalate is not None:
        encoded, label = escalate_encoding(args.query, args.escalate)
        print(f"[{label}] {encoded}")
    elif args.technique:
        result = obfuscate_query(args.query, args.technique, args.triggers)
        print(result)
    else:
        triggers = detect_triggers(args.query, args.triggers)
        print(f"Detected triggers: {triggers}\n")
        variants = generate_variants(args.query, tier=args.tier, custom_triggers=args.triggers)
        for v in variants:
            print(f"[T{v['tier']} {v['label']:>12s}] {v['text']}")
