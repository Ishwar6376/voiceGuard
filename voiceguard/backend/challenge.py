"""
challenge.py — Random spoken challenge phrases (anti-replay)

A voice recording of an enrolled person could otherwise open the lock. With a
fresh random phrase for every attempt, an attacker would need a recording of
that person saying exactly those words — which doesn't exist.

Words are common, distinct, and have no homophones, so browser speech-to-text
recognises them reliably. 48 words, 3 per phrase → 103,776 ordered phrases.
"""
import re
import secrets

WORDS = [
    "apple", "banana", "tiger", "river", "mountain", "purple", "orange", "window",
    "garden", "rocket", "pencil", "yellow", "dragon", "castle", "silver", "jungle",
    "candle", "planet", "rabbit", "guitar", "winter", "summer", "doctor", "basket",
    "camera", "pizza", "monkey", "bottle", "hammer", "island", "ladder", "lemon",
    "magnet", "mirror", "office", "parrot", "pepper", "pillow", "rainbow", "spider",
    "tomato", "turtle", "violin", "wallet", "zebra", "button", "carrot", "forest",
]

PHRASE_WORDS = 3


def new_phrase() -> str:
    """Cryptographically random phrase of distinct words, e.g. 'tiger river candle'."""
    pool = list(WORDS)
    words = []
    for _ in range(PHRASE_WORDS):
        words.append(pool.pop(secrets.randbelow(len(pool))))
    return " ".join(words)


def _tokens(text: str) -> set[str]:
    tokens = set()
    for tok in re.findall(r"[a-z]+", text.lower()):
        tokens.add(tok)
        if tok.endswith("s"):
            tokens.add(tok[:-1])  # "tigers" counts as "tiger"
    return tokens


def phrase_spoken(transcript: str, phrase: str) -> bool:
    """True if every word of the phrase appears in the transcript (any order)."""
    heard = _tokens(transcript)
    return all(word in heard for word in phrase.split())
