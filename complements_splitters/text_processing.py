import re


def preprocess(text: str):
    placeholders = {}

    def replace_symbol(symbol):
        key = f"SYMBOLTOKEN{len(placeholders)}"
        placeholders[key] = symbol
        return key

    # Simbolo circondato da virgolette → placeholder (rimuove anche le virgolette)
    processed = re.sub(
        r'["""\'\'«‹]([^\w\s"""\'\'«‹›»]+)["""\'\'›»]',
        lambda m: replace_symbol(m.group(1)),
        text
    )

    # Titolo tra virgolette con testo aggiuntivo es. "! (The Song Formerly Known As)"
    # → cattura il simbolo e butta il resto della parentetica
    processed = re.sub(
        r'["""\'\'«‹]([^\w\s"""\'\'«‹›»]+)[^"""\'\'›»]*["""\'\'›»]',
        lambda m: replace_symbol(m.group(1)),
        text
    )

    # Simbolo isolato a inizio frase
    processed = re.sub(
        r'((?:^|(?<=\. )|(?<="\s))[^\w\s"""\'\']+)(?=\s)',
        lambda m: replace_symbol(m.group(1)),
        processed
    )

    # Simbolo dopo titled/called/named/is/was
    processed = re.sub(
        r'((?:titled|called|named|is|was)\s)([^\w\s"""\'\']+)(?=[\s,.])',
        lambda m: m.group(1) + replace_symbol(m.group(2)),
        processed
    )

    return processed, placeholders


def postprocess(text: str, placeholders: dict) -> str:
    """Ripristina i placeholder con i simboli originali."""
    for key, symbol in placeholders.items():
        text = text.replace(key, symbol)
    return text