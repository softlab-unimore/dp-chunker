import re


def preprocess(text: str):
    placeholders = {}

    def replace_symbol(symbol):
        key = f"SYMBOLTOKEN{len(placeholders)}"
        placeholders[key] = symbol
        return key

    processed = text

    # Solo simboli NON-word tra virgolette → placeholder (es. "!", "♪", "#1")
    # NON tocca nomi propri o testo normale come "Agent Hunter"
    processed = re.sub(
        r'["""\'\'«‹]([^\w\s"""\'\'«‹›»]+)["""\'\'›»]',
        lambda m: replace_symbol(m.group(1)),
        processed
    )

    # Virgolette attorno a testo normale (parole) → rimuovi solo le virgolette,
    # lascia il testo intatto. Es: "Agent Hunter" → Agent Hunter
    processed = re.sub(
        r'["""\'\'«‹](\w[^"""\'\'«‹›»]*\w)["""\'\'›»]',
        lambda m: m.group(1).strip(),
        processed
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