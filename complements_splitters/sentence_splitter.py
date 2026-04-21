import re

import spacy

# Dipendenze considerate "complemento autonomo" (ognuna genera una prop.)
COMPLEMENT_DEPS = {
    "dobj",  # oggetto diretto
    "iobj",  # oggetto indiretto
    "attr",  # attributo nominale (es. "X is a Y")
    "prep",  # sintagma preposizionale
    "advmod",  # avverbio modificatore
    "acomp",  # complemento aggettivale
    "xcomp",  # complemento a distanza (open)
    "pcomp",  # complemento di preposizione
    "agent",  # agente passivo
    "oprd",  # predicato oggetto
    "npadvmod",  # NP usato come avverbio
    "amod",  # aggettivo predicativo (raro sul verbo)
}

# Dipendenze di soggetto
SUBJECT_DEPS = {"nsubj", "nsubjpass", "expl", "csubj", "csubjpass"}

# Dipendenze da espandere nella ricostruzione del sottoalbero
EXPAND_DEPS = {
    "det", "amod", "compound", "poss", "nummod",
    "advmod", "neg", "prep", "pobj", "case",
    "aux", "auxpass", "mark", "nmod", "appos",
    "relcl", "acl",
}


# ---------------------------------------------------------------------------
# Funzioni di utilità
# ---------------------------------------------------------------------------

def get_span_text(token: spacy.tokens.Token) -> str:
    """
    Ricostruisce il testo del sottoalbero di un token mantenendo l'ordine
    lineare originale, ma escludendo le congiunzioni (cc/conj) che
    verrebbero trattate separatamente.
    """

    # raccoglie tutti i token del sottoalbero TRANNE figli 'conj' ricorsivi
    def collect(tok):
        tokens = []
        for child in tok.lefts:
            if child.dep_ not in ("conj", "cc", "punct"):
                tokens.extend(collect(child))
        tokens.append(tok)
        for child in tok.rights:
            if child.dep_ not in ("conj", "cc", "punct"):
                tokens.extend(collect(child))
        return tokens

    collected = collect(token)
    # ordina per posizione nella frase originale
    collected.sort(key=lambda t: t.i)
    return " ".join(t.text for t in collected).strip()


def get_subject(verb: spacy.tokens.Token):
    """
    Restituisce il token soggetto (primo livello) di un verbo, oppure None.
    Cerca anche nei verbi 'conj' dello stesso soggetto.
    """
    for child in verb.children:
        if child.dep_ in SUBJECT_DEPS:
            return child
    # Nessun soggetto esplicito: risali al verbo head per ereditarlo
    if verb.dep_ == "conj" and verb.head.dep_ == "ROOT":
        return get_subject(verb.head)
    return None


def build_proposition(subj_text: str, verb: spacy.tokens.Token,
                      complement_token: spacy.tokens.Token) -> str:
    """
    Costruisce una stringa "soggetto verbo complemento" ricostruendo
    gli ausiliari del verbo e il sottoalbero del complemento.
    """
    # Raccoglie ausiliari e negazioni legate al verbo
    aux_parts = []
    for child in verb.children:
        if child.dep_ in ("aux", "auxpass", "neg") and child.i < verb.i:
            aux_parts.append(child.text)

    verb_phrase = " ".join(aux_parts + [verb.text])
    complement_text = get_span_text(complement_token)

    proposition = f"{subj_text} {verb_phrase} {complement_text}".strip()
    # Capitalizza e aggiungi punto
    if proposition:
        proposition = proposition[0].upper() + proposition[1:]
        if not proposition.endswith((".", "?", "!")):
            proposition += "."
    return proposition


# ---------------------------------------------------------------------------
# Funzione principale di splitting
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """
    Sostituisce simboli isolati usati come titolo con un placeholder
    che spaCy può parsare correttamente come soggetto.
    Mappa salvata per ripristinare il testo originale nelle proposizioni.
    """
    placeholders = {}

    def replace(m):
        symbol = m.group(1)
        key = f"SYMBOLTOKEN{len(placeholders)}"
        placeholders[key] = symbol
        return key

    # Simbolo isolato a inizio frase
    processed = re.sub(r'((?:^|(?<=\. ))[^\w\s]+)(?=\s)', replace, text)
    return processed, placeholders


def postprocess(text: str, placeholders: dict) -> str:
    """Ripristina i placeholder con i simboli originali."""
    for key, symbol in placeholders.items():
        text = text.replace(key, symbol)
    return text

def split_atomic(text: str, nlp) -> list[str]:
    """
    Riceve un testo (paragrafo) e restituisce una lista di proposizioni
    atomiche.
    """
    processed, placeholders = preprocess(text)
    doc = nlp(processed)
    propositions = []

    for sent in doc.sents:
        _process_sentence(sent.root, propositions, inherited_subj=None)

    seen = set()
    result = []
    for p in propositions:
        p = postprocess(p, placeholders)  # ripristina "!" al posto di SYMBOLTOKEN0
        if p not in seen:
            seen.add(p)
            result.append(p)

    return result


def _process_sentence(verb: spacy.tokens.Token,
                      propositions: list,
                      inherited_subj: str | None) -> None:
    """
    Elabora ricorsivamente un verbo (o una catena conj) e produce
    proposizioni atomiche.
    """
    # 1. Individua soggetto
    subj_token = get_subject(verb)
    if subj_token is not None:
        subj_text = get_span_text(subj_token)
    elif inherited_subj is not None:
        subj_text = inherited_subj
    else:
        subj_text = None

    # 2. Per ogni figlio che è un complemento → proposizione atomica
    if subj_text:
        for child in verb.children:
            if child.dep_ in COMPLEMENT_DEPS:
                prop = build_proposition(subj_text, verb, child)
                propositions.append(prop)

    # 3. Clausole complemento (ccomp): scendi ricorsivamente senza emettere
    #    una proposizione per il verbo reggente (said, claimed, believed...)
    for child in verb.children:
        if child.dep_ == "ccomp":
            _process_sentence(child, propositions, inherited_subj=None)

    # 4. Gestisce le congiunzioni verbali (stesso soggetto, verbo diverso)
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
            _process_sentence(child, propositions,
                              inherited_subj=subj_text or inherited_subj)

    # 5. Gestisce le clausole relative / acl attaccate a un NP del verbo
    for child in verb.children:
        if child.dep_ in ("relcl", "acl"):
            rel_subj = get_span_text(child.head) if child.head != verb else subj_text
            _process_sentence(child, propositions, inherited_subj=rel_subj)
