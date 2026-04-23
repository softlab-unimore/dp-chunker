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
    "pcomp",  # complemento di preposizione
    "agent",  # agente passivo
    "oprd",  # predicato oggetto
    "amod",  # aggettivo predicativo (raro sul verbo)
    "advcl"
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
    for child in verb.children:
        if child.dep_ in SUBJECT_DEPS:
            if child.pos_ == "PRON" and child.text.lower() in ("which", "that", "who", "whom"):
                # risale al nominale referente: head del relcl (che è il verbo),
                # poi head di quello (che è il nome)
                if verb.dep_ == "relcl":
                    return verb.head  # il nome su cui è attaccato il relcl
            return child
    if verb.dep_ == "conj" and verb.head.dep_ == "ROOT":
        return get_subject(verb.head)
    return None


def build_proposition(subj_text: str, verb: spacy.tokens.Token,
                      complement_token: spacy.tokens.Token,
                      verb_prefix: str | None = None) -> str:
    aux_parts = []
    for child in verb.children:
        if child.dep_ in ("aux", "auxpass", "neg") and child.i < verb.i:
            aux_parts.append(child.text)

    verb_phrase = " ".join(aux_parts + [verb.text])
    if verb_prefix:
        verb_phrase = f"{verb_prefix} {verb_phrase}"

    complement_text = get_span_text(complement_token)
    proposition = f"{subj_text} {verb_phrase} {complement_text}".strip()
    if proposition:
        proposition = proposition[0].upper() + proposition[1:]
        if not proposition.endswith((".", "?", "!")):
            proposition += "."
    return proposition


# ---------------------------------------------------------------------------
# Funzione principale di splitting
# ---------------------------------------------------------------------------

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


def get_conj_chain(token: spacy.tokens.Token) -> list:
    """Raccoglie ricorsivamente tutti i conj di un token."""
    result = []
    for child in token.children:
        if child.dep_ == "conj":
            result.append(child)
            result.extend(get_conj_chain(child))
    return result


def expand_pobj_appos(subj_text: str, verb: spacy.tokens.Token,
                      prep_token: spacy.tokens.Token, propositions: list):
    for pobj in prep_token.children:
        if pobj.dep_ != "pobj":
            continue

        # Raccoglie appos e conj ricorsivamente
        expansions = []
        for child in pobj.children:
            if child.dep_ in ("appos", "conj") and child.pos_ in ("NOUN", "PROPN"):
                expansions.append(child)
                expansions.extend(get_conj_chain(child))

        # Proposizione base sempre emessa
        prop = build_proposition(subj_text, verb, prep_token)
        propositions.append(prop)

        # Proposizioni per ogni espansione
        for exp in expansions:
            complement_text = get_span_text(exp)
            aux_parts = [c.text for c in verb.children
                         if c.dep_ in ("aux", "auxpass") and c.i < verb.i]
            verb_phrase = " ".join(aux_parts + [verb.text])
            proposition = f"{subj_text} {verb_phrase} {prep_token.text} {complement_text}".strip()
            proposition = proposition[0].upper() + proposition[1:]
            if not proposition.endswith((".", "?", "!")):
                proposition += "."
            propositions.append(proposition)


def _process_sentence(verb: spacy.tokens.Token,
                       propositions: list,
                       inherited_subj: str | None,
                       inherited_verb_prefix: str | None = None) -> None:
    # 1. Individua soggetto
    subj_token = get_subject(verb)
    if subj_token is not None:
        subj_text = get_span_text(subj_token)

        # Salta verbi copulari tautologici
        attr_tokens = [c for c in verb.children if c.dep_ == "attr"]
        if attr_tokens:
            attr_text = get_span_text(attr_tokens[0])
            if attr_text == subj_text:
                for child in verb.children:
                    if child.dep_ == "ccomp":
                        _process_sentence(child, propositions, inherited_subj=None)
                    if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
                        _process_sentence(child, propositions, inherited_subj=subj_text)
                return

        # appos sul soggetto
        for appos in subj_token.children:
            if appos.dep_ == "appos":
                appos_text = get_span_text(appos)
                prop = f"{subj_text} is {appos_text}."
                prop = prop[0].upper() + prop[1:]
                propositions.append(prop)
                for appos_child in appos.children:
                    if appos_child.dep_ == "acl":
                        _process_sentence(appos_child, propositions,
                                         inherited_subj=subj_text)
                    if appos_child.dep_ == "conj" and appos_child.pos_ in ("NOUN", "PROPN"):
                        conj_text = get_span_text(appos_child)
                        prop = f"{subj_text} is {conj_text}."
                        prop = prop[0].upper() + prop[1:]
                        propositions.append(prop)
                        for conj_child in appos_child.children:
                            if conj_child.dep_ == "relcl":
                                _process_sentence(conj_child, propositions,
                                                 inherited_subj=get_span_text(appos_child))

    elif inherited_subj is not None:
        subj_text = inherited_subj
    else:
        subj_text = None

    # 2. Complementi → proposizioni atomiche
    if subj_text:
        for child in verb.children:
            if child.dep_ in COMPLEMENT_DEPS:
                if child.dep_ == "advmod" and child.text.lower() in (
                    "yet", "also", "too", "still", "just", "where",
                    "when", "why", "how", "then"
                ):
                    continue
                if child.dep_ == "prep":
                    has_verbal_pcomp = any(
                        c.dep_ == "pcomp" and c.pos_ == "VERB"
                        for c in child.children
                    )
                    if has_verbal_pcomp:
                        for c in child.children:
                            if c.dep_ == "pcomp" and c.pos_ == "VERB":
                                _process_sentence(c, propositions,
                                                 inherited_subj=subj_text)  # fix: era None
                    else:
                        expand_pobj_appos(subj_text, verb, child, propositions)
                elif child.dep_ == "dobj":
                    prop = build_proposition(subj_text, verb, child,
                                           verb_prefix=inherited_verb_prefix)
                    propositions.append(prop)
                    for dobj_child in child.children:
                        if dobj_child.dep_ == "prep":
                            expand_pobj_appos(subj_text, verb, dobj_child, propositions)
                        elif dobj_child.dep_ in ("appos", "conj") and dobj_child.pos_ in ("NOUN", "PROPN"):
                            appos_chain = [dobj_child] + get_conj_chain(dobj_child)
                            for appos in appos_chain:
                                appos_text = get_span_text(appos)
                                aux_parts = [c.text for c in verb.children
                                             if c.dep_ in ("aux", "auxpass") and c.i < verb.i]
                                verb_phrase = " ".join(aux_parts + [verb.text])
                                if inherited_verb_prefix:
                                    verb_phrase = f"{inherited_verb_prefix} {verb_phrase}"
                                prop = f"{subj_text} {verb_phrase} {appos_text}."
                                prop = prop[0].upper() + prop[1:]
                                propositions.append(prop)
                                for appos_child in appos.children:
                                    if appos_child.dep_ == "relcl":
                                        _process_sentence(appos_child, propositions,
                                                         inherited_subj=get_span_text(appos))
                        elif dobj_child.dep_ == "relcl":
                            _process_sentence(dobj_child, propositions,
                                             inherited_subj=child.text)
                elif child.dep_ == "attr":
                    prop = build_proposition(subj_text, verb, child,
                                           verb_prefix=inherited_verb_prefix)
                    propositions.append(prop)
                    for attr_child in child.children:
                        if attr_child.dep_ == "acl":
                            _process_sentence(attr_child, propositions,
                                             inherited_subj=get_span_text(child))
                        if attr_child.dep_ == "prep":
                            has_verbal_pcomp = any(
                                c.dep_ == "pcomp" and c.pos_ in ("VERB", "AUX")
                                for c in attr_child.children
                            )
                            if has_verbal_pcomp:
                                for c in attr_child.children:
                                    if c.dep_ == "pcomp" and c.pos_ in ("VERB", "AUX"):
                                        _process_sentence(c, propositions,
                                                         inherited_subj=get_span_text(child))
                            else:
                                expand_pobj_appos(get_span_text(child), verb,
                                                 attr_child, propositions)
                else:
                    prop = build_proposition(subj_text, verb, child,
                                           verb_prefix=inherited_verb_prefix)
                    propositions.append(prop)

        # appos diretto sul verbo
        for child in verb.children:
            if child.dep_ == "appos":
                appos_text = get_span_text(child)
                prop = f"{subj_text} is {appos_text}."
                prop = prop[0].upper() + prop[1:]
                propositions.append(prop)
                for appos_child in child.children:
                    if appos_child.dep_ == "acl":
                        _process_sentence(appos_child, propositions,
                                         inherited_subj=subj_text)
                    if appos_child.dep_ == "conj" and appos_child.pos_ in ("NOUN", "PROPN"):
                        conj_text = get_span_text(appos_child)
                        prop = f"{subj_text} is {conj_text}."
                        prop = prop[0].upper() + prop[1:]
                        propositions.append(prop)
                        for conj_child in appos_child.children:
                            if conj_child.dep_ == "relcl":
                                _process_sentence(conj_child, propositions,
                                                 inherited_subj=get_span_text(appos_child))

    # 3. ccomp, advcl, xcomp e pcomp verbali
    for child in verb.children:
        if child.dep_ in ("ccomp", "advcl"):
            _process_sentence(child, propositions, inherited_subj=None)
        elif child.dep_ == "xcomp" and child.pos_ == "VERB":
            parent_aux = [c.text for c in verb.children
                          if c.dep_ in ("aux", "auxpass", "neg") and c.i < verb.i]
            parent_verb_phrase = " ".join(parent_aux + [verb.text])
            _process_sentence(child, propositions,
                             inherited_subj=subj_text or inherited_subj,
                             inherited_verb_prefix=parent_verb_phrase)
        elif child.dep_ == "pcomp" and child.pos_ in ("VERB", "AUX"):
            _process_sentence(child, propositions,
                             inherited_subj=subj_text or inherited_subj)  # fix: era None

    # 4. Congiunzioni verbali
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
            _process_sentence(child, propositions,
                              inherited_subj=subj_text or inherited_subj)

    # 5. relcl / acl sul verbo
    for child in verb.children:
        if child.dep_ in ("relcl", "acl"):
            rel_subj = get_span_text(child.head) if child.head != verb else subj_text
            # Salta acl con soggetto ereditato troppo lungo (probabile rumore)
            if rel_subj and len(rel_subj.split()) > 6:
                continue
            _process_sentence(child, propositions, inherited_subj=rel_subj)