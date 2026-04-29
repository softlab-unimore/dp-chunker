import spacy

from complements_splitters.text_processing import preprocess, postprocess

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

# Verbi intransitivi / particelle verbali che non richiedono complemento
# per essere proposizioni valide (soggetto + predicato).
# Usato nel controllo "nessun complemento trovato".
INTRANSITIVE_OK = True  # abilita sempre l'emissione di proposizioni senza complemento


# ---------------------------------------------------------------------------
# Funzioni di utilità
# ---------------------------------------------------------------------------

def get_span_text(token: spacy.tokens.Token) -> str:
    """
    Ricostruisce il testo del sottoalbero di un token mantenendo l'ordine
    lineare originale, ma escludendo le congiunzioni (cc/conj) che
    verrebbero trattate separatamente.
    """

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
    collected.sort(key=lambda t: t.i)
    return " ".join(t.text for t in collected).strip()

def get_span_text_no_relcl(token: spacy.tokens.Token) -> str:
    """
    Come get_span_text, ma esclude anche i sottoalberi relcl/acl.
    Usato per costruire il soggetto da ereditare a una relativa clausola:
    "a former I.C.O.N. agent" (senza "who met Hero…").
    """
    def collect(tok):
        tokens = []
        for child in tok.lefts:
            if child.dep_ not in ("conj", "cc", "punct", "relcl", "acl"):
                tokens.extend(collect(child))
        tokens.append(tok)
        for child in tok.rights:
            if child.dep_ not in ("conj", "cc", "punct", "relcl", "acl"):
                tokens.extend(collect(child))
        return tokens

    collected = collect(token)
    collected.sort(key=lambda t: t.i)
    return " ".join(t.text for t in collected).strip()



def get_subject(verb: spacy.tokens.Token):
    for child in verb.children:
        if child.dep_ in SUBJECT_DEPS:
            if child.pos_ == "PRON" and child.text.lower() in ("which", "that", "who", "whom"):
                if verb.dep_ == "relcl":
                    return verb.head
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


def build_intransitive_proposition(subj_text: str, verb: spacy.tokens.Token,
                                   verb_prefix: str | None = None) -> str:
    """
    FIX 5 — Costruisce una proposizione soggetto+predicato senza complemento.
    Utile per verbi intransitivi o verbi con particella (grow up, set out…).
    Includono ausiliari, negazioni e particelle (prt) se presenti.
    """
    aux_parts = []
    prt_parts = []
    for child in verb.children:
        if child.dep_ in ("aux", "auxpass", "neg") and child.i < verb.i:
            aux_parts.append(child.text)
        elif child.dep_ == "prt":
            prt_parts.append(child.text)

    verb_phrase = " ".join(aux_parts + [verb.text] + prt_parts)
    if verb_prefix:
        verb_phrase = f"{verb_prefix} {verb_phrase}"

    proposition = f"{subj_text} {verb_phrase}".strip()
    if proposition:
        proposition = proposition[0].upper() + proposition[1:]
        if not proposition.endswith((".", "?", "!")):
            proposition += "."
    return proposition


# ---------------------------------------------------------------------------
# Funzione principale di splitting
# ---------------------------------------------------------------------------


def split_atomic(text: str, nlp) -> list[str]:
    processed, placeholders = preprocess(text)
    doc = nlp(processed)
    propositions = []

    for sent in doc.sents:
        _process_sentence(sent.root, propositions, inherited_subj=None)

    seen = set()
    result = []
    for p in propositions:
        p = postprocess(p, placeholders)
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

        expansions = []
        for child in pobj.children:
            if child.dep_ in ("appos", "conj") and child.pos_ in ("NOUN", "PROPN"):
                expansions.append(child)
                expansions.extend(get_conj_chain(child))

        prop = build_proposition(subj_text, verb, prep_token)
        propositions.append(prop)

        # Visita appos/relcl sul pobj per catturare relative clausole come
        # "narrated by Hunter, a former agent who met Hero" — il "who met Hero"
        # è relcl su un appos del pobj e non viene altrimenti raggiunto.
        _expand_pobj_relcl(pobj, propositions, _process_sentence)

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



def _expand_pobj_relcl(pobj: "spacy.tokens.Token", propositions: list,
                        process_fn) -> None:
    """
    Visita ricorsivamente i figli `appos` di un pobj cercando relcl da processare.

    Problema che risolve: in frasi come
      "narrated by Agent Hunter, a former agent who met Hero"
    `agent` è `appos` di `Hunter` (pobj), e `who met Hero` è `relcl` di `agent`.
    Il codice principale non raggiunge mai questo livello perché visita solo
    appos diretti del soggetto o del verbo — non appos di pobj.

    Soluzione: dopo aver emesso la prop per la prep, scendiamo nel pobj e
    per ogni `appos` (e per il pobj stesso) processiamo eventuali `relcl`
    e `acl`, usando il testo del nodo come soggetto ereditato.
    """
    # Il pobj stesso può avere relcl/acl
    for child in pobj.children:
        if child.dep_ in ("relcl", "acl"):
            process_fn(child, propositions, inherited_subj=get_span_text(pobj))

    # Appos del pobj (es. "agent" appos di "Hunter")
    for child in pobj.children:
        if child.dep_ == "appos":
            # Il soggetto da ereditare è il testo dell'appos SENZA la relcl,
            # altrimenti "who met Hero…" finisce nel soggetto delle proposizioni.
            appos_subj = get_span_text_no_relcl(child)

            # Emetti "pobj is appos"
            pobj_text = get_span_text(pobj)
            prop = f"{pobj_text} is {appos_subj}."
            prop = prop[0].upper() + prop[1:]
            propositions.append(prop)
            # Processa relcl/acl sull'appos
            for appos_child in child.children:
                if appos_child.dep_ in ("relcl", "acl"):
                    process_fn(appos_child, propositions,
                               inherited_subj=appos_subj)
            # Ricorsione: l'appos stesso può avere ulteriori appos
            _expand_pobj_relcl(child, propositions, process_fn)


def _get_verb_phrase(verb: spacy.tokens.Token, verb_prefix: str | None = None) -> str:
    """Restituisce la frase verbale completa (ausiliari + verbo)."""
    aux_parts = [c.text for c in verb.children
                 if c.dep_ in ("aux", "auxpass", "neg") and c.i < verb.i]
    vp = " ".join(aux_parts + [verb.text])
    if verb_prefix:
        vp = f"{verb_prefix} {vp}"
    return vp


def _collect_aux(verb: spacy.tokens.Token) -> list[str]:
    """
    Raccoglie gli ausiliari/negazioni che precedono il verbo.
    Se il verbo è un `conj` e non ha ausiliari propri, risale al head per
    ereditarli (es. "forced" conj di "born" non ha "is" come figlio diretto,
    ma "born" sì).
    """
    own = [c.text for c in verb.children
           if c.dep_ in ("aux", "auxpass", "neg") and c.i < verb.i]
    if own:
        return own
    if verb.dep_ == "conj":
        return [c.text for c in verb.head.children
                if c.dep_ in ("aux", "auxpass", "neg") and c.i < verb.head.i]
    return []


def _build_xcomp_prefix(parent: spacy.tokens.Token,
                         xcomp_child: spacy.tokens.Token,
                         inherited_prefix: str | None = None) -> str:
    """
    Costruisce il prefisso verbale che precede i complementi del verbo xcomp.

    Esempi:
      "is forced to flee …"  →  parent=forced (conj di born, eredita "is")
        prefix = "is forced to"
      "begins to preach …"   →  parent=begins, xcomp=preach
        prefix = "begins to"

    Se inherited_prefix è già valorizzato (catena multipla), viene preposto.
    """
    aux_parts = _collect_aux(parent)
    parent_vp = " ".join(aux_parts + [parent.text])

    if inherited_prefix:
        parent_vp = f"{inherited_prefix} {parent_vp}"

    # Mark "to" sull'xcomp (infinito)
    mark = next(
        (c.text for c in xcomp_child.children
         if c.dep_ == "mark" and c.text.lower() == "to"),
        None
    )
    if mark:
        parent_vp = f"{parent_vp} {mark}"

    return parent_vp


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
    #
    # FIX 3 — raccoglie i token "prep" diretti figli del verbo (non di dobj/attr)
    # per evitare di emettere prep che dipendono dal complemento oggetto.
    direct_prep_children = {
        child for child in verb.children
        if child.dep_ == "prep"
    }

    if subj_text:
        # Tiene traccia se almeno un complemento è stato emesso (per FIX 5)
        emitted_complement = False

        # Pre-calcola se ci sono complementi non-advmod: se sì, gli advmod
        # non vanno emessi come proposizioni autonome (es. "ending eventually"
        # quando c'è anche "ending with his resurrection").
        has_non_advmod_complement = any(
            c.dep_ in COMPLEMENT_DEPS and c.dep_ != "advmod"
            for c in verb.children
        )

        for child in verb.children:
            if child.dep_ in COMPLEMENT_DEPS:
                if child.dep_ == "advmod" and child.text.lower() in (
                    "yet", "also", "too", "still", "just", "where",
                    "when", "why", "how", "then"
                ):
                    continue
                # Sopprimi advmod come proposizione autonoma se ci sono altri
                # complementi più informativi sullo stesso verbo.
                if child.dep_ == "advmod" and has_non_advmod_complement:
                    continue

                if child.dep_ == "prep":
                    # FIX 3 — emetti solo se la prep è figlia DIRETTA del verbo
                    if child not in direct_prep_children:
                        continue
                    has_verbal_pcomp = any(
                        c.dep_ == "pcomp" and c.pos_ == "VERB"
                        for c in child.children
                    )
                    if has_verbal_pcomp:
                        for c in child.children:
                            if c.dep_ == "pcomp" and c.pos_ == "VERB":
                                _process_sentence(c, propositions,
                                                 inherited_subj=subj_text)
                    else:
                        expand_pobj_appos(subj_text, verb, child, propositions)
                    emitted_complement = True

                elif child.dep_ == "dobj":
                    prop = build_proposition(subj_text, verb, child,
                                           verb_prefix=inherited_verb_prefix)
                    propositions.append(prop)
                    emitted_complement = True

                    for dobj_child in child.children:
                        if dobj_child.dep_ == "prep":
                            # FIX 2 — espandi le prep figlie del dobj con soggetto
                            # e verbo originale, per non perdere es. "to New York City".
                            # Eccezione: "of" introduce quasi sempre un modificatore
                            # nominale del dobj ("principles of Christianity") e non
                            # un complemento verbale autonomo — va lasciata nel sintagma.
                            if dobj_child.text.lower() == "of":
                                continue
                            expand_pobj_appos(subj_text, verb, dobj_child, propositions)
                        elif dobj_child.dep_ in ("appos", "conj") and dobj_child.pos_ in ("NOUN", "PROPN"):
                            # FIX 1 — non emettere prop separata per appos/conj
                            # del dobj: get_span_text(dobj) li include già nel
                            # testo base (es. "Petrov Peter" include già "Peter").
                            # Processiamo solo eventuali relcl sull'appos.
                            appos_chain = [dobj_child] + get_conj_chain(dobj_child)
                            for appos in appos_chain:
                                for appos_child in appos.children:
                                    if appos_child.dep_ == "relcl":
                                        _process_sentence(appos_child, propositions,
                                                         inherited_subj=get_span_text_no_relcl(appos))
                        elif dobj_child.dep_ == "relcl":
                            _process_sentence(dobj_child, propositions,
                                             inherited_subj=child.text)

                elif child.dep_ == "attr":
                    # FIX B — evita proposizioni tautologiche del tipo
                    # "a child named Jesus is a child named Jesus" quando
                    # get_span_text del soggetto include già il sottoalbero
                    # dell'attributo (acl/appos sul soggetto già espanso).
                    attr_text = get_span_text(child)
                    if attr_text in subj_text or subj_text in attr_text:
                        pass  # tautologia: salta la proposizione principale
                    else:
                        prop = build_proposition(subj_text, verb, child,
                                               verb_prefix=inherited_verb_prefix)
                        propositions.append(prop)
                    emitted_complement = True
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
                    emitted_complement = True
                    # Se il complemento è un "agent" (by + pobj), scendi nel
                    # pobj per catturare appos con relcl.
                    # Es: "narrated by Hunter, a former agent who met Hero"
                    # → Hunter (pobj) → agent (appos) → met (relcl)
                    if child.dep_ == "agent":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                _expand_pobj_relcl(pobj, propositions, _process_sentence)

        # appos diretto sul verbo
        for child in verb.children:
            if child.dep_ == "appos":
                appos_text = get_span_text(child)
                prop = f"{subj_text} is {appos_text}."
                prop = prop[0].upper() + prop[1:]
                propositions.append(prop)
                emitted_complement = True
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

        # FIX 5 — verbi intransitivi: emetti (soggetto, predicato) se nessun
        # complemento è stato trovato E il verbo non è puramente ausiliare.
        # FIX A — non emettere la forma intransitiva se il verbo ha un xcomp
        # verbale con soggetto disponibile: il predicato completo viene costruito
        # dalla gestione xcomp nel blocco 3 (es. "is forced to flee …").
        has_verbal_xcomp = any(
            c.dep_ == "xcomp" and c.pos_ == "VERB"
            for c in verb.children
        )
        suppress_intransitive = has_verbal_xcomp and bool(subj_text or inherited_subj)
        if not emitted_complement and not suppress_intransitive and verb.pos_ in ("VERB", "AUX"):
            # Evita di emettere per verbi copulari/ausiliari puri senza predicato
            # (es. "is", "are" da soli) oppure per xcomp/ccomp già gestiti sopra.
            if verb.dep_ not in ("aux", "auxpass") and verb.lemma_ not in ("be", "have"):
                prop = build_intransitive_proposition(subj_text, verb,
                                                      verb_prefix=inherited_verb_prefix)
                propositions.append(prop)

    # 3. ccomp, advcl, xcomp e pcomp verbali
    for child in verb.children:
        if child.dep_ == "ccomp":
            _process_sentence(child, propositions, inherited_subj=None)
        elif child.dep_ == "advcl":
            # advcl condivide il soggetto della frase principale (es. "using
            # references" in "The storyline progresses, using references").
            _process_sentence(child, propositions,
                              inherited_subj=subj_text or inherited_subj)
        elif child.dep_ == "xcomp" and child.pos_ == "VERB":
            # FIX 4 — appiattisce la catena verb → xcomp in un unico prefisso
            # verbale usando _build_xcomp_prefix, poi delega i complementi
            # al verbo xcomp.  Funziona anche in catene multi-livello
            # (es. "is forced to flee") perché _process_sentence viene chiamata
            # ricorsivamente con il prefisso già accumulato.
            prefix = _build_xcomp_prefix(verb, child, inherited_verb_prefix)
            _process_sentence(child, propositions,
                             inherited_subj=subj_text or inherited_subj,
                             inherited_verb_prefix=prefix)
        elif child.dep_ == "pcomp" and child.pos_ in ("VERB", "AUX"):
            _process_sentence(child, propositions,
                             inherited_subj=subj_text or inherited_subj)

    # 4. Congiunzioni verbali e verbi con dep="dep" (coordinati mal-etichettati)
    # spaCy a volte assegna dep_="dep" invece di "conj" a verbi coordinati in
    # frasi lunghe (es. "progresses … continues dep progresses"). Li trattiamo
    # esattamente come conj verbali, ereditando il soggetto.
    for child in verb.children:
        if child.dep_ in ("conj", "dep") and child.pos_ in ("VERB", "AUX"):
            _process_sentence(child, propositions,
                              inherited_subj=subj_text or inherited_subj)

    # 5. relcl / acl sul verbo
    for child in verb.children:
        if child.dep_ in ("relcl", "acl"):
            rel_subj = get_span_text(child.head) if child.head != verb else subj_text
            if rel_subj and len(rel_subj.split()) > 6:
                continue
            _process_sentence(child, propositions, inherited_subj=rel_subj)