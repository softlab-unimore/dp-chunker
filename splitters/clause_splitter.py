import spacy

from splitters.advcl_splitter import AdvclSplitter
from splitters.acl_splitter import AclSplitter
from splitters.relcl_splitter import RelclSplitter
from splitters.conj_splitter import ConjSplitter
from splitters.ccomp_splitter import CcompSplitter


class ClauseSplitter:

    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)
        self.advcl_splitter = AdvclSplitter(self.nlp)
        self.acl_splitter   = AclSplitter(self.nlp)
        self.relcl_splitter = RelclSplitter(self.nlp)
        self.conj_splitter  = ConjSplitter(self.nlp)
        self.ccomp_splitter = CcompSplitter(self.nlp)

        self.splitters = {
            "advcl": lambda doc, token: self.advcl_splitter.split(doc, token),
            "acl":   lambda doc, token: self.acl_splitter.split(doc, token),
            "relcl": lambda doc, token: self.relcl_splitter.split(doc, token),
            "conj":  lambda doc, token: self.conj_splitter.split(doc, token),
            "ccomp": lambda doc, token: self.ccomp_splitter.split(doc, token),
            "pobj":  lambda doc, token: (
                self.acl_splitter.split(
                    doc, token,
                    noun=next(
                        (ch for ch in token.head.head.children
                         if ch.dep_ in {"dobj", "obj"} and ch.i < token.head.i),
                        token.head.head
                    )
                )
                if token.pos_ == "VERB"
                and token.head.dep_ == "prep"
                and token.head.text.lower() == "to"
                else None
            ),
        }

    # ------------------------------------------------------------------
    # EXPAND_NOMINAL_CONJ
    # ------------------------------------------------------------------
    def expand_nominal_conj(self, doc):
        results = []
        visited = set()

        for token in doc:
            if (token.dep_ not in {"conj", "punct", "cc"} and
                token.pos_ in {"NOUN", "PROPN"} and
                any(ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"}
                    for ch in token.children)):

                if token.i in visited:
                    continue
                visited.add(token.i)

                nouns = [token] + [
                    ch for ch in token.children
                    if ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"}
                ]
                for n in nouns:
                    visited.add(n.i)

                shared_mods = []
                for ch in token.children:
                    if ch.dep_ in {"amod", "det"} and ch.pos_ in {"ADJ", "DET"}:
                        shared_mods.append(ch)
                        for mod_conj in ch.children:
                            if mod_conj.dep_ == "conj":
                                shared_mods.append(mod_conj)

                relcl_tokens = [
                    ch for ch in token.children
                    if ch.dep_ == "relcl"
                ]

                results.append({
                    "nouns": nouns,
                    "mods": shared_mods,
                    "relcl": relcl_tokens,
                    "head_token": token,
                })

        return results

    # ------------------------------------------------------------------
    # PROCESS NESTED SUBORDINATES
    # ------------------------------------------------------------------
    def process_nested(self, doc, token, splits, used_tokens):
        for t in token.subtree:
            if t.i == token.i:
                continue
            if t.i in used_tokens:
                continue

            actual_dep = t.dep_

            # Reindirizza relcl infinitive a acl
            if t.dep_ == "relcl":
                has_to = any(
                    ch.dep_ == "aux" and ch.text.lower() == "to"
                    for ch in t.children
                )
                if has_to:
                    actual_dep = "acl"

            # Reindirizza conj di ccomp a ccomp
            if t.dep_ == "conj":
                root = t
                while root.dep_ == "conj":
                    root = root.head
                if root.dep_ in {"ccomp", "xcomp"}:
                    actual_dep = "ccomp"

            if actual_dep in self.splitters:
                nested_result = self.splitters[actual_dep](doc, t)
                if nested_result:
                    splits.append({
                        "type": nested_result["type"],
                        "subordinate": nested_result["subordinate"]
                    })
                    used_tokens.update(nt.i for nt in nested_result["tokens"])
                    self.process_nested(doc, t, splits, used_tokens)

    # ------------------------------------------------------------------
    # MAIN DISPATCH METHOD
    # ------------------------------------------------------------------
    def split_sentence(self, sentence):
        doc = self.nlp(sentence)
        splits = []
        used_tokens = set()

        # ---- Gestione nominal conj ----------------------------------
        nominal_groups = self.expand_nominal_conj(doc)
        for group in nominal_groups:
            nouns      = group["nouns"]
            mods       = group["mods"]
            relcl_list = group["relcl"]

            # Marca tutti i token del gruppo come usati
            for noun in nouns:
                used_tokens.add(noun.i)
            for mod in mods:
                used_tokens.add(mod.i)
            for relcl in relcl_list:
                for t in relcl.subtree:
                    used_tokens.add(t.i)

            # Indici del gruppo nominale (nomi + modificatori + cc)
            nominal_idxs = set()
            for noun in nouns:
                nominal_idxs.add(noun.i)
                for ch in noun.children:
                    if ch.dep_ == "cc":
                        nominal_idxs.add(ch.i)
            for mod in mods:
                nominal_idxs.add(mod.i)
                for ch in mod.children:
                    if ch.dep_ == "cc":
                        nominal_idxs.add(ch.i)

            root = [t for t in doc if t.dep_ == "ROOT"][0]

            # Indici dei ccomp della ROOT da escludere dal predicato
            ccomp_idxs = set()
            for ch in root.children:
                if ch.dep_ == "ccomp":
                    ccomp_idxs.update(t.i for t in ch.subtree)

            # Indici degli advcl della ROOT da escludere dal predicato
            advcl_idxs = set()
            for ch in root.children:
                if ch.dep_ == "advcl":
                    advcl_idxs.update(t.i for t in ch.subtree)

            # Indici delle relcl
            relcl_idxs = {r.i for relcl in relcl_list for r in relcl.subtree}

            # Estrai il predicato escludendo gruppo nominale, relcl, ccomp e advcl
            predicate_tokens = [
                t for t in root.subtree
                if t.i not in nominal_idxs
                and t.dep_ not in {"punct"}
                and t.i not in relcl_idxs
                and t.i not in ccomp_idxs
                and t.i not in advcl_idxs
            ]
            predicate_tokens = sorted(predicate_tokens, key=lambda t: t.i)
            predicate_text = " ".join(t.text for t in predicate_tokens)
            used_tokens.update(t.i for t in predicate_tokens)

            # Combina ogni espansione nominale con il predicato
            adj_mods = [m for m in mods if m.pos_ == "ADJ"]
            for noun in nouns:
                noun_mods = sorted(
                    [ch for ch in noun.children if ch.dep_ in {"det", "amod"}],
                    key=lambda t: t.i
                )
                noun_text = " ".join(t.text for t in noun_mods) + " " + noun.text if noun_mods else noun.text

                if adj_mods:
                    for mod in adj_mods:
                        splits.append({
                            "type": "nominal_conj",
                            "subordinate": f"{mod.text} {noun_text} {predicate_text}"
                        })
                else:
                    splits.append({
                        "type": "nominal_conj",
                        "subordinate": f"{noun_text} {predicate_text}"
                    })

            # Propaga relcl a tutti i nomi del gruppo
            for relcl in relcl_list:
                for noun in nouns:
                    result = self.relcl_splitter.split(doc, relcl)
                    if result:
                        original_head = relcl.head.text
                        clause = result["subordinate"].replace(original_head, noun.text)
                        splits.append({
                            "type": "relcl_propagated",
                            "subordinate": clause
                        })

            # Processa i ccomp della ROOT separatamente
            for ch in root.children:
                if ch.dep_ == "ccomp" and ch.i not in used_tokens:
                    split_result = self.ccomp_splitter.split(doc, ch)
                    if split_result:
                        splits.append({
                            "type": split_result["type"],
                            "subordinate": split_result["subordinate"]
                        })
                        used_tokens.update(t.i for t in split_result["tokens"])
                        self.process_nested(doc, ch, splits, used_tokens)

            # Processa gli advcl della ROOT separatamente
            for ch in root.children:
                if ch.dep_ == "advcl" and ch.i not in used_tokens:
                    split_result = self.advcl_splitter.split(doc, ch)
                    if split_result:
                        splits.append({
                            "type": split_result["type"],
                            "subordinate": split_result["subordinate"]
                        })
                        used_tokens.update(t.i for t in split_result["tokens"])

        # ---- Gestione subordinate normali ---------------------------
        for token in doc:
            if token.dep_ in self.splitters and token.i not in used_tokens:

                actual_dep = token.dep_

                # Reindirizza relcl infinitive a acl
                if token.dep_ == "relcl":
                    has_to = any(
                        ch.dep_ == "aux" and ch.text.lower() == "to"
                        for ch in token.children
                    )
                    if has_to:
                        actual_dep = "acl"

                # Reindirizza conj di ccomp a ccomp
                if token.dep_ == "conj":
                    root = token
                    while root.dep_ == "conj":
                        root = root.head
                    if root.dep_ in {"ccomp", "xcomp"}:
                        actual_dep = "ccomp"

                split_result = self.splitters[actual_dep](doc, token)
                if split_result:
                    splits.append({
                        "type": split_result["type"],
                        "subordinate": split_result["subordinate"]
                    })
                    used_tokens.update(t.i for t in split_result["tokens"])

                    # Processa subordinate annidate per relcl, acl e ccomp
                    if actual_dep in {"ccomp", "relcl", "acl"}:
                        self.process_nested(doc, token, splits, used_tokens)

                    # Se è un pobj VERB, marca anche il "to" (head prep) come usato
                    if token.dep_ == "pobj" and token.pos_ == "VERB":
                        if token.head.dep_ == "prep" and token.head.text.lower() == "to":
                            used_tokens.add(token.head.i)

        # ---- Frase principale ---------------------------------------
        if not nominal_groups:
            main_tokens = [
                t for t in doc
                if t.i not in used_tokens and t.dep_ not in ["punct", "mark", "cc"]
            ]
            if main_tokens:
                main_text = " ".join(t.text for t in sorted(main_tokens, key=lambda t: t.i))
                splits.insert(0, {"type": "main", "subordinate": main_text.strip()})

        return [s["subordinate"] for s in splits]