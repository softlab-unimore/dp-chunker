import spacy

from splitters.base_splitter import BaseSplitter
from splitters.advcl_splitter import AdvclSplitter
from splitters.acl_splitter import AclSplitter
from splitters.relcl_splitter import RelclSplitter
from splitters.conj_splitter import ConjSplitter
from splitters.ccomp_splitter import CcompSplitter
from splitters.parataxis_splitter import ParataxisSplitter


class ClauseSplitter(BaseSplitter):

    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)
        self.advcl_splitter     = AdvclSplitter(self.nlp)
        self.acl_splitter       = AclSplitter(self.nlp)
        self.relcl_splitter     = RelclSplitter(self.nlp)
        self.conj_splitter      = ConjSplitter(self.nlp)
        self.ccomp_splitter     = CcompSplitter(self.nlp)
        self.parataxis_splitter = ParataxisSplitter(self.nlp)

        self.splitters = {
            "advcl":    lambda doc, token: self.advcl_splitter.split(doc, token),
            "acl":      lambda doc, token: self.acl_splitter.split(doc, token),
            "relcl":    lambda doc, token: self.relcl_splitter.split(doc, token),
            "conj":     lambda doc, token: self.conj_splitter.split(doc, token),
            "ccomp":    lambda doc, token: self.ccomp_splitter.split(doc, token),
            "parataxis":lambda doc, token: self.parataxis_splitter.split(doc, token),
            "pobj":     lambda doc, token: (
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
    # RESOLVE ACTUAL DEP (redirections)
    # ------------------------------------------------------------------
    def resolve_dep(self, token, doc=None):
        dep = token.dep_

        if dep == "relcl":
            if any(ch.dep_ == "aux" and ch.text.lower() == "to" for ch in token.children):
                return "acl"

        if dep == "conj":
            root = token
            while root.dep_ == "conj":
                root = root.head
            if root.dep_ in {"ccomp", "xcomp"}:
                return "ccomp"

        # Redirect disguised parataxis (ccomp separated by : or ;)
        if dep == "ccomp" and doc is not None:
            if self.parataxis_splitter.is_disguised_parataxis(doc, token):
                return "parataxis"

        return dep

    # ------------------------------------------------------------------
    # DISPATCH SPLIT + APPEND
    # ------------------------------------------------------------------
    def dispatch(self, doc, token, splits, used_tokens, recurse=True):
        actual_dep = self.resolve_dep(token, doc)
        if actual_dep not in self.splitters:
            return

        result = self.splitters[actual_dep](doc, token)
        if not result:
            return

        splits.append({"type": result["type"], "subordinate": result["subordinate"]})
        used_tokens.update(t.i for t in result["tokens"])

        if recurse and actual_dep in {"ccomp", "relcl", "acl", "conj", "parataxis"}:
            self.process_nested(doc, token, splits, used_tokens)

        if token.dep_ == "pobj" and token.pos_ == "VERB":
            if token.head.dep_ == "prep" and token.head.text.lower() == "to":
                used_tokens.add(token.head.i)

    # ------------------------------------------------------------------
    # PROCESS NESTED SUBORDINATES
    # ------------------------------------------------------------------
    def process_nested(self, doc, token, splits, used_tokens):
        for t in token.subtree:
            if t.i == token.i or t.i in used_tokens:
                continue
            if self.resolve_dep(t, doc) in self.splitters:
                self.dispatch(doc, t, splits, used_tokens, recurse=True)

    # ------------------------------------------------------------------
    # EXPAND NOMINAL CONJ
    # ------------------------------------------------------------------
    def expand_nominal_conj(self, doc):
        results = []
        visited = set()

        for token in doc:
            if (token.dep_ not in {"conj", "punct", "cc"}
                    and token.pos_ in {"NOUN", "PROPN"}
                    and any(ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"} for ch in token.children)):

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
                        shared_mods += [c for c in ch.children if c.dep_ == "conj"]

                relcl_list = [ch for ch in token.children if ch.dep_ == "relcl"]

                results.append({
                    "nouns": nouns,
                    "mods": shared_mods,
                    "relcl": relcl_list,
                    "head_token": token,
                })

        return results

    # ------------------------------------------------------------------
    # MAIN DISPATCH METHOD
    # ------------------------------------------------------------------
    def split_sentence(self, sentence):
        doc = self.nlp(sentence)
        splits = []
        used_tokens = set()

        root = next(t for t in doc if t.dep_ == "ROOT")

        # ---- Gestione nominal conj ----------------------------------
        nominal_groups = self.expand_nominal_conj(doc)
        for group in nominal_groups:
            nouns      = group["nouns"]
            mods       = group["mods"]
            relcl_list = group["relcl"]

            for noun in nouns:
                used_tokens.add(noun.i)
            for mod in mods:
                used_tokens.add(mod.i)
            for relcl in relcl_list:
                used_tokens.update(t.i for t in relcl.subtree)

            nominal_idxs = set()
            for noun in nouns:
                nominal_idxs.add(noun.i)
                nominal_idxs.update(ch.i for ch in noun.children if ch.dep_ == "cc")
            for mod in mods:
                nominal_idxs.add(mod.i)
                nominal_idxs.update(ch.i for ch in mod.children if ch.dep_ == "cc")

            # Usa il verbo testa del gruppo nominale come base del predicato
            # (può essere diverso dalla ROOT se il gruppo è soggetto di una subordinata)
            head_noun = group["head_token"]
            pred_root = head_noun.head if head_noun.head.pos_ in {"VERB", "AUX"} else root

            excluded_from_pred = (
                nominal_idxs
                | {r.i for relcl in relcl_list for r in relcl.subtree}
                | self.collect_subtree_idxs(pred_root, "ccomp")
                | self.collect_subtree_idxs(pred_root, "advcl")
                | self.collect_subtree_idxs(pred_root, "parataxis")
                | self.collect_subtree_idxs(pred_root, "relcl")
            )

            predicate_tokens = sorted(
                [t for t in pred_root.subtree if t.i not in excluded_from_pred and t.dep_ != "punct"],
                key=lambda t: t.i
            )
            predicate_text = self.build_clause_text(predicate_tokens)
            used_tokens.update(t.i for t in predicate_tokens)

            adj_mods = [m for m in mods if m.pos_ == "ADJ"]
            adj_mod_idxs = {m.i for m in adj_mods}
            for noun in nouns:
                noun_mods = sorted(
                    [ch for ch in noun.children
                     if ch.dep_ in {"det", "amod"} and ch.i not in adj_mod_idxs],
                    key=lambda t: t.i
                )
                noun_text = (" ".join(t.text for t in noun_mods) + " " + noun.text) if noun_mods else noun.text

                if adj_mods:
                    for mod in adj_mods:
                        splits.append({"type": "nominal_conj", "subordinate": f"{mod.text} {noun_text} {predicate_text}"})
                else:
                    splits.append({"type": "nominal_conj", "subordinate": f"{noun_text} {predicate_text}"})

            for relcl in relcl_list:
                for noun in nouns:
                    result = self.relcl_splitter.split(doc, relcl)
                    if result:
                        # Ricostruisci la clausola sostituendo il noun_np della relcl
                        # con il noun corretto (+ adj_mod se presente)
                        # Usa i token della relcl escludendo il noun_np originale
                        relcl_head_np_idxs = {relcl.head.i} | {
                            ch.i for ch in relcl.head.children
                            if ch.dep_ in {"det", "amod", "nummod", "poss", "compound"}
                        }
                        relcl_body = " ".join(
                            t.text for t in sorted(result["tokens"], key=lambda t: t.i)
                            if t.i not in relcl_head_np_idxs and t.dep_ not in {"punct"}
                        )
                        if adj_mods:
                            for mod in adj_mods:
                                clause = f"{mod.text} {noun.text} {relcl_body}"
                                splits.append({"type": "relcl_propagated", "subordinate": clause})
                        else:
                            clause = f"{noun.text} {relcl_body}"
                            splits.append({"type": "relcl_propagated", "subordinate": clause})

            for dep in ("ccomp", "advcl", "parataxis"):
                for ch in pred_root.children:
                    if ch.dep_ == dep and ch.i not in used_tokens:
                        self.dispatch(doc, ch, splits, used_tokens, recurse=(dep != "advcl"))

            # Se pred_root != ROOT globale, processa la ROOT come clausola separata
            if pred_root != root and root.i not in used_tokens:
                ccomp_idxs = self.collect_subtree_idxs(root, "ccomp")
                advcl_idxs = self.collect_subtree_idxs(root, "advcl")
                root_tokens = sorted(
                    [t for t in root.subtree
                     if t.i not in used_tokens
                     and t.i not in ccomp_idxs
                     and t.i not in advcl_idxs
                     and t.dep_ not in {"punct", "mark", "cc"}],
                    key=lambda t: t.i
                )
                if root_tokens:
                    splits.append({"type": "main", "subordinate": self.build_clause_text(root_tokens)})
                    used_tokens.update(t.i for t in root_tokens)
                for dep in ("ccomp", "advcl", "parataxis"):
                    for ch in root.children:
                        if ch.dep_ == dep and ch.i not in used_tokens:
                            self.dispatch(doc, ch, splits, used_tokens, recurse=(dep != "advcl"))

        # ---- Gestione subordinate normali ---------------------------
        for token in doc:
            if token.dep_ in self.splitters and token.i not in used_tokens:
                self.dispatch(doc, token, splits, used_tokens)

        # ---- Frase principale ---------------------------------------
        if not nominal_groups:
            main_tokens = [
                t for t in doc
                if t.i not in used_tokens and t.dep_ not in {"punct", "mark", "cc"}
            ]
            if main_tokens:
                splits.insert(0, {"type": "main", "subordinate": self.build_clause_text(main_tokens)})

        return [s["subordinate"] for s in splits]