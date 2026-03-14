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

    def _is_inside_relcl(self, token):
        """Returns True if token is nested inside a relcl subtree."""
        t = token.head
        while t != t.head:  # walk up to root
            if t.dep_ == "relcl":
                return True
            t = t.head
        return False

    def resolve_dep(self, token, doc=None):
        dep = token.dep_

        if dep == "relcl":
            if any(ch.dep_ == "aux" and ch.text.lower() == "to" for ch in token.children):
                return "acl"

        # Sopprime acl il cui head è un dobj annidato dentro una relcl:
        # spaCy a volte attacca il verbo principale come acl di un dobj interno,
        # generando clausole errate come "the exam asked for help".
        if dep == "acl" and token.head.dep_ == "dobj" and self._is_inside_relcl(token.head):
            return "SKIP"

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

        # Raccoglie i conj ricorsivamente (es. Marx→Engels→Weber)
        def collect_conj_nouns(t):
            result = []
            for ch in t.children:
                if ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"}:
                    result.append(ch)
                    result.extend(collect_conj_nouns(ch))
            return result

        # Raccoglie amod coordinati ricorsivamente (es. American→British)
        def collect_conj_adjs(t):
            result = []
            for ch in t.children:
                if ch.dep_ == "conj" and ch.pos_ == "ADJ":
                    result.append(ch)
                    result.extend(collect_conj_adjs(ch))
            return result

        for token in doc:
            if token.i in visited:
                continue

            # CASO 1: noun con conj nominali (es. "professors and students")
            if (token.dep_ not in {"conj", "punct", "cc"}
                    and token.pos_ in {"NOUN", "PROPN"}
                    and any(ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"} for ch in token.children)):

                visited.add(token.i)
                nouns = [token] + collect_conj_nouns(token)
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

            # CASO 2: noun con amod coordinati (es. "American and British professors")
            elif (token.dep_ not in {"conj", "punct", "cc"}
                    and token.pos_ in {"NOUN", "PROPN"}
                    and not any(ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"} for ch in token.children)):

                coord_amods = []
                for ch in token.children:
                    if ch.dep_ == "amod" and ch.pos_ == "ADJ":
                        conj_adjs = collect_conj_adjs(ch)
                        if conj_adjs:
                            coord_amods = [ch] + conj_adjs
                            break

                if not coord_amods:
                    continue

                visited.add(token.i)
                for adj in coord_amods:
                    visited.add(adj.i)

                relcl_list = [ch for ch in token.children if ch.dep_ == "relcl"]
                results.append({
                    "nouns": [token],          # un solo noun ripetuto per ogni adj
                    "mods": [],
                    "relcl": relcl_list,
                    "head_token": token,
                    "coord_amods": coord_amods, # amod coordinati da espandere
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
            coord_amods = group.get("coord_amods")  # solo per CASO 2

            # CASO 2: amod coordinati su un solo noun (es. "American and British professors")
            if coord_amods:
                noun = nouns[0]
                used_tokens.add(noun.i)
                for adj in coord_amods:
                    used_tokens.add(adj.i)
                    # segna anche i cc tra gli amod
                    for ch in adj.children:
                        if ch.dep_ == "cc":
                            used_tokens.add(ch.i)
                # segna cc figli del primo amod
                for ch in coord_amods[0].children:
                    if ch.dep_ == "cc":
                        used_tokens.add(ch.i)
                for relcl in relcl_list:
                    used_tokens.update(t.i for t in relcl.subtree)

                # Costruisce predicate_tokens escludendo noun, amod e relcl
                amod_idxs = {adj.i for adj in coord_amods}
                amod_idxs.update(ch.i for adj in coord_amods for ch in adj.children if ch.dep_ == "cc")
                excluded = {noun.i} | amod_idxs | {t.i for relcl in relcl_list for t in relcl.subtree}
                head_noun = group["head_token"]
                pred_root = head_noun.head if head_noun.head.pos_ in {"VERB", "AUX"} else root
                predicate_tokens = sorted(
                    [t for t in pred_root.subtree
                     if t.i not in excluded and t.dep_ not in {"punct", "cc"}],
                    key=lambda t: t.i
                )
                used_tokens.update(t.i for t in predicate_tokens)

                for adj in coord_amods:
                    all_tokens = sorted([adj, noun] + predicate_tokens, key=lambda t: t.i)
                    splits.append({"type": "nominal_conj", "subordinate": self.build_clause_text(all_tokens)})

                for dep in ("ccomp", "advcl", "parataxis"):
                    for ch in pred_root.children:
                        if ch.dep_ == dep and ch.i not in used_tokens:
                            self.dispatch(doc, ch, splits, used_tokens, recurse=(dep != "advcl"))
                continue

            for noun in nouns:
                used_tokens.add(noun.i)
                # Segna compound e cc di tutti i noun (inclusi quelli intermedi della catena)
                for ch in noun.children:
                    if ch.dep_ in {"compound", "cc"}:
                        used_tokens.add(ch.i)
            for mod in mods:
                used_tokens.add(mod.i)
            for relcl in relcl_list:
                used_tokens.update(t.i for t in relcl.subtree)

            nominal_idxs = set()
            for noun in nouns:
                nominal_idxs.add(noun.i)
                nominal_idxs.update(ch.i for ch in noun.children if ch.dep_ in {"cc", "compound"})
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
            head_det = [ch for ch in head_noun.children if ch.dep_ == "det"]
            for noun in nouns:
                noun_mods = sorted(
                    [ch for ch in noun.children
                     if ch.dep_ in {"det", "amod", "compound"} and ch.i not in adj_mod_idxs],
                    key=lambda t: t.i
                )
                # Eredita det dal head_noun se il noun non ne ha uno proprio
                if not any(m.dep_ == "det" for m in noun_mods):
                    noun_mods = sorted(head_det + noun_mods, key=lambda t: t.i)

                # Determina il range posizionale "di proprietà" di questo noun:
                # da noun.i fino al noun successivo del gruppo (escluso), oppure fine frase
                noun_positions = sorted(n.i for n in nouns)
                noun_pos_idx = noun_positions.index(noun.i)
                pos_start = noun.i
                pos_end = noun_positions[noun_pos_idx + 1] if noun_pos_idx + 1 < len(noun_positions) else len(doc)

                # Indici dei compound/det/amod diretti di altri noun: appartengono
                # sempre al loro noun, mai al territorio di questo
                other_direct_mods = set()
                for other_noun in nouns:
                    if other_noun.i != noun.i:
                        other_direct_mods.add(other_noun.i)
                        for ch in other_noun.children:
                            if ch.dep_ in {"compound", "det", "amod"}:
                                other_direct_mods.add(ch.i)

                # Indici di tutti i token appartenenti alle relcl del gruppo:
                # vengono gestiti separatamente e non devono mai finire in noun_tokens
                relcl_idxs = set()
                for relcl in relcl_list:
                    relcl_idxs.update(t.i for t in relcl.subtree)

                # Token del sottoalbero degli altri noun nel range posizionale di questo noun
                # vanno inclusi (es. "of the Rings" per "Lord"), ma solo se non sono
                # compound/det/amod diretti di un altro noun (es. "Joe" non va a "Obama")
                # e non appartengono a una relcl (es. "who attended" non va a "students")
                other_subtree_in_range = set()
                other_subtree_out_range = set()
                for other_noun in nouns:
                    if other_noun.i != noun.i:
                        for t in other_noun.subtree:
                            if t.i in other_direct_mods or t.i in relcl_idxs or t.dep_ == "cc":
                                other_subtree_out_range.add(t.i)
                            elif pos_start <= t.i < pos_end:
                                other_subtree_in_range.add(t.i)
                            else:
                                other_subtree_out_range.add(t.i)

                # noun_tokens = proprio sottoalbero (compound, det, amod) +
                #               token degli altri noun nel proprio range posizionale
                noun_tokens = sorted(
                    set(noun_mods + [noun]) | {doc[i] for i in other_subtree_in_range},
                    key=lambda t: t.i
                )
                # Dal predicato escludiamo solo i token fuori dal nostro range posizionale
                filtered_pred = [t for t in predicate_tokens if t.i not in other_subtree_out_range and t.i not in other_subtree_in_range]

                if adj_mods:
                    for mod in adj_mods:
                        all_tokens = sorted(filtered_pred + noun_tokens + [mod], key=lambda t: t.i)
                        splits.append({"type": "nominal_conj", "subordinate": self.build_clause_text(all_tokens)})
                else:
                    all_tokens = sorted(filtered_pred + noun_tokens, key=lambda t: t.i)
                    splits.append({"type": "nominal_conj", "subordinate": self.build_clause_text(all_tokens)})

            for relcl in relcl_list:
                for noun in nouns:
                    result = self.relcl_splitter.split(doc, relcl)
                    if result:
                        # Ricalcola noun_text per questo noun specifico
                        noun_mods_rel = sorted(
                            [ch for ch in noun.children
                             if ch.dep_ in {"det", "amod", "compound"} and ch.i not in adj_mod_idxs],
                            key=lambda t: t.i
                        )
                        if not any(m.dep_ == "det" for m in noun_mods_rel):
                            noun_mods_rel = sorted(head_det + noun_mods_rel, key=lambda t: t.i)
                        noun_text = (" ".join(t.text for t in noun_mods_rel) + " " + noun.text) if noun_mods_rel else noun.text

                        relcl_head_np_idxs = {relcl.head.i} | {
                            ch.i for ch in relcl.head.children
                            if ch.dep_ in {"det", "amod", "nummod", "poss", "compound"}
                        }
                        # Escludi anche pronomi relativi (who/that/which/whom)
                        rel_pron_idxs = {
                            ch.i for ch in relcl.children
                            if ch.text.lower() in {"who", "that", "which", "whom"}
                        }
                        relcl_body = " ".join(
                            t.text for t in sorted(result["tokens"], key=lambda t: t.i)
                            if t.i not in relcl_head_np_idxs
                            and t.i not in rel_pron_idxs
                            and t.dep_ not in {"punct"}
                        )
                        # Usa noun_text (con det ereditato) invece di noun.text
                        if adj_mods:
                            for mod in adj_mods:
                                clause = f"{mod.text} {noun_text} {relcl_body}"
                                splits.append({"type": "relcl_propagated", "subordinate": clause})
                        else:
                            clause = f"{noun_text} {relcl_body}"
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
            # I noun head di relcl top-level (attaccate direttamente a un token
            # che è figlio della ROOT) vengono marcati in used_tokens dalla relcl,
            # ma devono comunque apparire nella clausola principale.
            # Includiamo solo quelli direttamente rilevanti per la main clause,
            # escludendo noun che sono nsubj interni alla relcl stessa.
            relcl_heads = set()
            for token in doc:
                if token.dep_ == "relcl":
                    head = token.head
                    # Solo se il head è direttamente figlio della ROOT (nsubj, dobj, ecc.)
                    # e non è esso stesso dentro un'altra relcl
                    if head.head == root or head == root:
                        relcl_heads.add(head.i)
                        for ch in head.children:
                            if ch.dep_ in {"det", "amod", "compound"}:
                                relcl_heads.add(ch.i)

            main_tokens = [
                t for t in doc
                if (t.i not in used_tokens or t.i in relcl_heads)
                and t.dep_ not in {"punct", "mark", "cc"}
            ]
            if main_tokens:
                splits.insert(0, {"type": "main", "subordinate": self.build_clause_text(main_tokens)})

        return [s["subordinate"] for s in splits]