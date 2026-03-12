import spacy

class ClauseSplitter:

    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)
        self.splitters = {
            "advcl": self.split_advcl,
            "acl": self.split_acl,
            "relcl": self.split_relcl,
        }

    def has_verb(self, token):
        """Check if the subtree contains a verb (VERB or AUX)."""
        return any(t.pos_ in ["VERB", "AUX"] for t in token.subtree)

    def find_name_modifiers(self, clause_tokens, noun):
        """Find adjectives and determiners that modify the noun and add them to the clause tokens."""
        for child in noun.children:
            if child.dep_ in ["amod", "det"]:
                clause_tokens.append(child)
        return clause_tokens

    def split_advcl(self, doc, token):

        if not self.has_verb(token):
            return None

        clause_tokens = list(token.subtree)
        clause_tokens = [t for t in clause_tokens if t.dep_ not in ["advmod", "mark", "punct"]]
        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)
        adv_clause = " ".join(t.text for t in clause_tokens)

        return {"type": "advcl", "subordinate": adv_clause.strip(), "tokens": clause_tokens}

    def split_acl(self, doc, token):
        noun = token.head
        clause_tokens = [noun]
        clause_tokens = self.find_name_modifiers(clause_tokens, noun)

        nested_idxs = set()
        for t in token.subtree:
            if t.dep_ in {"advcl", "relcl"} and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)

        for t in token.subtree:
            if t.dep_ == "punct":
                continue
            if t.i in nested_idxs:
                continue
            clause_tokens.append(t)

        clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
        acl_clause = " ".join(t.text for t in clause_tokens)
        clause_tokens = [t for t in clause_tokens if t.i > noun.i]

        return {"type": "acl", "subordinate": acl_clause.strip(), "tokens": clause_tokens}

    def split_relcl(self, doc, token):
        noun = token.head

        noun_mod_deps = {"det", "amod", "nummod", "poss", "compound"}
        noun_np = [noun]
        for ch in noun.children:
            if ch.dep_ in noun_mod_deps:
                noun_np.append(ch)
        noun_np = sorted(noun_np, key=lambda t: t.i)

        subtree_tokens = [t for t in token.subtree if t.dep_ != "punct"]

        rel_pron = None
        for ch in token.children:
            if ch.dep_ in {"nsubj", "nsubjpass"} and ch.text.lower() in {"who", "that", "which", "whom"}:
                rel_pron = ch
                break
        if rel_pron is None:
            for ch in noun.children:
                if ch.text.lower() in {"who", "that", "which", "whom"}:
                    rel_pron = ch
                    break

        rel_subj_heads = [
            t for t in subtree_tokens
            if t.dep_ in {"nsubj", "nsubjpass"} and t.text.lower() not in {"who", "that", "which", "whom"}
        ]
        rel_subj_with_mods = []
        for subj in rel_subj_heads:
            for ch in subj.children:
                if ch.dep_ in {"amod", "det"}:
                    rel_subj_with_mods.append(ch)
            rel_subj_with_mods.append(subj)
        rel_subj = sorted(rel_subj_with_mods, key=lambda t: t.i)

        rel_dobj = [t for t in subtree_tokens if t.dep_ in {"dobj", "obj"}]

        has_rel_pron = rel_pron is not None or any(
            t.text.lower() in {"that", "which", "whom", "who"}
            for t in subtree_tokens
        )

        excluded = set()
        excluded.update(t.i for t in rel_subj)
        excluded.update(t.i for t in rel_dobj)
        excluded.add(token.i)
        if rel_pron:
            excluded.add(rel_pron.i)

        other_tokens = [t for t in subtree_tokens if t.i not in excluded]

        # -------------------------------------------------------
        # CASE 1: relative pronoun is SUBJECT (who/which subject)
        # -------------------------------------------------------
        if rel_pron is not None and not rel_subj:
            clause_tokens = noun_np + [token] + other_tokens + rel_dobj
            for dobj in rel_dobj:
                self.find_name_modifiers(clause_tokens, dobj)

            seen = set()
            clause_ordered = []
            np_idxs = {t.i for t in noun_np}
            for t in sorted(clause_tokens, key=lambda t: (0 if t.i in np_idxs else 1, t.i)):
                if t.i not in seen:
                    clause_ordered.append(t)
                    seen.add(t.i)

            rel_clause = " ".join(t.text for t in clause_ordered)
            used_tokens = sorted([t for t in subtree_tokens if t.i > noun.i], key=lambda t: t.i)
            return {"type": "relcl", "subordinate": rel_clause.strip(), "tokens": used_tokens}

        # -------------------------------------------------------
        # CASE 2: explicit subject + explicit relative pronoun
        # -------------------------------------------------------
        elif rel_subj and has_rel_pron:
            rel_pron_as_obj = [
                t for t in rel_dobj
                if t.text.lower() in {"that", "which", "whom", "who"} or t.pos_ == "VERB"
            ]
            true_dobj = [t for t in rel_dobj if t not in rel_pron_as_obj]

            main_verb_idxs = set()
            for v in rel_pron_as_obj:
                if v.pos_ == "VERB":
                    main_verb_idxs.update(t.i for t in v.subtree)

            excluded.update(t.i for t in rel_pron_as_obj)
            excluded.update(main_verb_idxs)

            other_tokens = [t for t in subtree_tokens if t.i not in excluded]
            clause_tokens = rel_subj + [token] + other_tokens

            if not true_dobj and (rel_pron_as_obj or rel_pron is not None):
                clause_tokens += noun_np
            else:
                clause_tokens += true_dobj
                for dobj in true_dobj:
                    self.find_name_modifiers(clause_tokens, dobj)

            seen = set()
            clause_ordered = []
            for t in clause_tokens:
                if t.i not in seen:
                    clause_ordered.append(t)
                    seen.add(t.i)

            rel_clause = " ".join(t.text for t in clause_ordered)

            used_tokens = sorted(
                [t for t in subtree_tokens if t.i > noun.i and t.i not in main_verb_idxs],
                key=lambda t: t.i
            )
            return {"type": "relcl", "subordinate": rel_clause.strip(), "tokens": used_tokens}

        # -------------------------------------------------------
        # CASE 3: zero relative (no relative pronoun)
        # -------------------------------------------------------
        else:
            zero_subj = [t for t in subtree_tokens if t.dep_ in {"nsubj", "nsubjpass"}]
            clause_tokens = zero_subj + [token] + other_tokens + noun_np

            for subj in zero_subj:
                self.find_name_modifiers(clause_tokens, subj)

            seen = set()
            clause_ordered = []
            for t in clause_tokens:
                if t.i not in seen:
                    clause_ordered.append(t)
                    seen.add(t.i)

            rel_clause = " ".join(t.text for t in clause_ordered)
            used_tokens = sorted([t for t in subtree_tokens if t.i > noun.i], key=lambda t: t.i)
            return {"type": "relcl", "subordinate": rel_clause.strip(), "tokens": used_tokens}

    # -----------------------------
    # MAIN DISPATCH METHOD
    # -----------------------------
    def split_sentence(self, sentence):
        doc = self.nlp(sentence)
        splits = []
        used_tokens = set()

        for token in doc:
            if token.dep_ in self.splitters and token.i not in used_tokens:

                actual_dep = token.dep_
                if token.dep_ == "relcl":
                    has_to = any(ch.dep_ == "aux" and ch.text.lower() == "to" for ch in token.children)
                    if has_to:
                        actual_dep = "acl"

                split_result = self.splitters[actual_dep](doc, token)
                if split_result:
                    splits.append({
                        "type": split_result["type"],
                        "subordinate": split_result["subordinate"]
                    })
                    used_tokens.update(t.i for t in split_result["tokens"])

        main_tokens = [t for t in doc if t.i not in used_tokens and t.dep_ not in ["punct", "mark"]]
        if main_tokens:
            main_text = " ".join(t.text for t in sorted(main_tokens, key=lambda t: t.i))
            splits.insert(0, {"type": "main", "subordinate": main_text.strip()})

        return [s["subordinate"] for s in splits]


if __name__=='__main__':

    sentences = [

        # relcl + advcl
        "The book that John wrote became famous because it inspired many readers.",
        "The woman who looked happy danced when the music started.",

        # relcl + acl
        "The scientist who discovered the cure had a chance to save millions.",
        "The painting which the museum bought had a story to tell.",

        # advcl + acl
        "She had a decision to make because her boss resigned.",
        "He found a way to escape before the door closed.",

        # relcl + relcl
        "The man I met introduced me to the woman who won the prize.",
        "The book that she wrote inspired the student who solved the problem.",

        # tutti e tre insieme
        "The scientist who discovered the cure had a chance to publish because the journal accepted his work.",
        "The movie that we watched had a scene to remember because it moved everyone.",

    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print("\nSentence:", s)
        for split in splits:
            print(" -", split)