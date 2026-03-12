from splitters.base_splitter import BaseSplitter


class ConjSplitter(BaseSplitter):

    def split(self, doc, token):

        head = token.head

        # CASO C: conj nominale → gestito da expand_nominal_conj nell'orchestrator
        if token.pos_ in {"NOUN", "ADJ", "PROPN"} and head.pos_ in {"NOUN", "ADJ", "PROPN"}:
            return None

        # CASO B: conj verbale con soggetto esplicito
        has_own_subj = any(ch.dep_ in {"nsubj", "nsubjpass"} for ch in token.children)
        if has_own_subj:
            clause_tokens = [t for t in token.subtree if t.dep_ != "punct"]
            clause_tokens = sorted(clause_tokens, key=lambda t: t.i)
            clause_text = " ".join(t.text for t in clause_tokens)
            return {"type": "conj", "subordinate": clause_text.strip(), "tokens": clause_tokens}

        # CASO A: conj verbale senza soggetto → eredita soggetto dalla testa
        # Risali la catena di conj fino alla radice per trovare il soggetto
        root = token
        while root.dep_ == "conj":
            root = root.head
        inherited_subj = [t for t in root.children if t.dep_ in {"nsubj", "nsubjpass"}]

        # Escludi i conj annidati e i loro subtree
        nested_idxs = set()
        for t in token.subtree:
            if t.dep_ == "conj" and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)

        clause_tokens = [
            t for t in token.subtree
            if t.dep_ not in {"punct", "cc"} and t.i not in nested_idxs
        ]
        clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
        clause_text = " ".join(
            t.text for t in sorted(inherited_subj + clause_tokens, key=lambda t: t.i)
        )

        return {"type": "conj", "subordinate": clause_text.strip(), "tokens": clause_tokens}