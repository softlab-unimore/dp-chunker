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
            nested_idxs = self.build_nested_idxs(token, {"ccomp", "relcl", "acl", "advcl"}, include_cc=True)
            clause_tokens = sorted(
                [t for t in token.subtree if t.dep_ not in {"punct", "cc"} and t.i not in nested_idxs],
                key=lambda t: t.i
            )
            return {"type": "conj", "subordinate": self.build_clause_text(clause_tokens), "tokens": clause_tokens}

        # CASO A: conj verbale senza soggetto → eredita soggetto dalla testa
        root = token
        while root.dep_ == "conj":
            root = root.head
        inherited_subj = [t for t in root.children if t.dep_ in {"nsubj", "nsubjpass"}]

        nested_idxs = self.build_nested_idxs(token, {"conj", "ccomp", "relcl", "acl", "advcl"}, include_cc=True)
        clause_tokens = sorted(
            set(t for t in token.subtree if t.dep_ not in {"punct", "cc"} and t.i not in nested_idxs),
            key=lambda t: t.i
        )
        clause_text = self.build_clause_text(inherited_subj + clause_tokens)

        return {"type": "conj", "subordinate": clause_text.strip(), "tokens": clause_tokens}