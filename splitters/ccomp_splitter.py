from splitters.base_splitter import BaseSplitter


class CcompSplitter(BaseSplitter):

    def split(self, doc, token):
        if token.dep_ == "xcomp":
            return None

        nested_idxs = self.build_nested_idxs(token, {"advcl", "relcl", "acl", "conj", "ccomp", "parataxis"}, include_cc=True)

        # Escludi cc diretti del token
        for ch in token.children:
            if ch.dep_ == "cc":
                nested_idxs.add(ch.i)

        # Gestisci "to + pobj VERB" come acl annidato
        for t in token.subtree:
            if t.dep_ == "pobj" and t.pos_ == "VERB":
                if t.head.dep_ == "prep" and t.head.text.lower() == "to":
                    nested_idxs.add(t.head.i)
                    nested_idxs.update(st.i for st in t.subtree)
            if t.dep_ == "prep" and t.text.lower() == "to":
                for ch in t.children:
                    if ch.dep_ == "pobj" and ch.pos_ == "VERB":
                        nested_idxs.add(t.i)
                        nested_idxs.update(st.i for st in ch.subtree)

        clause_tokens = sorted(
            [t for t in token.subtree if t.dep_ not in {"punct", "mark"} and t.i not in nested_idxs],
            key=lambda t: t.i
        )

        return {"type": "ccomp", "subordinate": self.build_clause_text(clause_tokens), "tokens": clause_tokens}