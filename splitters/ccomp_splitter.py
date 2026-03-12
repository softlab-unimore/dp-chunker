from splitters.base_splitter import BaseSplitter


class CcompSplitter(BaseSplitter):

    def split(self, doc, token):

        # Ignora xcomp completamente
        if token.dep_ == "xcomp":
            return None

        # Escludi subordinate annidate e i loro subtree
        nested_idxs = set()
        for t in token.subtree:
            if t.dep_ in {"advcl", "relcl", "acl", "conj", "ccomp"} and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)
                for ch in t.children:
                    if ch.dep_ == "cc":
                        nested_idxs.add(ch.i)

        # Escludi cc diretti del token che introducono un conj annidato
        for ch in token.children:
            if ch.dep_ == "cc":
                nested_idxs.add(ch.i)

        clause_tokens = [
            t for t in token.subtree
            if t.dep_ not in {"punct", "mark"}
            and t.i not in nested_idxs
        ]
        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)
        clause_text = " ".join(t.text for t in clause_tokens)

        return {"type": "ccomp", "subordinate": clause_text.strip(), "tokens": clause_tokens}