from splitters.base_splitter import BaseSplitter


class CcompSplitter(BaseSplitter):

    def split(self, doc, token):

        if token.dep_ == "xcomp":
            return None

        nested_idxs = set()
        for t in token.subtree:
            if t.dep_ in {"advcl", "relcl", "acl", "conj", "ccomp"} and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)
                for ch in t.children:
                    if ch.dep_ == "cc":
                        nested_idxs.add(ch.i)

            if t.dep_ == "pobj" and t.pos_ == "VERB":
                if t.head.dep_ == "prep" and t.head.text.lower() == "to":
                    nested_idxs.add(t.head.i)
                    nested_idxs.update(st.i for st in t.subtree)

            if t.dep_ == "prep" and t.text.lower() == "to":
                for ch in t.children:
                    if ch.dep_ == "pobj" and ch.pos_ == "VERB":
                        nested_idxs.add(t.i)
                        nested_idxs.update(st.i for st in ch.subtree)

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

        # Restituisce SOLO i token usati nel testo, non tutto il subtree
        # così process_nested può processare i token in nested_idxs
        return {"type": "ccomp", "subordinate": clause_text.strip(), "tokens": clause_tokens}